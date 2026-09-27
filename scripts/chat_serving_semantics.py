"""Bounded public Chat acceptance helpers, NOT a serving/output parser.

Only consume already-parsed OpenAI JSON/SSE. Model-specific parsing remains in
vLLM. The sole executable tool is a bounded integer addition, with no eval/I/O.
"""

import copy
import json

TOOL = {
    "type": "function",
    "function": {
        "name": "synthetic_add",
        "description": "Add two integers using the local test calculator.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def execute_tools(message):
    """Validate the ENTIRE call set before executing any synthetic operation."""
    calls = message.get("tool_calls")
    require(
        isinstance(calls, list) and 1 <= len(calls) <= 4, "expected 1..4 tool calls"
    )
    seen, validated = set(), []
    for call in calls:
        ident = call.get("id")
        require(isinstance(ident, str) and 0 < len(ident) <= 256, "invalid tool ID")
        require(ident not in seen, "duplicate tool ID")
        seen.add(ident)
        function = call.get("function", {})
        require(
            call.get("type") == "function" and function.get("name") == "synthetic_add",
            "unapproved synthetic tool",
        )
        raw = function.get("arguments")
        require(isinstance(raw, str) and len(raw) <= 4096, "invalid argument size/type")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                require(key not in result, "duplicate argument key")
                result[key] = value
            return result

        args = json.loads(raw, object_pairs_hook=unique)
        require(isinstance(args, dict) and set(args) == {"a", "b"}, "argument keys")
        require(
            all(type(v) is int and abs(v) <= 1_000_000 for v in args.values()),
            "arguments must be bounded integers (not booleans)",
        )
        validated.append((ident, args))
    return [
        {
            "role": "tool",
            "tool_call_id": ident,
            "content": json.dumps({"result": args["a"] + args["b"]}),
        }
        for ident, args in validated
    ]


def followup(payload, message):
    result = copy.deepcopy(payload)
    result["messages"] += [copy.deepcopy(message), *execute_tools(message)]
    result["tool_choice"] = "none"
    return result


class ChatAccumulator:
    """Strict n=1 semantic checks; ignores irrelevant chunk partition/UUID/time."""

    def __init__(self):
        self.message = {"role": "assistant", "content": "", "reasoning": ""}
        self.calls = {}
        self.finish_reason = None
        self.usage = None
        self.errors = []
        self.done = False
        self.first = {}  # Seconds since request start, meaningful deltas only.

    def feed(self, event, elapsed=None):
        require(not self.done, "event after DONE")
        if "error" in event:
            self.errors.append(event["error"])
            return
        if event.get("usage") is not None:
            self.usage = event["usage"]
        choices = event.get("choices", [])
        require(len(choices) <= 1, "acceptance contract is n=1")
        for choice in choices:
            require(choice.get("index") == 0, "unexpected choice index")
            require(self.finish_reason is None, "choice after terminal choice")
            delta = choice.get("delta", choice.get("message", {}))
            require(delta.get("role", "assistant") == "assistant", "unexpected role")
            for key in ("content", "reasoning"):
                value = delta.get(key)
                if value:
                    require(isinstance(value, str), "non-text output")
                    self.message[key] += value
                    self.first.setdefault(key, elapsed)
            for position, call in enumerate(delta.get("tool_calls") or []):
                self.first.setdefault("tool", elapsed)
                index = call.get("index", position)
                require(
                    type(index) is int and 0 <= index < 4, "tool index out of bounds"
                )
                target = self.calls.setdefault(
                    index,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if call.get("id"):
                    require(
                        target["id"] in (None, call["id"]), "tool ID changed midstream"
                    )
                    target["id"] = call["id"]
                require(
                    call.get("type", "function") == "function", "unexpected tool type"
                )
                function = call.get("function") or {}
                # OpenAI function name is normally emitted once, not re-created.
                if function.get("name"):
                    target["function"]["name"] += function["name"]
                target["function"]["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]

    def result(self):
        require(not self.errors, f"stream contains error: {self.errors}")
        require(
            self.finish_reason in ("stop", "length", "tool_calls"),
            "missing/invalid finish",
        )
        require(isinstance(self.usage, dict), "missing usage")
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            require(
                type(self.usage.get(field)) is int and self.usage[field] >= 0,
                "invalid usage",
            )
        require(
            self.usage["total_tokens"]
            == self.usage["prompt_tokens"] + self.usage["completion_tokens"],
            "inconsistent usage",
        )
        result = copy.deepcopy(self.message)
        if self.calls:
            require(
                sorted(self.calls) == list(range(len(self.calls))), "tool index gap"
            )
            result["tool_calls"] = [self.calls[i] for i in sorted(self.calls)]
            require(
                all(c["id"] and c["function"]["name"] for c in result["tool_calls"]),
                "incomplete tool identity",
            )
        return {
            "message": result,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "first_meaningful_seconds": self.first,
        }


def decode_response(body, stream):
    accumulator = ChatAccumulator()
    if not stream:
        accumulator.feed(json.loads(body))
    else:
        # Decode only complete records: arbitrary HTTP byte boundaries are not
        # tokenizer boundaries and must never be independently UTF-8 decoded.
        text = body.decode("utf-8").replace("\r\n", "\n")
        for record in text.split("\n\n"):
            data = "\n".join(
                line[5:].lstrip(" ")
                for line in record.splitlines()
                if line.startswith("data:")
            )
            if not data:
                continue
            require(not accumulator.done, "data after DONE")
            if data == "[DONE]":
                accumulator.done = True
            else:
                accumulator.feed(json.loads(data))
        require(accumulator.done, "truncated SSE (no DONE)")
    return accumulator.result()


def semantic_cases(model):
    base = {
        "model": model,
        "temperature": 0,
        "seed": 17,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    simple = [{"role": "user", "content": "Reply with the word hello."}]
    yield "plain", {**base, "messages": simple}, "text"
    yield "parts", {
        **base,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Repeat: café 中文 "},
                    {"type": "text", "text": "🙂"},
                ],
            }
        ],
    }, "text"
    history = [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Hello Ada.", "reasoning": "A greeting."},
        {"role": "user", "content": "What is my name?"},
    ]
    yield "history", {**base, "messages": history}, "text"
    yield "thinking", {
        **base,
        "messages": simple,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": True},
    }, "reasoning"
    yield "reasoning-none", {
        **base,
        "messages": simple,
        "reasoning_effort": "none",
    }, "text"
    yield "json-object", {
        **base,
        "messages": [{"role": "user", "content": 'Return JSON {"answer":42}.'}],
        "response_format": {"type": "json_object"},
    }, "json"
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer", "enum": [42]}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    yield "json-schema", {
        **base,
        "messages": [{"role": "user", "content": "Return answer 42 as JSON."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": schema},
        },
    }, "schema"
    yield "structured-choice", {
        **base,
        "messages": [{"role": "user", "content": "Choose red or blue."}],
        "structured_outputs": {"choice": ["red", "blue"]},
    }, "choice"
    for choice in (
        "auto",
        "required",
        {"type": "function", "function": {"name": "synthetic_add"}},
    ):
        name = choice if isinstance(choice, str) else "named"
        yield "tool-" + name, {
            **base,
            "messages": [
                {
                    "role": "user",
                    "content": "Use synthetic_add to add 17 and 25. Do not calculate it yourself. Call the tool now.",
                }
            ],
            "tools": [copy.deepcopy(TOOL)],
            "tool_choice": choice,
        }, "tool"
    yield "tool-none", {
        **base,
        "messages": simple,
        "tools": [copy.deepcopy(TOOL)],
        "tool_choice": "none",
    }, "text"


def check_semantics(result, expected):
    message = result["message"]
    if expected == "tool":
        require(
            result["finish_reason"] in ("tool_calls", "stop"),
            "tool response was truncated",
        )
        outputs = execute_tools(message)
        require(
            all(json.loads(x["content"]) == {"result": 42} for x in outputs),
            "wrong generated arguments",
        )
    elif expected == "reasoning":
        require(
            bool(message["reasoning"]) and bool(message["content"]),
            "missing reasoning/content",
        )
    elif expected == "json":
        require(isinstance(json.loads(message["content"]), dict), "not a JSON object")
    elif expected == "schema":
        value = json.loads(message["content"])
        require(
            isinstance(value, dict)
            and set(value) == {"answer"}
            and type(value["answer"]) is int
            and value["answer"] == 42,
            "output schema violation",
        )
    elif expected == "choice":
        require(message["content"] in ("red", "blue"), "invalid structured choice")
    else:
        require(bool(message["content"]), "empty text answer")


def run_gpu_cases(validation):
    """Extend existing owned-cohort harness. One attempt per case, no retry-to-pass."""
    from pathlib import Path
    from render_bridge_gpu_validate import raw_request, save

    for label, overrides in (
        ("unknown-model", {"model": "cmb-deliberately-unserved"}),
        ("negative-length", {"max_tokens": -1}),
        ("zero-choices", {"n": 0}),
    ):

        def check_error(label=label, overrides=overrides):
            validation.idle()
            payload = {
                "model": validation.model,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
                **overrides,
            }
            raw = json.dumps(payload).encode()
            results = []
            for endpoint in (validation.workers[0], validation.args.router):
                status, _, body = raw_request(endpoint, "/v1/chat/completions", raw)
                results.append({"status": status, "response": json.loads(body)})
            save(validation.out / ("agent-error-" + label + ".json"), results)
            require(
                400 <= results[0]["status"] < 500
                and results[0]["status"] == results[1]["status"],
                "baseline request rejection status changed",
            )
            require(
                all(
                    isinstance(value["response"].get("error"), dict)
                    for value in results
                ),
                "baseline API error envelope lost",
            )
            validation.idle()
            return {
                "name": "agent-error-" + label,
                "status": "PASS",
                "worker_accepts": False,
                "error_preserved": True,
                "responses": results,
            }

        validation.case("agent-error-" + label, check_error)

    for case, request, expected in semantic_cases(validation.model):
        for stream in (False, True):
            name = "agent-" + case + ("-sse" if stream else "-json")
            payload = {**request, "stream": stream, "return_token_ids": True}
            if stream:
                payload["stream_options"] = {"include_usage": True}

            def run(name=name, payload=payload, expected=expected, stream=stream):
                route = "/v1/chat/completions"
                raw = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ).encode()
                direct_timing = {}
                status, headers, body = raw_request(
                    validation.workers[0], route, raw, timings=direct_timing
                )
                save(
                    validation.out / (name + ".direct.json"),
                    {
                        "status": status,
                        "headers": headers,
                        "body": body.decode(),
                        "timings": direct_timing,
                    },
                )
                # Unsupported official combinations are baseline errors, not
                # successful tool calls. Compare status and error payload.
                if status != 200:
                    routed_status, _, routed_body = raw_request(
                        validation.args.router, route, raw
                    )
                    require(
                        routed_status == status
                        and json.loads(routed_body) == json.loads(body),
                        "Router changed baseline error",
                    )
                    return {
                        "worker_accepts": False,
                        "error_preserved": True,
                        "name": name,
                        "status": "PASS",
                        "http_status": status,
                        "semantic_status": "UNSUPPORTED",
                    }
                _, ids, _ = validation.oracle(payload, name + "-direct")
                from render_bridge_gpu_validate import generation_tokens

                require(
                    generation_tokens(body, stream, True) == ids,
                    "direct input IDs differ",
                )
                direct = decode_response(body, stream)
                evidence = validation.routed(name, payload)
                routed = decode_response(
                    Path(validation.out / (name + ".response.bin")).read_bytes(), stream
                )
                check_semantics(direct, expected)
                check_semantics(routed, expected)
                if payload.get("tool_choice") == "none":
                    require(
                        not direct["message"].get("tool_calls")
                        and not routed["message"].get("tool_calls"),
                        "tool_choice=none produced tool calls",
                    )
                evidence.update(
                    direct_semantics=direct,
                    routed_semantics=routed,
                    direct_timings=direct_timing,
                    prepared_chat=False,
                    optimization_reason="raw_chat_only",
                )
                if expected == "tool":
                    # Use EACH execution's actual tool IDs/arguments/results.
                    for label, endpoint, first in (
                        ("direct", validation.workers[0], direct),
                        ("router", validation.args.router, routed),
                    ):
                        second = followup(payload, first["message"])
                        if label == "router":
                            second_name = name + "-followup"
                            validation.routed(second_name, second)
                            answer_body = Path(
                                validation.out / (second_name + ".response.bin")
                            ).read_bytes()
                        else:
                            second_raw, second_ids, _ = validation.oracle(
                                second, name + "-followup-direct"
                            )
                            code, _, answer_body = raw_request(
                                endpoint, route, second_raw
                            )
                            require(
                                code == 200
                                and generation_tokens(answer_body, stream, True)
                                == second_ids,
                                "direct followup failed or input IDs differ",
                            )
                        answer = decode_response(answer_body, stream)
                        require(
                            "42" in answer["message"]["content"]
                            and not answer["message"].get("tool_calls"),
                            "model did not use actual synthetic result",
                        )
                        evidence[label + "_tool_loop"] = {
                            "request": second,
                            "answer": answer,
                        }
                save(validation.out / (name + ".semantics.json"), evidence)
                return evidence

            validation.case(name, run)
