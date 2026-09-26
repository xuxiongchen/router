//! Focused compatibility checks for the actual PyO3 extraction primitive used
//! by RenderBridge. Run with `--features pyo3/abi3-py38` to exercise the wheel's
//! limited-API path, and without it for the normal interpreter-specific path.
//! These do not reimplement or claim coverage of the private reply parser.

use std::borrow::Cow;

use pyo3::{
    exceptions::{PyTypeError, PyUnicodeEncodeError},
    prelude::*,
    types::PyString,
};

#[test]
fn cow_preserves_unicode_nul_and_owns_data_beyond_python_binding() {
    Python::initialize();
    let expected = "public-contract-中文-🦀-\0-e\u{301}";
    let owned = Python::attach(|py| {
        let source = PyString::new(py, expected);
        let extracted = source.extract::<Cow<'_, str>>().unwrap();
        assert_eq!(extracted.as_ref(), expected);
        // Both the borrowed CPython representation and the owned limited-API
        // representation must become an independent String before returning.
        extracted.into_owned()
    });
    assert_eq!(owned, expected);
    Python::attach(|py| {
        py.import("gc").unwrap().call_method0("collect").unwrap();
    });
    assert_eq!(owned.as_bytes(), expected.as_bytes());
}

#[test]
fn cow_rejects_non_strings_instead_of_coercing_them() {
    Python::initialize();
    Python::attach(|py| {
        for expression in [
            c"b'exact'",
            c"123",
            c"None",
            c"True",
            c"type('Pretend', (), {'__str__': lambda self: 'exact'})()",
        ] {
            let source = py.eval(expression, None, None).unwrap();
            let error = source.extract::<Cow<'_, str>>().unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
        }
    });
}

#[test]
fn cow_rejects_unpaired_surrogates_without_lossy_replacement() {
    Python::initialize();
    Python::attach(|py| {
        for expression in [c"'\\ud800'", c"'prefix-\\udfff-suffix'"] {
            let source = py.eval(expression, None, None).unwrap();
            let error = source.extract::<Cow<'_, str>>().unwrap_err();
            assert!(error.is_instance_of::<PyUnicodeEncodeError>(py));
        }
    });
}

#[test]
fn cow_preserves_str_subclass_and_utf8_byte_length_semantics() {
    Python::initialize();
    Python::attach(|py| {
        // PyString cast has always accepted str subclasses. Extraction reads
        // their actual value, not an overridden __str__ or normalization.
        let source = py
            .eval(
                c"type('Status', (str,), {'__str__': lambda self: 'unavailable'})('exact')",
                None,
                None,
            )
            .unwrap();
        let recognized = match source.extract::<Cow<'_, str>>().unwrap().as_ref() {
            "exact" => "exact",
            _ => "unexpected",
        };
        assert_eq!(recognized, "exact");

        for (characters, bytes) in [(85, 255), (86, 258)] {
            let text = "界".repeat(characters);
            let source = PyString::new(py, &text);
            let extracted = source.extract::<Cow<'_, str>>().unwrap();
            assert_eq!(extracted.len(), bytes);
            assert_eq!(extracted.as_ref(), text.as_str());
        }
    });
}
