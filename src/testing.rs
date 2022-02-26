use crate::interpreter::Interpreter;
use crate::reader::read;
use crate::value::RuntimeValue;

const EXPECTED_STARTING_SCOPE_LEN: usize = 1;

pub fn run_eval_test(test_cases: &[(&str, RuntimeValue)]) {
    let mut has_err = false;
    for (input, expected) in test_cases {
        let forms = match read(input) {
            Ok(forms) => forms,
            Err(err) => {
                has_err = true;
                let context = err.context(input);
                println!(
                    "error reading `{}`: {} while reading {}",
                    input, err, context
                );
                continue;
            }
        };

        let mut interpreter = Interpreter::default();
        let mut final_result: Option<RuntimeValue> = None;
        let original_scope_len = interpreter.scopes.len();
        assert!(original_scope_len == EXPECTED_STARTING_SCOPE_LEN);
        // assert!(interpreter.apply_stack.is_empty());
        for form in &forms {
            match interpreter.analyze_and_evaluate(form) {
                Ok(result) => {
                    final_result = Some(result);
                }
                Err(e) => {
                    has_err = true;
                    println!(
                        "failure: evaluating `{}` from `{}` should give `{}` but errored: {}",
                        form, input, expected, e
                    );
                    break;
                }
            }
        }
        if has_err {
            continue;
        }
        assert!(interpreter.scopes.len() == original_scope_len);
        // assert!(interpreter.apply_stack.is_empty());
        if let Some(final_result) = final_result {
            if final_result != *expected {
                has_err = true;
                println!(
                    "failure: evaluating `{}` should give `{}` but got: {}",
                    input, expected, final_result
                );
            }
        }
    }
    assert!(!has_err);
}
