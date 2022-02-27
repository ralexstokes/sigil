use crate::interpreter::Interpreter;
use crate::value::RuntimeValue;

const EXPECTED_STARTING_SCOPE_LEN: usize = 0;

pub fn run_eval_test(test_cases: &[(&str, RuntimeValue)]) {
    let mut has_err = false;
    for (input, expected) in test_cases {
        let mut interpreter = Interpreter::default();
        let mut final_result: Option<RuntimeValue> = None;
        let original_scope_len = interpreter.scopes.len();
        assert!(original_scope_len == EXPECTED_STARTING_SCOPE_LEN);
        // assert!(interpreter.apply_stack.is_empty());
        match interpreter.interpret(input) {
            Ok(result) => {
                final_result = Some(result.last().unwrap().clone());
            }
            Err(e) => {
                has_err = true;
                println!(
                    "failure: evaluating `{}`; should give `{}` but errored: {}",
                    input, expected, e
                );
                break;
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
