use crate::interpreter::InterpreterBuilder;
use crate::reader::read;
use crate::value::Value;

pub fn run_eval_test(test_cases: &[(&str, Value)]) {
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

        let mut interpreter = InterpreterBuilder::default().build();
        let mut final_result: Option<Value> = None;
        for form in &forms {
            match interpreter.evaluate(form) {
                Ok(result) => {
                    final_result = Some(result);
                }
                Err(e) => {
                    has_err = true;
                    println!(
                        "failure: evaluating `{}` should give `{}` but errored: {}",
                        input, expected, e
                    );
                }
            }
        }
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
