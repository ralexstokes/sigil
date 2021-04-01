use crate::interpreter::{EvaluationError, PrimitiveEvaluationError};
use crate::value::Value;

pub fn plus(args: &[Value]) -> Result<Value, EvaluationError> {
    args.iter()
        .try_fold(i64::default(), |acc, x| match x {
            &Value::Number(n) => acc.checked_add(n).ok_or_else(|| {
                EvaluationError::Primitve(PrimitiveEvaluationError::Failure(
                    "overflow detected".to_string(),
                ))
            }),
            _ => Err(EvaluationError::Primitve(
                PrimitiveEvaluationError::Failure("plus only takes number arguments".to_string()),
            )),
        })
        .map(Value::Number)
}

pub fn subtract(args: &[Value]) -> Result<Value, EvaluationError> {
    match args.len() {
        0 => Err(EvaluationError::Primitve(
            PrimitiveEvaluationError::Failure("subtract needs more than 0 args".to_string()),
        )),
        1 => match &args[0] {
            &Value::Number(first) => first
                .checked_neg()
                .ok_or_else(|| {
                    EvaluationError::Primitve(PrimitiveEvaluationError::Failure(
                        "negation failed".to_string(),
                    ))
                })
                .map(Value::Number),
            _ => Err(EvaluationError::Primitve(
                PrimitiveEvaluationError::Failure(
                    "negation requires an integer argument".to_string(),
                ),
            )),
        },
        _ => {
            let first_value = &args[0];
            let rest_values = &args[1..];
            match first_value {
                &Value::Number(first) => rest_values
                    .iter()
                    .try_fold(first, |acc, x| match x {
                        &Value::Number(next) => acc.checked_sub(next).ok_or_else(|| {
                            EvaluationError::Primitve(PrimitiveEvaluationError::Failure(
                                "underflow detected".to_string(),
                            ))
                        }),
                        _ => Err(EvaluationError::Primitve(
                            PrimitiveEvaluationError::Failure(
                                "subtract only takes number arguments".to_string(),
                            ),
                        )),
                    })
                    .map(Value::Number),
                _ => Err(EvaluationError::Primitve(
                    PrimitiveEvaluationError::Failure(
                        "subtract only takes number arguments".to_string(),
                    ),
                )),
            }
        }
    }
}

pub fn multiply(args: &[Value]) -> Result<Value, EvaluationError> {
    args.iter()
        .try_fold(1 as i64, |acc, x| match x {
            &Value::Number(n) => acc.checked_mul(n).ok_or_else(|| {
                EvaluationError::Primitve(PrimitiveEvaluationError::Failure(
                    "overflow detected".to_string(),
                ))
            }),
            _ => Err(EvaluationError::Primitve(
                PrimitiveEvaluationError::Failure(
                    "multiply only takes number arguments".to_string(),
                ),
            )),
        })
        .map(Value::Number)
}

pub fn divide(args: &[Value]) -> Result<Value, EvaluationError> {
    match args.len() {
        0 => Err(EvaluationError::Primitve(
            PrimitiveEvaluationError::Failure("divide needs more than 0 args".to_string()),
        )),
        1 => match &args[0] {
            &Value::Number(first) => (1 as i64)
                .checked_div_euclid(first)
                .ok_or_else(|| {
                    EvaluationError::Primitve(PrimitiveEvaluationError::Failure(
                        "overflow detected".to_string(),
                    ))
                })
                .map(Value::Number),
            _ => Err(EvaluationError::Primitve(
                PrimitiveEvaluationError::Failure("divide requires number arguments".to_string()),
            )),
        },
        _ => {
            let first_value = &args[0];
            let rest_values = &args[1..];
            match first_value {
                &Value::Number(first) => rest_values
                    .iter()
                    .try_fold(first, |acc, x| match x {
                        &Value::Number(next) => acc.checked_div_euclid(next).ok_or_else(|| {
                            EvaluationError::Primitve(PrimitiveEvaluationError::Failure(
                                "overflow detected".to_string(),
                            ))
                        }),
                        _ => Err(EvaluationError::Primitve(
                            PrimitiveEvaluationError::Failure(
                                "divide only takes number arguments".to_string(),
                            ),
                        )),
                    })
                    .map(Value::Number),
                _ => Err(EvaluationError::Primitve(
                    PrimitiveEvaluationError::Failure(
                        "divide only takes number arguments".to_string(),
                    ),
                )),
            }
        }
    }
}
