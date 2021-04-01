use crate::interpreter::{EvaluationError, PrimitiveEvaluationError};
use crate::value::Value;

pub fn plus(args: &[Value]) -> Result<Value, EvaluationError> {
    args.iter()
        .map(|arg| match arg {
            Value::Number(n) => Ok(*n),
            _ => Err(EvaluationError::Primitve(
                PrimitiveEvaluationError::Failure("plus only takes number arguments".to_string()),
            )),
        })
        .collect::<Result<Vec<u64>, EvaluationError>>()
        .map(|args| Value::Number(args.iter().sum::<u64>()))
}
