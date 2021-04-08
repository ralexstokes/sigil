use crate::interpreter::{
    EvaluationError, EvaluationResult, Interpreter, ListEvaluationError, PrimitiveEvaluationError,
};
use crate::reader::read;
use crate::value::{
    atom_impl_into_inner, atom_with_value, list_with_values, vector_with_values, Value,
};
use itertools::join;
use std::fmt::Write;
use std::fs;

pub fn plus(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    args.iter()
        .try_fold(i64::default(), |acc, x| match x {
            &Value::Number(n) => acc.checked_add(n).ok_or_else(|| {
                EvaluationError::Primitive(PrimitiveEvaluationError::Failure(
                    "overflow detected".to_string(),
                ))
            }),
            _ => Err(EvaluationError::Primitive(
                PrimitiveEvaluationError::Failure("plus only takes number arguments".to_string()),
            )),
        })
        .map(Value::Number)
}

pub fn subtract(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    match args.len() {
        0 => Err(EvaluationError::Primitive(
            PrimitiveEvaluationError::Failure("subtract needs more than 0 args".to_string()),
        )),
        1 => match &args[0] {
            &Value::Number(first) => first
                .checked_neg()
                .ok_or_else(|| {
                    EvaluationError::Primitive(PrimitiveEvaluationError::Failure(
                        "negation failed".to_string(),
                    ))
                })
                .map(Value::Number),
            _ => Err(EvaluationError::Primitive(
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
                            EvaluationError::Primitive(PrimitiveEvaluationError::Failure(
                                "underflow detected".to_string(),
                            ))
                        }),
                        _ => Err(EvaluationError::Primitive(
                            PrimitiveEvaluationError::Failure(
                                "subtract only takes number arguments".to_string(),
                            ),
                        )),
                    })
                    .map(Value::Number),
                _ => Err(EvaluationError::Primitive(
                    PrimitiveEvaluationError::Failure(
                        "subtract only takes number arguments".to_string(),
                    ),
                )),
            }
        }
    }
}

pub fn multiply(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    args.iter()
        .try_fold(1 as i64, |acc, x| match x {
            &Value::Number(n) => acc.checked_mul(n).ok_or_else(|| {
                EvaluationError::Primitive(PrimitiveEvaluationError::Failure(
                    "overflow detected".to_string(),
                ))
            }),
            _ => Err(EvaluationError::Primitive(
                PrimitiveEvaluationError::Failure(
                    "multiply only takes number arguments".to_string(),
                ),
            )),
        })
        .map(Value::Number)
}

pub fn divide(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    match args.len() {
        0 => Err(EvaluationError::Primitive(
            PrimitiveEvaluationError::Failure("divide needs more than 0 args".to_string()),
        )),
        1 => match &args[0] {
            &Value::Number(first) => (1 as i64)
                .checked_div_euclid(first)
                .ok_or_else(|| {
                    EvaluationError::Primitive(PrimitiveEvaluationError::Failure(
                        "overflow detected".to_string(),
                    ))
                })
                .map(Value::Number),
            _ => Err(EvaluationError::Primitive(
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
                            EvaluationError::Primitive(PrimitiveEvaluationError::Failure(
                                "overflow detected".to_string(),
                            ))
                        }),
                        _ => Err(EvaluationError::Primitive(
                            PrimitiveEvaluationError::Failure(
                                "divide only takes number arguments".to_string(),
                            ),
                        )),
                    })
                    .map(Value::Number),
                _ => Err(EvaluationError::Primitive(
                    PrimitiveEvaluationError::Failure(
                        "divide only takes number arguments".to_string(),
                    ),
                )),
            }
        }
    }
}

pub fn pr(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    print!("{}", join(args, " "));
    Ok(Value::Nil)
}

pub fn prn(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    println!("{}", join(args, " "));
    Ok(Value::Nil)
}

pub fn list(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    Ok(list_with_values(args.iter().map(|arg| arg.clone())))
}

pub fn is_list(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        &Value::List(_) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn is_empty(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::List(elems) => Ok(Value::Bool(elems.is_empty())),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn count(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::List(elems) => Ok(Value::Number(elems.len() as i64)),
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn less(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Number(a) => match &args[1] {
            Value::Number(b) => Ok(Value::Bool(a < b)),
            _ => {
                return Err(EvaluationError::List(ListEvaluationError::Failure(
                    "incorrect argument".to_string(),
                )));
            }
        },
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn less_eq(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Number(a) => match &args[1] {
            Value::Number(b) => Ok(Value::Bool(a <= b)),
            _ => {
                return Err(EvaluationError::List(ListEvaluationError::Failure(
                    "incorrect argument".to_string(),
                )));
            }
        },
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn greater(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Number(a) => match &args[1] {
            Value::Number(b) => Ok(Value::Bool(a > b)),
            _ => {
                return Err(EvaluationError::List(ListEvaluationError::Failure(
                    "incorrect argument".to_string(),
                )));
            }
        },
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn greater_eq(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Number(a) => match &args[1] {
            Value::Number(b) => Ok(Value::Bool(a >= b)),
            _ => {
                return Err(EvaluationError::List(ListEvaluationError::Failure(
                    "incorrect argument".to_string(),
                )));
            }
        },
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn equal(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Number(a) => match &args[1] {
            Value::Number(b) => Ok(Value::Bool(a == b)),
            _ => {
                return Err(EvaluationError::List(ListEvaluationError::Failure(
                    "incorrect argument".to_string(),
                )));
            }
        },
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn read_string(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::String(s) => {
            let mut forms = read(s)?;
            if forms.len() != 1 {
                return Err(EvaluationError::List(ListEvaluationError::Failure(
                    "`read-string` only reads one form".to_string(),
                )));
            }
            Ok(forms.pop().unwrap())
        }
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn spit(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::String(path) => {
            let mut contents = String::new();
            let _ = write!(&mut contents, "{}", &args[1]);
            let _ = fs::write(path, contents).unwrap();
            Ok(Value::Nil)
        }
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn slurp(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::String(path) => {
            let contents = fs::read_to_string(path).unwrap();
            Ok(Value::String(contents))
        }
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn eval(interpreter: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }

    interpreter.evaluate(&args[0])
}

pub fn to_str(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    let mut result = String::new();
    for arg in args {
        let _ = write!(&mut result, "{}", arg.to_readable_string());
    }
    Ok(Value::String(result))
}

pub fn to_atom(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    Ok(atom_with_value(args[0].clone()))
}

pub fn is_atom(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        &Value::Atom(_) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn deref(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Atom(inner) => Ok(atom_impl_into_inner(inner)),
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn reset_atom(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Atom(inner) => {
            let value = args[1].clone();
            *inner.borrow_mut() = value.clone();
            Ok(value)
        }
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn swap_atom(interpreter: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() < 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Atom(cell) => match &args[1] {
            lambda @ Value::Fn(_) => {
                let mut inner = cell.borrow_mut();
                let original_value = inner.clone();
                let mut elems = vec![lambda.clone(), original_value];
                elems.extend_from_slice(&args[2..]);
                let form = list_with_values(elems);
                let new_value = interpreter.evaluate(&form)?;
                *inner = new_value.clone();
                Ok(new_value)
            }
            Value::Primitive(native_fn) => {
                let mut inner = cell.borrow_mut();
                let original_value = inner.clone();
                let mut fn_args = vec![original_value];
                fn_args.extend_from_slice(&args[2..]);
                let new_value = native_fn(interpreter, &fn_args)?;
                *inner = new_value.clone();
                Ok(new_value)
            }
            _ => {
                return Err(EvaluationError::List(ListEvaluationError::Failure(
                    "incorrect argument".to_string(),
                )));
            }
        },
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn cons(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[1] {
        Value::List(seq) => Ok(Value::List(seq.push_front(args[0].clone()))),
        Value::Vector(seq) => {
            let mut result = vec![args[0].clone()];
            for elem in seq {
                result.push(elem.clone());
            }
            Ok(list_with_values(result.into_iter()))
        }
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub fn concat(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    let mut elems = vec![];
    for arg in args {
        match arg {
            Value::List(seq) => elems.extend(seq.iter().cloned()),
            Value::Vector(seq) => elems.extend(seq.iter().cloned()),
            _ => {
                return Err(EvaluationError::List(ListEvaluationError::Failure(
                    "incorrect argument".to_string(),
                )));
            }
        }
    }
    Ok(list_with_values(elems))
}

pub fn vec(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::List(elems) => Ok(vector_with_values(elems.iter().cloned())),
        Value::Vector(elems) => Ok(vector_with_values(elems.iter().cloned())),
        Value::Nil => Ok(vector_with_values([].iter().cloned())),
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    }
}

pub const SOURCE: &str = r#"
(def! load-file (fn* [f] (eval (read-string (str "(do " (slurp f) " nil)")))))
"#;