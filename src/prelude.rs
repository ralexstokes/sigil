use crate::interpreter::{
    EvaluationError, EvaluationResult, Interpreter, InterpreterError, ListEvaluationError,
    PrimitiveEvaluationError,
};
use crate::reader::read;
use crate::value::{
    atom_impl_into_inner, atom_with_value, exception, exception_into_thrown, list_with_values,
    map_with_values, set_with_values, var_impl_into_inner, vector_with_values, FnImpl, Value,
};
use itertools::Itertools;
use rpds::HashTrieSet as PersistentSet;
use rpds::List as PersistentList;
use rpds::Vector as PersistentVector;
use std::fmt::Write;
use std::io::{BufRead, Write as IOWrite};
use std::iter::FromIterator;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fs, io};

pub fn plus(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    args.iter()
        .try_fold(i64::default(), |acc, x| match *x {
            Value::Number(n) => acc.checked_add(n).ok_or_else(|| {
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
        1 => match args[0] {
            Value::Number(first) => first
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
            match *first_value {
                Value::Number(first) => rest_values
                    .iter()
                    .try_fold(first, |acc, x| match *x {
                        Value::Number(next) => acc.checked_sub(next).ok_or_else(|| {
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
        .try_fold(1_i64, |acc, x| match *x {
            Value::Number(n) => acc.checked_mul(n).ok_or_else(|| {
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
        1 => match args[0] {
            Value::Number(first) => 1_i64
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
            match *first_value {
                Value::Number(first) => rest_values
                    .iter()
                    .try_fold(first, |acc, x| match *x {
                        Value::Number(next) => acc.checked_div_euclid(next).ok_or_else(|| {
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
    let result = args.iter().map(|arg| arg.to_readable_string()).join(" ");
    print!("{}", result);
    io::stdout().flush().unwrap();
    Ok(Value::Nil)
}

pub fn prn(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    let result = args.iter().map(|arg| arg.to_readable_string()).join(" ");
    println!("{}", result);
    Ok(Value::Nil)
}

pub fn pr_str(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    let result = args.iter().map(|arg| arg.to_readable_string()).join(" ");
    Ok(Value::String(result))
}

pub fn print_(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    print!("{}", args.iter().format(" "));
    io::stdout().flush().unwrap();
    Ok(Value::Nil)
}

pub fn println(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    println!("{}", args.iter().format(" "));
    Ok(Value::Nil)
}

pub fn print_str(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    let mut result = String::new();
    write!(&mut result, "{}", args.iter().format(" ")).expect("can write to string");
    Ok(Value::String(result))
}

pub fn list(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    Ok(list_with_values(args.iter().cloned()))
}

pub fn is_list(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match args[0] {
        Value::List(_) => Ok(Value::Bool(true)),
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
        Value::Nil => Ok(Value::Bool(true)),
        Value::String(s) => Ok(Value::Bool(s.is_empty())),
        Value::List(elems) => Ok(Value::Bool(elems.is_empty())),
        Value::Vector(elems) => Ok(Value::Bool(elems.is_empty())),
        Value::Map(elems) => Ok(Value::Bool(elems.is_empty())),
        Value::Set(elems) => Ok(Value::Bool(elems.is_empty())),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn count(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Nil => Ok(Value::Number(0)),
        Value::String(s) => Ok(Value::Number(s.len() as i64)),
        Value::List(elems) => Ok(Value::Number(elems.len() as i64)),
        Value::Vector(elems) => Ok(Value::Number(elems.len() as i64)),
        Value::Map(elems) => Ok(Value::Number(elems.size() as i64)),
        Value::Set(elems) => Ok(Value::Number(elems.size() as i64)),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
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
            _ => Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            ))),
        },
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
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
            _ => Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            ))),
        },
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
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
            _ => Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            ))),
        },
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
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
            _ => Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            ))),
        },
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn equal(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    Ok(Value::Bool(args[0] == args[1]))
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
            match forms.len() {
                0 => Ok(Value::Nil),
                1 => Ok(forms.pop().unwrap()),
                _ => Err(EvaluationError::List(ListEvaluationError::Failure(
                    "`read-string` only reads one form".to_string(),
                ))),
            }
        }
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
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
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
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
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn eval(interpreter: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }

    interpreter.evaluate_in_global_scope(&args[0])
}

pub fn to_str(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() == 1 && matches!(&args[0], Value::Nil) {
        return Ok(Value::String("".to_string()));
    }
    let mut result = String::new();
    for arg in args {
        match arg {
            Value::String(s) => {
                write!(result, "{}", s).expect("can write to string");
            }
            _ => write!(result, "{}", arg.to_readable_string()).expect("can write to string"),
        }
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
    match args[0] {
        Value::Atom(_) => Ok(Value::Bool(true)),
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
        Value::Var(inner) => Ok(var_impl_into_inner(inner)),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
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
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
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
            Value::Fn(FnImpl {
                body,
                arity,
                level,
                variadic,
            }) => {
                let mut inner = cell.borrow_mut();
                let original_value = inner.clone();
                let mut elems = vec![original_value];
                elems.extend_from_slice(&args[2..]);
                let fn_args = PersistentList::from_iter(elems);
                let new_value =
                    interpreter.apply_fn_inner(body, *arity, *level, *variadic, fn_args, true)?;
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
            _ => Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            ))),
        },
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
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
            let mut inner = PersistentList::new();
            for elem in seq.iter().rev() {
                inner.push_front_mut(elem.clone());
            }
            inner.push_front_mut(args[0].clone());
            Ok(Value::List(inner))
        }
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
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
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn nth(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match args[1] {
        Value::Number(index) if index >= 0 => match &args[0] {
            Value::List(seq) => seq
                .iter()
                .nth(index as usize)
                .ok_or_else(|| {
                    EvaluationError::List(ListEvaluationError::Failure(
                        "collection does not have an element at this index".to_string(),
                    ))
                })
                .map(|elem| elem.clone()),
            Value::Vector(seq) => seq
                .iter()
                .nth(index as usize)
                .ok_or_else(|| {
                    EvaluationError::List(ListEvaluationError::Failure(
                        "collection does not have an element at this index".to_string(),
                    ))
                })
                .map(|elem| elem.clone()),
            _ => Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            ))),
        },
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn first(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::List(elems) => {
            if let Some(first) = elems.first() {
                Ok(first.clone())
            } else {
                Ok(Value::Nil)
            }
        }
        Value::Vector(elems) => {
            if let Some(first) = elems.first() {
                Ok(first.clone())
            } else {
                Ok(Value::Nil)
            }
        }
        Value::Nil => Ok(Value::Nil),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn rest(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::List(elems) => {
            if let Some(rest) = elems.drop_first() {
                Ok(Value::List(rest))
            } else {
                Ok(Value::List(PersistentList::new()))
            }
        }
        Value::Vector(elems) => {
            let mut result = PersistentList::new();
            for elem in elems.iter().skip(1).rev() {
                result.push_front_mut(elem.clone())
            }
            Ok(Value::List(result))
        }
        Value::Nil => Ok(Value::List(PersistentList::new())),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn ex_info(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::String(msg) => Ok(exception(msg, &args[1])),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn throw(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    let exc = match &args[0] {
        n @ Value::Nil => exception("", n),
        b @ Value::Bool(_) => exception("", b),
        n @ Value::Number(_) => exception("", n),
        s @ Value::String(_) => exception("", s),
        k @ Value::Keyword(..) => exception("", k),
        s @ Value::Symbol(..) => exception("", s),
        coll @ Value::List(_) => exception("", coll),
        coll @ Value::Vector(_) => exception("", coll),
        coll @ Value::Map(_) => exception("", coll),
        coll @ Value::Set(_) => exception("", coll),
        e @ Value::Exception(_) => e.clone(),
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )))
        }
    };
    Ok(exception_into_thrown(&exc))
}

pub fn apply(interpreter: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() < 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    let (last, prefix) = args.split_last().expect("has enough elements");
    let (first, middle) = prefix.split_first().expect("has enough elements");
    let fn_args = match last {
        Value::List(elems) => {
            let mut fn_args = Vec::with_capacity(middle.len() + elems.len());
            for elem in middle.iter().chain(elems.iter()) {
                fn_args.push(elem.clone())
            }
            fn_args
        }
        Value::Vector(elems) => {
            let mut fn_args = Vec::with_capacity(middle.len() + elems.len());
            for elem in middle.iter().chain(elems.iter()) {
                fn_args.push(elem.clone())
            }
            fn_args
        }
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )))
        }
    };
    match &first {
        Value::Fn(FnImpl {
            body,
            arity,
            level,
            variadic,
        }) => {
            let fn_args = PersistentList::from_iter(fn_args);
            interpreter.apply_fn_inner(body, *arity, *level, *variadic, fn_args, false)
        }
        Value::Primitive(native_fn) => native_fn(interpreter, &fn_args),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn map(interpreter: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    let fn_args: Vec<_> = match &args[1] {
        Value::List(elems) => elems.iter().collect(),
        Value::Vector(elems) => elems.iter().collect(),
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    };
    let mut result = PersistentList::new();
    match &args[0] {
        Value::Fn(FnImpl {
            body,
            arity,
            level,
            variadic,
        }) => {
            for arg in fn_args.into_iter().rev() {
                let mut wrapped_arg = PersistentList::new();
                wrapped_arg.push_front_mut(arg.clone());
                let mapped_arg = interpreter.apply_fn_inner(
                    body,
                    *arity,
                    *level,
                    *variadic,
                    wrapped_arg,
                    false,
                )?;
                result.push_front_mut(mapped_arg);
            }
        }
        Value::Primitive(native_fn) => {
            for arg in fn_args.into_iter().rev() {
                let mapped_arg = native_fn(interpreter, &[arg.clone()])?;
                result.push_front_mut(mapped_arg);
            }
        }
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )));
        }
    };
    Ok(Value::List(result))
}

pub fn is_nil(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Nil => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn is_true(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Bool(true) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn is_false(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Bool(false) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn is_symbol(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Symbol(..) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn to_symbol(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::String(name) => Ok(Value::Symbol(name.clone(), None)),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn to_keyword(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::String(name) => Ok(Value::Keyword(name.clone(), None)),
        k @ Value::Keyword(..) => Ok(k.clone()),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn is_keyword(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Keyword(..) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn to_vector(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    Ok(vector_with_values(args.iter().cloned()))
}

pub fn is_vector(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Vector(..) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn is_sequential(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::List(..) | Value::Vector(..) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn to_map(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() % 2 != 0 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "map needs an even number of arguments".to_string(),
        )));
    }
    Ok(map_with_values(args.iter().cloned().tuples()))
}

pub fn is_map(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Map(..) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn to_set(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Nil => Ok(Value::Set(PersistentSet::new())),
        Value::String(s) => Ok(set_with_values(
            s.chars().map(|c| Value::String(c.to_string())),
        )),
        Value::List(coll) => Ok(set_with_values(coll.iter().cloned())),
        Value::Vector(coll) => Ok(set_with_values(coll.iter().cloned())),
        Value::Map(coll) => Ok(set_with_values(coll.iter().map(|(k, v)| {
            let mut inner = PersistentVector::new();
            inner.push_back_mut(k.clone());
            inner.push_back_mut(v.clone());
            Value::Vector(inner)
        }))),
        s @ Value::Set(..) => Ok(s.clone()),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn is_set(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Set(..) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn assoc(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() < 3 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    if (args.len() - 1) % 2 != 0 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "assoc needs keys and values to pair".to_string(),
        )));
    }
    match &args[0] {
        Value::Map(map) => {
            let mut result = map.clone();
            for (key, val) in args.iter().skip(1).tuples() {
                result.insert_mut(key.clone(), val.clone());
            }
            Ok(Value::Map(result))
        }
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn dissoc(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.is_empty() {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Map(map) => {
            let mut result = map.clone();
            for key in args.iter().skip(1) {
                result.remove_mut(key);
            }
            Ok(Value::Map(result))
        }
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn get(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::Map(map) => {
            let result = if let Some(val) = map.get(&args[1]) {
                val.clone()
            } else {
                Value::Nil
            };
            Ok(result)
        }
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn does_contain(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Map(map) => {
            let contains = map.contains_key(&args[1]);
            Ok(Value::Bool(contains))
        }
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn to_keys(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    let result = match &args[0] {
        Value::Map(map) => {
            if map.is_empty() {
                Value::Nil
            } else {
                list_with_values(map.keys().cloned())
            }
        }
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )))
        }
    };
    Ok(result)
}

pub fn to_vals(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    let result = match &args[0] {
        Value::Map(map) => {
            if map.is_empty() {
                Value::Nil
            } else {
                list_with_values(map.values().cloned())
            }
        }
        _ => {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "incorrect argument".to_string(),
            )))
        }
    };
    Ok(result)
}

pub fn last(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::List(elems) => {
            if let Some(elem) = elems.last() {
                Ok(elem.clone())
            } else {
                Ok(Value::Nil)
            }
        }
        Value::Vector(elems) => {
            if let Some(elem) = elems.last() {
                Ok(elem.clone())
            } else {
                Ok(Value::Nil)
            }
        }
        Value::Nil => Ok(Value::Nil),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn is_string(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::String(..) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn is_number(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Number(..) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn is_fn(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Fn(..) | Value::Primitive(..) | Value::Macro(..) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn conj(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() < 2 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Nil => Ok(list_with_values(args[1..].iter().cloned())),
        Value::List(seq) => {
            let mut inner = seq.clone();
            for elem in &args[1..] {
                inner.push_front_mut(elem.clone());
            }
            Ok(Value::List(inner))
        }
        Value::Vector(seq) => {
            let mut inner = seq.clone();
            for elem in &args[1..] {
                inner.push_back_mut(elem.clone());
            }
            Ok(Value::Vector(inner))
        }
        Value::Map(seq) => {
            let mut inner = seq.clone();
            for elem in &args[1..] {
                match elem {
                    Value::Vector(kv) if kv.len() == 2 => {
                        let k = &kv[0];
                        let v = &kv[1];
                        inner.insert_mut(k.clone(), v.clone());
                    }
                    Value::Map(elems) => {
                        for (k, v) in elems {
                            inner.insert_mut(k.clone(), v.clone());
                        }
                    }
                    _ => {
                        return Err(EvaluationError::List(ListEvaluationError::Failure(
                            "incorrect argument".to_string(),
                        )))
                    }
                }
            }
            Ok(Value::Map(inner))
        }
        Value::Set(seq) => {
            let mut inner = seq.clone();
            for elem in &args[1..] {
                inner.insert_mut(elem.clone());
            }
            Ok(Value::Set(inner))
        }
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn is_macro(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Macro(..) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn time_in_millis(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if !args.is_empty() {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| -> EvaluationError { InterpreterError::SystemTimeError(err).into() })?;
    Ok(Value::Number(duration.as_millis() as i64))
}

pub fn to_seq(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
        Value::String(s) if s.is_empty() => Ok(Value::Nil),
        Value::String(s) => Ok(list_with_values(
            s.chars().map(|c| Value::String(c.to_string())),
        )),
        Value::List(coll) if coll.is_empty() => Ok(Value::Nil),
        l @ Value::List(..) => Ok(l.clone()),
        Value::Vector(coll) if coll.is_empty() => Ok(Value::Nil),
        Value::Vector(coll) => Ok(list_with_values(coll.iter().cloned())),
        Value::Map(coll) if coll.is_empty() => Ok(Value::Nil),
        Value::Map(coll) => Ok(list_with_values(coll.iter().map(|(k, v)| {
            let mut inner = PersistentVector::new();
            inner.push_back_mut(k.clone());
            inner.push_back_mut(v.clone());
            Value::Vector(inner)
        }))),
        Value::Set(coll) if coll.is_empty() => Ok(Value::Nil),
        Value::Set(coll) => Ok(list_with_values(coll.iter().cloned())),
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn readline(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "wrong arity".to_string(),
        )));
    }
    match &args[0] {
        Value::String(s) => {
            let stdout = io::stdout();
            let stdin = io::stdin();
            let mut stdout = stdout.lock();
            let mut stdin = stdin.lock();

            stdout
                .write(s.as_bytes())
                .map_err(|err| -> EvaluationError { InterpreterError::IOError(err).into() })?;

            stdout
                .flush()
                .map_err(|err| -> EvaluationError { InterpreterError::IOError(err).into() })?;

            let mut input = String::new();
            let count = stdin
                .read_line(&mut input)
                .map_err(|err| -> EvaluationError { InterpreterError::IOError(err).into() })?;
            if count == 0 {
                writeln!(stdout)
                    .map_err(|err| -> EvaluationError { InterpreterError::IOError(err).into() })?;
                Ok(Value::Nil)
            } else {
                if input.ends_with('\n') {
                    input.pop();
                }
                Ok(Value::String(input))
            }
        }
        _ => Err(EvaluationError::List(ListEvaluationError::Failure(
            "incorrect argument".to_string(),
        ))),
    }
}

pub fn to_meta(_: &mut Interpreter, _args: &[Value]) -> EvaluationResult<Value> {
    Ok(Value::Nil)
}

pub fn with_meta(_: &mut Interpreter, _args: &[Value]) -> EvaluationResult<Value> {
    Ok(Value::Nil)
}

pub const BINDINGS: &[(&str, Value)] = &[
    ("+", Value::Primitive(plus)),
    ("-", Value::Primitive(subtract)),
    ("*", Value::Primitive(multiply)),
    ("/", Value::Primitive(divide)),
    ("pr", Value::Primitive(pr)),
    ("prn", Value::Primitive(prn)),
    ("pr-str", Value::Primitive(pr_str)),
    ("print", Value::Primitive(print_)),
    ("println", Value::Primitive(println)),
    ("print-str", Value::Primitive(print_str)),
    ("list", Value::Primitive(list)),
    ("list?", Value::Primitive(is_list)),
    ("empty?", Value::Primitive(is_empty)),
    ("count", Value::Primitive(count)),
    ("<", Value::Primitive(less)),
    ("<=", Value::Primitive(less_eq)),
    (">", Value::Primitive(greater)),
    (">=", Value::Primitive(greater_eq)),
    ("=", Value::Primitive(equal)),
    ("read-string", Value::Primitive(read_string)),
    ("spit", Value::Primitive(spit)),
    ("slurp", Value::Primitive(slurp)),
    ("eval", Value::Primitive(eval)),
    ("str", Value::Primitive(to_str)),
    ("atom", Value::Primitive(to_atom)),
    ("atom?", Value::Primitive(is_atom)),
    ("deref", Value::Primitive(deref)),
    ("reset!", Value::Primitive(reset_atom)),
    ("swap!", Value::Primitive(swap_atom)),
    ("cons", Value::Primitive(cons)),
    ("concat", Value::Primitive(concat)),
    ("vec", Value::Primitive(vec)),
    ("nth", Value::Primitive(nth)),
    ("first", Value::Primitive(first)),
    ("rest", Value::Primitive(rest)),
    ("ex-info", Value::Primitive(ex_info)),
    ("throw", Value::Primitive(throw)),
    ("apply", Value::Primitive(apply)),
    ("map", Value::Primitive(map)),
    ("nil?", Value::Primitive(is_nil)),
    ("true?", Value::Primitive(is_true)),
    ("false?", Value::Primitive(is_false)),
    ("symbol?", Value::Primitive(is_symbol)),
    ("symbol", Value::Primitive(to_symbol)),
    ("keyword", Value::Primitive(to_keyword)),
    ("keyword?", Value::Primitive(is_keyword)),
    ("vector", Value::Primitive(to_vector)),
    ("vector?", Value::Primitive(is_vector)),
    ("sequential?", Value::Primitive(is_sequential)),
    ("hash-map", Value::Primitive(to_map)),
    ("map?", Value::Primitive(is_map)),
    ("set", Value::Primitive(to_set)),
    ("set?", Value::Primitive(is_set)),
    ("assoc", Value::Primitive(assoc)),
    ("dissoc", Value::Primitive(dissoc)),
    ("get", Value::Primitive(get)),
    ("contains?", Value::Primitive(does_contain)),
    ("keys", Value::Primitive(to_keys)),
    ("vals", Value::Primitive(to_vals)),
    ("last", Value::Primitive(last)),
    ("string?", Value::Primitive(is_string)),
    ("number?", Value::Primitive(is_number)),
    ("fn?", Value::Primitive(is_fn)),
    ("conj", Value::Primitive(conj)),
    ("macro?", Value::Primitive(is_macro)),
    ("time-ms", Value::Primitive(time_in_millis)),
    ("seq", Value::Primitive(to_seq)),
    ("readline", Value::Primitive(readline)),
    ("meta", Value::Primitive(to_meta)),
    ("with-meta", Value::Primitive(with_meta)),
];
