use crate::interpreter::{EvaluationError, EvaluationResult, Interpreter, InterpreterError};
use crate::namespace::{Namespace, DEFAULT_NAME};
use crate::reader::read;
use crate::value::{
    atom_impl_into_inner, atom_with_value, exception, list_with_values, map_with_values,
    set_with_values, var_impl_into_inner, vector_with_values, ExceptionImpl, FnWithCapturesImpl,
    PersistentList, PersistentSet, PersistentVector, Value,
};
use itertools::Itertools;
use std::fmt::Write;
use std::io::{BufRead, Write as IOWrite};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fs, io};

const BINDINGS: &[(&str, Value)] = &[
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
    ("zero?", Value::Primitive(is_zero)),
];

pub fn register(interpreter: &mut Interpreter) {
    let mut namespace = Namespace::new(DEFAULT_NAME);
    for (identifier, value) in BINDINGS {
        namespace
            .intern(identifier, value)
            .expect("prelude vars installed correctly");
    }

    interpreter.load_namespace(namespace);
}

pub fn plus(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    args.iter()
        .try_fold(i64::default(), |acc, x| match x {
            Value::Number(n) => acc
                .checked_add(*n)
                .ok_or_else(|| EvaluationError::Overflow(acc, *n)),
            other => Err(EvaluationError::WrongType {
                expected: "Number",
                realized: other.clone(),
            }),
        })
        .map(Value::Number)
}

pub fn subtract(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    match args.len() {
        0 => Err(EvaluationError::WrongArity {
            expected: 1,
            realized: 0,
        }),
        1 => match &args[0] {
            Value::Number(first) => first
                .checked_neg()
                .ok_or_else(|| EvaluationError::Negation(*first))
                .map(Value::Number),
            other => Err(EvaluationError::WrongType {
                expected: "Number",
                realized: other.clone(),
            }),
        },
        _ => {
            let first_value = &args[0];
            let rest_values = &args[1..];
            match first_value {
                Value::Number(first) => rest_values
                    .iter()
                    .try_fold(*first, |acc, x| match x {
                        Value::Number(next) => acc
                            .checked_sub(*next)
                            .ok_or_else(|| EvaluationError::Underflow(acc, *next)),
                        other => Err(EvaluationError::WrongType {
                            expected: "Number",
                            realized: other.clone(),
                        }),
                    })
                    .map(Value::Number),
                other => Err(EvaluationError::WrongType {
                    expected: "Number",
                    realized: other.clone(),
                }),
            }
        }
    }
}

pub fn multiply(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    args.iter()
        .try_fold(1_i64, |acc, x| match x {
            Value::Number(n) => acc
                .checked_mul(*n)
                .ok_or_else(|| EvaluationError::Overflow(acc, *n)),
            other => Err(EvaluationError::WrongType {
                expected: "Number",
                realized: other.clone(),
            }),
        })
        .map(Value::Number)
}

pub fn divide(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    match args.len() {
        0 => Err(EvaluationError::WrongArity {
            expected: 1,
            realized: 0,
        }),
        1 => match &args[0] {
            Value::Number(first) => 1_i64
                .checked_div_euclid(*first)
                .ok_or_else(|| EvaluationError::Overflow(1, *first))
                .map(Value::Number),
            other => Err(EvaluationError::WrongType {
                expected: "Number",
                realized: other.clone(),
            }),
        },
        _ => {
            let first_value = &args[0];
            let rest_values = &args[1..];
            match first_value {
                Value::Number(first) => rest_values
                    .iter()
                    .try_fold(*first, |acc, x| match x {
                        Value::Number(next) => acc
                            .checked_div_euclid(*next)
                            .ok_or_else(|| EvaluationError::Overflow(acc, *next)),
                        other => Err(EvaluationError::WrongType {
                            expected: "Number",
                            realized: other.clone(),
                        }),
                    })
                    .map(Value::Number),
                other => Err(EvaluationError::WrongType {
                    expected: "Number",
                    realized: other.clone(),
                }),
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
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match args[0] {
        Value::List(_) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn is_empty(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::Nil => Ok(Value::Bool(true)),
        Value::String(s) => Ok(Value::Bool(s.is_empty())),
        Value::List(elems) => Ok(Value::Bool(elems.is_empty())),
        Value::Vector(elems) => Ok(Value::Bool(elems.is_empty())),
        Value::Map(elems) => Ok(Value::Bool(elems.is_empty())),
        Value::Set(elems) => Ok(Value::Bool(elems.is_empty())),
        other => Err(EvaluationError::WrongType {
            expected: "Nil, String, List, Vector, Map, Set",
            realized: other.clone(),
        }),
    }
}

pub fn count(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::Nil => Ok(Value::Number(0)),
        Value::String(s) => Ok(Value::Number(s.len() as i64)),
        Value::List(elems) => Ok(Value::Number(elems.len() as i64)),
        Value::Vector(elems) => Ok(Value::Number(elems.len() as i64)),
        Value::Map(elems) => Ok(Value::Number(elems.size() as i64)),
        Value::Set(elems) => Ok(Value::Number(elems.size() as i64)),
        other => Err(EvaluationError::WrongType {
            expected: "Nil, String, List, Vector, Map, Set",
            realized: other.clone(),
        }),
    }
}

macro_rules! comparator {
    ($name:ident, $comparison:tt) => {
        pub fn $name(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
            if args.len() != 2 {
                return Err(EvaluationError::WrongArity {
                    expected: 2,
                    realized: args.len(),
                });
            }
            match &args[0] {
                Value::Number(a) => match &args[1] {
                    Value::Number(b) => Ok(Value::Bool(a $comparison b)),
                    other => Err(EvaluationError::WrongType {
                        expected: "Number",
                        realized: other.clone(),
                    }),
                },
                other => Err(EvaluationError::WrongType {
                    expected: "Number",
                    realized: other.clone(),
                }),
            }
        }
    };
}

comparator!(less, <);
comparator!(less_eq, <=);
comparator!(greater, >);
comparator!(greater_eq, >=);

pub fn equal(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
    }
    Ok(Value::Bool(args[0] == args[1]))
}

pub fn read_string(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::String(s) => {
            let mut forms = read(s).map_err(|err| {
                let context = err.context(s);
                EvaluationError::ReaderError(err, context.to_string())
            })?;
            if forms.is_empty() {
                Ok(Value::Nil)
            } else {
                Ok(forms.pop().unwrap())
            }
        }
        other => Err(EvaluationError::WrongType {
            expected: "String",
            realized: other.clone(),
        }),
    }
}

pub fn spit(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::String(path) => {
            let mut contents = String::new();
            let _ = write!(&mut contents, "{}", &args[1]);
            let _ = fs::write(path, contents).unwrap();
            Ok(Value::Nil)
        }
        other => Err(EvaluationError::WrongType {
            expected: "String",
            realized: other.clone(),
        }),
    }
}

pub fn slurp(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::String(path) => {
            let contents = fs::read_to_string(path).unwrap();
            Ok(Value::String(contents))
        }
        other => Err(EvaluationError::WrongType {
            expected: "String",
            realized: other.clone(),
        }),
    }
}

pub fn eval(interpreter: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
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
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    Ok(atom_with_value(args[0].clone()))
}

pub fn is_atom(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match args[0] {
        Value::Atom(_) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

pub fn deref(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::Atom(inner) => Ok(atom_impl_into_inner(inner)),
        Value::Var(var) => var_impl_into_inner(var)
            .ok_or_else(|| EvaluationError::CannotDerefUnboundVar(Value::Var(var.clone()))),
        other => Err(EvaluationError::WrongType {
            expected: "Atom, Var",
            realized: other.clone(),
        }),
    }
}

pub fn reset_atom(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::Atom(inner) => {
            let value = args[1].clone();
            *inner.borrow_mut() = value.clone();
            Ok(value)
        }
        other => Err(EvaluationError::WrongType {
            expected: "Atom",
            realized: other.clone(),
        }),
    }
}

pub fn swap_atom(interpreter: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() < 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::Atom(cell) => match &args[1] {
            Value::Fn(f) => {
                let mut inner = cell.borrow_mut();
                let original_value = inner.clone();
                let mut fn_args = vec![original_value];
                fn_args.extend_from_slice(&args[2..]);
                let new_value = interpreter.apply_fn_inner(f, &fn_args, fn_args.len())?;
                *inner = new_value.clone();
                Ok(new_value)
            }
            Value::FnWithCaptures(FnWithCapturesImpl { f, captures }) => {
                interpreter.extend_from_captures(captures)?;
                let mut inner = cell.borrow_mut();
                let original_value = inner.clone();
                let mut fn_args = vec![original_value];
                fn_args.extend_from_slice(&args[2..]);
                let new_value = interpreter.apply_fn_inner(f, &fn_args, fn_args.len());
                interpreter.leave_scope();

                let new_value = new_value?;
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
            other => Err(EvaluationError::WrongType {
                expected: "Fn, FnWithCaptures, Primitive",
                realized: other.clone(),
            }),
        },
        other => Err(EvaluationError::WrongType {
            expected: "Atom",
            realized: other.clone(),
        }),
    }
}

pub fn cons(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
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
        other => Err(EvaluationError::WrongType {
            expected: "List, Vector",
            realized: other.clone(),
        }),
    }
}

pub fn concat(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    let mut elems = vec![];
    for arg in args {
        match arg {
            Value::List(seq) => elems.extend(seq.iter().cloned()),
            Value::Vector(seq) => elems.extend(seq.iter().cloned()),
            other => {
                return Err(EvaluationError::WrongType {
                    expected: "List, Vector",
                    realized: other.clone(),
                });
            }
        }
    }
    Ok(list_with_values(elems))
}

pub fn vec(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::List(elems) => Ok(vector_with_values(elems.iter().cloned())),
        Value::Vector(elems) => Ok(vector_with_values(elems.iter().cloned())),
        Value::Nil => Ok(vector_with_values([].iter().cloned())),
        other => Err(EvaluationError::WrongType {
            expected: "List, Vector, Nil",
            realized: other.clone(),
        }),
    }
}

pub fn nth(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
    }
    match &args[1] {
        Value::Number(index) if *index >= 0 => {
            let index = *index as usize;
            match &args[0] {
                Value::List(seq) => seq
                    .iter()
                    .nth(index)
                    .ok_or_else(|| EvaluationError::IndexOutOfBounds(index, seq.len()))
                    .map(|elem| elem.clone()),
                Value::Vector(seq) => seq
                    .iter()
                    .nth(index)
                    .ok_or_else(|| EvaluationError::IndexOutOfBounds(index, seq.len()))
                    .map(|elem| elem.clone()),
                other => Err(EvaluationError::WrongType {
                    expected: "List, Vector",
                    realized: other.clone(),
                }),
            }
        }
        other => Err(EvaluationError::WrongType {
            expected: "Number",
            realized: other.clone(),
        }),
    }
}

pub fn first(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
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
        other => Err(EvaluationError::WrongType {
            expected: "List, Vector, Nil",
            realized: other.clone(),
        }),
    }
}

pub fn rest(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
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
        other => Err(EvaluationError::WrongType {
            expected: "List, Vector, Nil",
            realized: other.clone(),
        }),
    }
}

pub fn ex_info(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::String(msg) => Ok(Value::Exception(ExceptionImpl::User(exception(
            msg, &args[1],
        )))),
        other => Err(EvaluationError::WrongType {
            expected: "String",
            realized: other.clone(),
        }),
    }
}

pub fn throw(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    let exception =
        match &args[0] {
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
            Value::Exception(e) => match e {
                ExceptionImpl::User(inner) => inner.clone(),
                _ => unreachable!("invariant to not nest system errors"),
            },
            other => return Err(EvaluationError::WrongType {
                expected:
                    "Nil, Bool, Number, String, Keyword, Symbol, List, Vector, Map, Set, Exception",
                realized: other.clone(),
            }),
        };
    Err(EvaluationError::Exception(exception))
}

pub fn apply(interpreter: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() < 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
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
        other => {
            return Err(EvaluationError::WrongType {
                expected: "List, Vector",
                realized: other.clone(),
            })
        }
    };
    match first {
        Value::Fn(f) => interpreter.apply_fn_inner(f, &fn_args, fn_args.len()),
        Value::FnWithCaptures(FnWithCapturesImpl { f, captures }) => {
            interpreter.extend_from_captures(captures)?;
            let result = interpreter.apply_fn_inner(f, &fn_args, fn_args.len());
            interpreter.leave_scope();
            result
        }
        Value::Primitive(native_fn) => native_fn(interpreter, &fn_args),
        other => Err(EvaluationError::WrongType {
            expected: "Fn, FnWithCaptures, Primitive",
            realized: other.clone(),
        }),
    }
}

pub fn map(interpreter: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
    }
    let fn_args: Vec<_> = match &args[1] {
        Value::Nil => return Ok(Value::List(PersistentList::new())),
        Value::List(elems) => elems.iter().collect(),
        Value::Vector(elems) => elems.iter().collect(),
        other => {
            return Err(EvaluationError::WrongType {
                expected: "Nil, List, Vector",
                realized: other.clone(),
            })
        }
    };
    let mut result = PersistentList::new();
    match &args[0] {
        Value::Fn(f) => {
            for arg in fn_args.into_iter().rev() {
                let mapped_arg = interpreter.apply_fn_inner(f, [arg], 1)?;
                result.push_front_mut(mapped_arg);
            }
        }
        Value::FnWithCaptures(FnWithCapturesImpl { f, captures }) => {
            interpreter.extend_from_captures(captures)?;
            for arg in fn_args.into_iter().rev() {
                let mapped_arg = interpreter.apply_fn_inner(f, [arg], 1)?;
                result.push_front_mut(mapped_arg);
            }
            interpreter.leave_scope();
        }
        Value::Primitive(native_fn) => {
            for arg in fn_args.into_iter().rev() {
                let mapped_arg = native_fn(interpreter, &[arg.clone()])?;
                result.push_front_mut(mapped_arg);
            }
        }
        other => {
            return Err(EvaluationError::WrongType {
                expected: "Fn, FnWithCaptures, Primitive",
                realized: other.clone(),
            });
        }
    };
    Ok(Value::List(result))
}

macro_rules! is_type {
    ($name:ident, $($target_type:pat) |*) => {
        pub fn $name(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
            if args.len() != 1 {
                return Err(EvaluationError::WrongArity {
                    expected: 1,
                    realized: args.len(),
                });
            }
            match &args[0] {
                $($target_type) |* => Ok(Value::Bool(true)),
                _ => Ok(Value::Bool(false)),
            }
        }
    };
}

is_type!(is_nil, Value::Nil);
is_type!(is_true, Value::Bool(true));
is_type!(is_false, Value::Bool(false));
is_type!(is_symbol, Value::Symbol(..));
is_type!(is_keyword, Value::Keyword(..));
is_type!(is_vector, Value::Vector(..));
is_type!(is_sequential, Value::List(..) | Value::Vector(..));
is_type!(is_map, Value::Map(..));
is_type!(is_set, Value::Set(..));
is_type!(is_string, Value::String(..));
is_type!(is_number, Value::Number(..));
is_type!(
    is_fn,
    Value::Fn(..) | Value::FnWithCaptures(..) | Value::Primitive(..) | Value::Macro(..)
);
is_type!(is_macro, Value::Macro(..));

pub fn to_symbol(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::String(name) => Ok(Value::Symbol(name.clone(), None)),
        other => Err(EvaluationError::WrongType {
            expected: "String",
            realized: other.clone(),
        }),
    }
}

pub fn to_keyword(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::String(name) => Ok(Value::Keyword(name.clone(), None)),
        k @ Value::Keyword(..) => Ok(k.clone()),
        other => Err(EvaluationError::WrongType {
            expected: "String, Keyword",
            realized: other.clone(),
        }),
    }
}

pub fn to_vector(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    Ok(vector_with_values(args.iter().cloned()))
}

pub fn to_map(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() % 2 != 0 {
        return Err(EvaluationError::MapRequiresPairs(
            vector_with_values(args.iter().cloned()),
            args.len(),
        ));
    }
    Ok(map_with_values(args.iter().cloned().tuples()))
}

pub fn to_set(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
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
        other => Err(EvaluationError::WrongType {
            expected: "Nil, String, List, Vector, Map, Set",
            realized: other.clone(),
        }),
    }
}

pub fn assoc(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() < 3 {
        return Err(EvaluationError::WrongArity {
            expected: 3,
            realized: args.len(),
        });
    }
    if (args.len() - 1) % 2 != 0 {
        return Err(EvaluationError::MapRequiresPairs(
            vector_with_values(args.iter().cloned()),
            args.len(),
        ));
    }
    match &args[0] {
        Value::Map(map) => {
            let mut result = map.clone();
            for (key, val) in args.iter().skip(1).tuples() {
                result.insert_mut(key.clone(), val.clone());
            }
            Ok(Value::Map(result))
        }
        other => Err(EvaluationError::WrongType {
            expected: "Map",
            realized: other.clone(),
        }),
    }
}

pub fn dissoc(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.is_empty() {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::Map(map) => {
            let mut result = map.clone();
            for key in args.iter().skip(1) {
                result.remove_mut(key);
            }
            Ok(Value::Map(result))
        }
        other => Err(EvaluationError::WrongType {
            expected: "Map",
            realized: other.clone(),
        }),
    }
}

pub fn get(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
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
        other => Err(EvaluationError::WrongType {
            expected: "Nil, Map",
            realized: other.clone(),
        }),
    }
}

pub fn does_contain(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::Nil => Ok(Value::Bool(false)),
        Value::Map(map) => {
            let contains = map.contains_key(&args[1]);
            Ok(Value::Bool(contains))
        }
        other => Err(EvaluationError::WrongType {
            expected: "Nil, Map",
            realized: other.clone(),
        }),
    }
}

pub fn to_keys(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    let result = match &args[0] {
        Value::Nil => Value::Nil,
        Value::Map(map) => {
            if map.is_empty() {
                Value::Nil
            } else {
                list_with_values(map.keys().cloned())
            }
        }
        other => {
            return Err(EvaluationError::WrongType {
                expected: "Nil, Map",
                realized: other.clone(),
            })
        }
    };
    Ok(result)
}

pub fn to_vals(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    let result = match &args[0] {
        Value::Nil => Value::Nil,
        Value::Map(map) => {
            if map.is_empty() {
                Value::Nil
            } else {
                list_with_values(map.values().cloned())
            }
        }
        other => {
            return Err(EvaluationError::WrongType {
                expected: "Nil, Map",
                realized: other.clone(),
            })
        }
    };
    Ok(result)
}

pub fn last(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::Nil => Ok(Value::Nil),
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
        other => Err(EvaluationError::WrongType {
            expected: "Nil, List, Vector",
            realized: other.clone(),
        }),
    }
}

pub fn conj(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() < 2 {
        return Err(EvaluationError::WrongArity {
            expected: 2,
            realized: args.len(),
        });
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
                    other => {
                        return Err(EvaluationError::WrongType {
                            expected: "Vector, Map",
                            realized: other.clone(),
                        })
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
        other => Err(EvaluationError::WrongType {
            expected: "Nil, List, Vector, Map, Set",
            realized: other.clone(),
        }),
    }
}

pub fn time_in_millis(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if !args.is_empty() {
        return Err(EvaluationError::WrongArity {
            expected: 0,
            realized: args.len(),
        });
    }
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| -> EvaluationError { InterpreterError::SystemTimeError(err).into() })?;
    Ok(Value::Number(duration.as_millis() as i64))
}

pub fn to_seq(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
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
        other => Err(EvaluationError::WrongType {
            expected: "Nil, String, List, Vector, Map, Set",
            realized: other.clone(),
        }),
    }
}

pub fn readline(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::String(s) => {
            let stdout = io::stdout();
            let stdin = io::stdin();
            let mut stdout = stdout.lock();
            let mut stdin = stdin.lock();

            stdout
                .write(s.as_bytes())
                .map_err(|err| -> EvaluationError {
                    let interpreter_error: InterpreterError = err.into();
                    interpreter_error.into()
                })?;

            stdout.flush().map_err(|err| -> EvaluationError {
                let interpreter_error: InterpreterError = err.into();
                interpreter_error.into()
            })?;

            let mut input = String::new();
            let count = stdin
                .read_line(&mut input)
                .map_err(|err| -> EvaluationError {
                    let interpreter_error: InterpreterError = err.into();
                    interpreter_error.into()
                })?;
            if count == 0 {
                writeln!(stdout).map_err(|err| -> EvaluationError {
                    let interpreter_error: InterpreterError = err.into();
                    interpreter_error.into()
                })?;
                Ok(Value::Nil)
            } else {
                if input.ends_with('\n') {
                    input.pop();
                }
                Ok(Value::String(input))
            }
        }
        other => Err(EvaluationError::WrongType {
            expected: "String",
            realized: other.clone(),
        }),
    }
}

pub fn to_meta(_: &mut Interpreter, _args: &[Value]) -> EvaluationResult<Value> {
    Ok(Value::Nil)
}

pub fn with_meta(_: &mut Interpreter, _args: &[Value]) -> EvaluationResult<Value> {
    Ok(Value::Nil)
}

pub fn is_zero(_: &mut Interpreter, args: &[Value]) -> EvaluationResult<Value> {
    if args.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: args.len(),
        });
    }
    match &args[0] {
        Value::Number(n) => Ok(Value::Bool(*n == 0)),
        other => Err(EvaluationError::WrongType {
            expected: "Number",
            realized: other.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::run_eval_test;
    use crate::value::{
        list_with_values, map_with_values, set_with_values, vector_with_values, Value::*,
    };
    use crate::value::{PersistentList, PersistentMap, PersistentSet, PersistentVector};
    use std::iter::FromIterator;

    #[test]
    fn test_basic_prelude() {
        let test_cases = vec![
            ("(list)", list_with_values(vec![])),
            (
                "(list 1 2)",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            ("(list? (list 1))", Bool(true)),
            ("(list? (list))", Bool(true)),
            ("(list? [1 2])", Bool(false)),
            ("(empty? (list))", Bool(true)),
            ("(empty? (list 1))", Bool(false)),
            ("(empty? [1 2 3])", Bool(false)),
            ("(empty? [])", Bool(true)),
            ("(count nil)", Number(0)),
            ("(count \"hi\")", Number(2)),
            ("(count \"\")", Number(0)),
            ("(count (list))", Number(0)),
            ("(count (list 44 42 41))", Number(3)),
            ("(count [])", Number(0)),
            ("(count [1 2 3])", Number(3)),
            ("(count {})", Number(0)),
            ("(count {:a 1 :b 2})", Number(2)),
            ("(count #{})", Number(0)),
            ("(count #{:a 1 :b 2})", Number(4)),
            ("(if (< 2 3) 12 13)", Number(12)),
            ("(> 13 12)", Bool(true)),
            ("(> 13 13)", Bool(false)),
            ("(> 12 13)", Bool(false)),
            ("(< 13 12)", Bool(false)),
            ("(< 13 13)", Bool(false)),
            ("(< 12 13)", Bool(true)),
            ("(<= 12 12)", Bool(true)),
            ("(<= 13 12)", Bool(false)),
            ("(<= 12 13)", Bool(true)),
            ("(>= 13 12)", Bool(true)),
            ("(>= 13 13)", Bool(true)),
            ("(>= 13 14)", Bool(false)),
            ("(= 12 12)", Bool(true)),
            ("(= 12 13)", Bool(false)),
            ("(= 13 12)", Bool(false)),
            ("(= 0 0)", Bool(true)),
            ("(= 1 0)", Bool(false)),
            ("(= true true)", Bool(true)),
            ("(= true false)", Bool(false)),
            ("(= false false)", Bool(true)),
            ("(= nil nil)", Bool(true)),
            ("(= (list) (list))", Bool(true)),
            ("(= (list) ())", Bool(true)),
            ("(= (list 1 2) '(1 2))", Bool(true)),
            ("(= (list 1 ) ())", Bool(false)),
            ("(= (list ) '(1))", Bool(false)),
            ("(= 0 (list))", Bool(false)),
            ("(= (list) 0)", Bool(false)),
            ("(= (list nil) (list))", Bool(false)),
            ("(= 1 (+ 1 1))", Bool(false)),
            ("(= 2 (+ 1 1))", Bool(true)),
            ("(= nil (+ 1 1))", Bool(false)),
            ("(= nil nil)", Bool(true)),
            ("(= \"\" \"\")", Bool(true)),
            ("(= \"abc\" \"abc\")", Bool(true)),
            ("(= \"\" \"abc\")", Bool(false)),
            ("(= \"abc\" \"\")", Bool(false)),
            ("(= \"abc\" \"def\")", Bool(false)),
            ("(= \"abc\" \"ABC\")", Bool(false)),
            ("(= (list) \"\")", Bool(false)),
            ("(= \"\" (list))", Bool(false)),
            ("(= :abc :abc)", Bool(true)),
            ("(= :abc :def)", Bool(false)),
            ("(= :abc \":abc\")", Bool(false)),
            ("(= (list :abc) (list :abc))", Bool(true)),
            ("(= [] (list))", Bool(true)),
            ("(= [7 8] [7 8])", Bool(true)),
            ("(= [:abc] [:abc])", Bool(true)),
            ("(= (list 1 2) [1 2])", Bool(true)),
            ("(= (list 1) [])", Bool(false)),
            ("(= [] (list 1))", Bool(false)),
            ("(= [] [1])", Bool(false)),
            ("(= 0 [])", Bool(false)),
            ("(= [] 0)", Bool(false)),
            ("(= [] \"\")", Bool(false)),
            ("(= \"\" [])", Bool(false)),
            ("(= [(list)] (list []))", Bool(true)),
            ("(= 'abc 'abc)", Bool(true)),
            ("(= 'abc 'abdc)", Bool(false)),
            ("(= 'abc \"abc\")", Bool(false)),
            ("(= \"abc\" 'abc)", Bool(false)),
            ("(= \"abc\" (str 'abc))", Bool(true)),
            ("(= 'abc nil)", Bool(false)),
            ("(= nil 'abc)", Bool(false)),
            ("(= {} {})", Bool(true)),
            ("(= {} (hash-map))", Bool(true)),
            ("(= {:a 11 :b 22} (hash-map :b 22 :a 11))", Bool(true)),
            (
                "(= {:a 11 :b [22 33]} (hash-map :b [22 33] :a 11))",
                Bool(true),
            ),
            (
                "(= {:a 11 :b {:c 22}} (hash-map :b (hash-map :c 22) :a 11))",
                Bool(true),
            ),
            ("(= {:a 11 :b 22} (hash-map :b 23 :a 11))", Bool(false)),
            ("(= {:a 11 :b 22} (hash-map :a 11))", Bool(false)),
            ("(= {:a [11 22]} {:a (list 11 22)})", Bool(true)),
            ("(= {:a 11 :b 22} (list :a 11 :b 22))", Bool(false)),
            ("(= {} [])", Bool(false)),
            ("(= [] {})", Bool(false)),
            (
                "(= [1 2 (list 3 4 [5 6])] (list 1 2 [3 4 (list 5 6)]))",
                Bool(true),
            ),
            (
                "(read-string \"(+ 1 2)\")",
                List(PersistentList::from_iter(vec![
                    Symbol("+".to_string(), None),
                    Number(1),
                    Number(2),
                ])),
            ),
            (
                "(read-string \"(1 2 (3 4) nil)\")",
                List(PersistentList::from_iter(vec![
                    Number(1),
                    Number(2),
                    List(PersistentList::from_iter(vec![Number(3), Number(4)])),
                    Nil,
                ])),
            ),
            ("(= nil (read-string \"nil\"))", Bool(true)),
            ("(read-string \"7 ;; comment\")", Number(7)),
            ("(read-string \"7;;!\")", Number(7)),
            ("(read-string \"7;;#\")", Number(7)),
            ("(read-string \"7;;$\")", Number(7)),
            ("(read-string \"7;;%\")", Number(7)),
            ("(read-string \"7;;'\")", Number(7)),
            ("(read-string \"7;;\\\\\")", Number(7)),
            ("(read-string \"7;;////////\")", Number(7)),
            ("(read-string \"7;;`\")", Number(7)),
            ("(read-string \"7;; &()*+,-./:;<=>?@[]^_{|}~\")", Number(7)),
            ("(read-string \";; comment\")", Nil),
            ("(eval (list + 1 2 3))", Number(6)),
            ("(eval (read-string \"(+ 2 3)\"))", Number(5)),
            (
                "(def! a 1) (let* [a 12] (eval (read-string \"a\")))",
                Number(1),
            ),
            (
                "(let* [b 12] (do (eval (read-string \"(def! aa 7)\")) aa))",
                Number(7),
            ),
            ("(str)", String("".to_string())),
            ("(str \"\")", String("".to_string())),
            ("(str \"hi\" 3 :foo)", String("hi3:foo".to_string())),
            ("(str \"hi   \" 3 :foo)", String("hi   3:foo".to_string())),
            ("(str [])", String("[]".to_string())),
            ("(str [\"hi\"])", String("[\"hi\"]".to_string())),
            (
                "(str \"A\" {:abc \"val\"} \"Z\")",
                String("A{:abc \"val\"}Z".to_string()),
            ),
            (
                "(str true \".\" false \".\" nil \".\" :keyw \".\" 'symb)",
                String("true.false.nil.:keyw.symb".to_string()),
            ),
            (
                "(str true \".\" false \".\" nil \".\" :keyw \".\" 'symb)",
                String("true.false.nil.:keyw.symb".to_string()),
            ),
            (
                "(pr-str \"A\" {:abc \"val\"} \"Z\")",
                String("\"A\" {:abc \"val\"} \"Z\"".to_string()),
            ),
            (
                "(pr-str true \".\" false \".\" nil \".\" :keyw \".\" 'symb)",
                String("true \".\" false \".\" nil \".\" :keyw \".\" symb".to_string()),
            ),
            (
                "(cons 1 (list))",
                list_with_values([Number(1)].iter().cloned()),
            ),
            ("(cons 1 [])", list_with_values([Number(1)].iter().cloned())),
            (
                "(cons 1 (list 2))",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(cons 1 (list 2 3))",
                list_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(cons 1 [2 3])",
                list_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(cons [1] [2 3])",
                list_with_values(
                    [vector_with_values(vec![Number(1)]), Number(2), Number(3)]
                        .iter()
                        .cloned(),
                ),
            ),
            (
                "(def! a [2 3]) (cons 1 a)",
                list_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(def! a [2 3]) (cons 1 a) a",
                vector_with_values(vec![Number(2), Number(3)]),
            ),
            (
                "(cons (list 1) (list 2 3))",
                list_with_values(
                    [
                        list_with_values([Number(1)].iter().cloned()),
                        Number(2),
                        Number(3),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            ("(concat)", List(PersistentList::new())),
            ("(concat (concat))", List(PersistentList::new())),
            ("(concat (list) (list))", List(PersistentList::new())),
            ("(= () (concat))", Bool(true)),
            (
                "(concat (list 1 2))",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(concat (list 1) (list 2 3))",
                list_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(concat (list 1) [3 3] (list 2 3))",
                list_with_values(
                    [Number(1), Number(3), Number(3), Number(2), Number(3)]
                        .iter()
                        .cloned(),
                ),
            ),
            (
                "(concat [1 2] '(3 4) [5 6])",
                list_with_values(
                    [
                        Number(1),
                        Number(2),
                        Number(3),
                        Number(4),
                        Number(5),
                        Number(6),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            (
                "(concat (list 1) (list 2 3) (list (list 4 5) 6))",
                list_with_values(
                    [
                        Number(1),
                        Number(2),
                        Number(3),
                        list_with_values([Number(4), Number(5)].iter().cloned()),
                        Number(6),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            (
                "(def! a (list 1 2)) (def! b (list 3 4)) (concat a b (list 5 6))",
                list_with_values(
                    [
                        Number(1),
                        Number(2),
                        Number(3),
                        Number(4),
                        Number(5),
                        Number(6),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            (
                "(def! a (list 1 2)) (def! b (list 3 4)) (concat a b (list 5 6)) a",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(def! a (list 1 2)) (def! b (list 3 4)) (concat a b (list 5 6)) b",
                list_with_values([Number(3), Number(4)].iter().cloned()),
            ),
            (
                "(concat [1 2])",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(vec '(1 2 3))",
                vector_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(vec [1 2 3])",
                vector_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            ("(vec nil)", vector_with_values([].iter().cloned())),
            ("(vec '())", vector_with_values([].iter().cloned())),
            ("(vec [])", vector_with_values([].iter().cloned())),
            (
                "(def! a '(1 2)) (vec a)",
                vector_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(def! a '(1 2)) (vec a) a",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(vec '(1))",
                vector_with_values([Number(1)].iter().cloned()),
            ),
            ("(nth [1 2 3] 2)", Number(3)),
            ("(nth [1] 0)", Number(1)),
            ("(nth [1 2 nil] 2)", Nil),
            ("(nth '(1 2 3) 1)", Number(2)),
            ("(nth '(1 2 3) 0)", Number(1)),
            ("(nth '(1 2 nil) 2)", Nil),
            ("(first '(1 2 3))", Number(1)),
            ("(first '())", Nil),
            ("(first [1 2 3])", Number(1)),
            ("(first [10])", Number(10)),
            ("(first [])", Nil),
            ("(first nil)", Nil),
            (
                "(rest '(1 2 3))",
                list_with_values([Number(2), Number(3)].iter().cloned()),
            ),
            ("(rest '(1))", list_with_values(vec![])),
            ("(rest '())", List(PersistentList::new())),
            (
                "(rest [1 2 3])",
                list_with_values([Number(2), Number(3)].iter().cloned()),
            ),
            ("(rest [])", List(PersistentList::new())),
            ("(rest nil)", List(PersistentList::new())),
            ("(rest [10])", List(PersistentList::new())),
            (
                "(rest [10 11 12])",
                list_with_values(vec![Number(11), Number(12)]),
            ),
            (
                "(rest (cons 10 [11 12]))",
                list_with_values(vec![Number(11), Number(12)]),
            ),
            ("(apply str [1 2 3])", String("123".to_string())),
            ("(apply str '(1 2 3))", String("123".to_string())),
            ("(apply str 0 1 2 '(1 2 3))", String("012123".to_string())),
            ("(apply + '(2 3))", Number(5)),
            ("(apply + 4 '(5))", Number(9)),
            ("(apply + 4 [5])", Number(9)),
            ("(apply list ())", list_with_values(vec![])),
            ("(apply list [])", list_with_values(vec![])),
            ("(apply symbol? (list 'two))", Bool(true)),
            ("(apply (fn* [a b] (+ a b)) '(2 3))", Number(5)),
            ("(apply (fn* [a b] (+ a b)) 4 '(5))", Number(9)),
            ("(apply (fn* [a b] (+ a b)) [2 3])", Number(5)),
            ("(apply (fn* [a b] (+ a b)) 4 [5])", Number(9)),
            ("(apply (fn* [& rest] (list? rest)) [1 2 3])", Bool(true)),
            ("(apply (fn* [& rest] (list? rest)) [])", Bool(true)),
            ("(apply (fn* [a & rest] (list? rest)) [1])", Bool(true)),
            (
                "(def! inc (fn* [a] (+ a 1))) (map inc [1 2 3])",
                list_with_values(vec![Number(2), Number(3), Number(4)]),
            ),
            (
                "(map inc '(1 2 3))",
                list_with_values(vec![Number(2), Number(3), Number(4)]),
            ),
            (
                "(map (fn* [x] (* 2 x)) [1 2 3])",
                list_with_values(vec![Number(2), Number(4), Number(6)]),
            ),
            (
                "(map (fn* [& args] (list? args)) [1 2])",
                list_with_values(vec![Bool(true), Bool(true)]),
            ),
            (
                "(map symbol? '(nil false true))",
                list_with_values(vec![Bool(false), Bool(false), Bool(false)]),
            ),
            (
                "(def! f (fn* [a] (fn* [b] (+ a b)))) (map (f 23) (list 1 2))",
                list_with_values(vec![Number(24), Number(25)]),
            ),
            ("(= () (map str ()))", Bool(true)),
            ("(nil? nil)", Bool(true)),
            ("(nil? true)", Bool(false)),
            ("(nil? false)", Bool(false)),
            ("(nil? [1 2 3])", Bool(false)),
            ("(true? true)", Bool(true)),
            ("(true? nil)", Bool(false)),
            ("(true? false)", Bool(false)),
            ("(true? true?)", Bool(false)),
            ("(true? [1 2 3])", Bool(false)),
            ("(false? false)", Bool(true)),
            ("(false? nil)", Bool(false)),
            ("(false? true)", Bool(false)),
            ("(false? [1 2 3])", Bool(false)),
            ("(symbol? 'a)", Bool(true)),
            ("(symbol? 'foo/a)", Bool(true)),
            ("(symbol? :foo/a)", Bool(false)),
            ("(symbol? :a)", Bool(false)),
            ("(symbol? false)", Bool(false)),
            ("(symbol? true)", Bool(false)),
            ("(symbol? nil)", Bool(false)),
            ("(symbol? (symbol \"abc\"))", Bool(true)),
            ("(symbol? [1 2 3])", Bool(false)),
            ("(symbol \"hi\")", Symbol("hi".to_string(), None)),
            ("(keyword \"hi\")", Keyword("hi".to_string(), None)),
            ("(keyword :hi)", Keyword("hi".to_string(), None)),
            ("(keyword? :a)", Bool(true)),
            ("(keyword? false)", Bool(false)),
            ("(keyword? 'abc)", Bool(false)),
            ("(keyword? \"hi\")", Bool(false)),
            ("(keyword? \"\")", Bool(false)),
            ("(keyword? (keyword \"abc\"))", Bool(true)),
            (
                "(keyword? (first (keys {\":abc\" 123 \":def\" 456})))",
                Bool(false),
            ),
            ("(vector)", Vector(PersistentVector::new())),
            (
                "(vector 1)",
                vector_with_values([Number(1)].iter().cloned()),
            ),
            (
                "(vector 1 2 3)",
                vector_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            ("(vector? [1 2])", Bool(true)),
            ("(vector? '(1 2))", Bool(false)),
            ("(vector? :hi)", Bool(false)),
            ("(= [] (vector))", Bool(true)),
            ("(sequential? '(1 2))", Bool(true)),
            ("(sequential? [1 2])", Bool(true)),
            ("(sequential? :hi)", Bool(false)),
            ("(sequential? nil)", Bool(false)),
            ("(sequential? \"abc\")", Bool(false)),
            ("(sequential? sequential?)", Bool(false)),
            ("(hash-map)", Map(PersistentMap::new())),
            (
                "(hash-map :a 2)",
                map_with_values(
                    [(Keyword("a".to_string(), None), Number(2))]
                        .iter()
                        .cloned(),
                ),
            ),
            ("(map? {:a 1 :b 2})", Bool(true)),
            ("(map? {})", Bool(true)),
            ("(map? '())", Bool(false)),
            ("(map? [])", Bool(false)),
            ("(map? 'abc)", Bool(false)),
            ("(map? :abc)", Bool(false)),
            ("(map? [1 2])", Bool(false)),
            (
                "(assoc {} :a 1)",
                map_with_values(
                    [(Keyword("a".to_string(), None), Number(1))]
                        .iter()
                        .cloned(),
                ),
            ),
            (
                "(assoc {} :a 1 :b 3)",
                map_with_values(
                    [
                        (Keyword("a".to_string(), None), Number(1)),
                        (Keyword("b".to_string(), None), Number(3)),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            (
                "(assoc {:a 1} :b 3)",
                map_with_values(
                    [
                        (Keyword("a".to_string(), None), Number(1)),
                        (Keyword("b".to_string(), None), Number(3)),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            (
                "(assoc {:a 1} :a 3 :c 33)",
                map_with_values(vec![
                    (Keyword("a".to_string(), None), Number(3)),
                    (Keyword("c".to_string(), None), Number(33)),
                ]),
            ),
            (
                "(assoc {} :a nil)",
                map_with_values(vec![(Keyword("a".to_string(), None), Nil)]),
            ),
            ("(dissoc {})", map_with_values([].iter().cloned())),
            ("(dissoc {} :a)", map_with_values([].iter().cloned())),
            (
                "(dissoc {:a 1 :b 3} :a)",
                map_with_values(
                    [(Keyword("b".to_string(), None), Number(3))]
                        .iter()
                        .cloned(),
                ),
            ),
            (
                "(dissoc {:a 1 :b 3} :a :b :c)",
                map_with_values([].iter().cloned()),
            ),
            ("(count (keys (assoc {} :b 2 :c 3)))", Number(2)),
            ("(get {:a 1} :a)", Number(1)),
            ("(get {:a 1} :b)", Nil),
            ("(get nil :b)", Nil),
            ("(contains? {:a 1} :b)", Bool(false)),
            ("(contains? {:a 1} :a)", Bool(true)),
            ("(contains? {:abc nil} :abc)", Bool(true)),
            ("(contains? nil :abc)", Bool(false)),
            ("(contains? nil 'abc)", Bool(false)),
            ("(contains? nil [1 2 3])", Bool(false)),
            ("(keyword? (nth (keys {:abc 123 :def 456}) 0))", Bool(true)),
            ("(keyword? (nth (vals {123 :abc 456 :def}) 0))", Bool(true)),
            ("(keys {})", Nil),
            ("(keys nil)", Nil),
            (
                "(= (set '(:a :b :c)) (set (keys {:a 1 :b 2 :c 3})))",
                Bool(true),
            ),
            (
                "(= (set '(:a :c)) (set (keys {:a 1 :b 2 :c 3})))",
                Bool(false),
            ),
            ("(vals {})", Nil),
            ("(vals nil)", Nil),
            (
                "(= (set '(1 2 3)) (set (vals {:a 1 :b 2 :c 3})))",
                Bool(true),
            ),
            (
                "(= (set '(1 2)) (set (vals {:a 1 :b 2 :c 3})))",
                Bool(false),
            ),
            ("(last '(1 2 3))", Number(3)),
            ("(last [1 2 3])", Number(3)),
            ("(last '())", Nil),
            ("(last [])", Nil),
            ("(not [])", Bool(false)),
            ("(not '(1 2 3))", Bool(false)),
            ("(not nil)", Bool(true)),
            ("(not true)", Bool(false)),
            ("(not false)", Bool(true)),
            ("(not 1)", Bool(false)),
            ("(not 0)", Bool(false)),
            ("(not :foo)", Bool(false)),
            ("(not \"a\")", Bool(false)),
            ("(not \"\")", Bool(false)),
            ("(not (= 1 1))", Bool(false)),
            ("(not (= 1 2))", Bool(true)),
            ("(set nil)", Set(PersistentSet::new())),
            // NOTE: these all rely on an _unguaranteed_ insertion order...
            (
                "(set \"hi\")",
                set_with_values(vec![String("h".to_string()), String("i".to_string())]),
            ),
            ("(set '(1 2))", set_with_values(vec![Number(1), Number(2)])),
            (
                "(set '(1 2 1 2 1 2 2 2 2))",
                set_with_values(vec![Number(1), Number(2)]),
            ),
            (
                "(set [1 2 1 2 1 2 2 2 2])",
                set_with_values(vec![Number(1), Number(2)]),
            ),
            (
                "(set {1 2 3 4})",
                set_with_values(vec![
                    vector_with_values(vec![Number(1), Number(2)]),
                    vector_with_values(vec![Number(3), Number(4)]),
                ]),
            ),
            (
                "(set #{1 2 3 4})",
                set_with_values(vec![Number(1), Number(2), Number(3), Number(4)]),
            ),
            ("(set? #{1 2 3 4})", Bool(true)),
            ("(set? nil)", Bool(false)),
            ("(set? '())", Bool(false)),
            ("(set? [])", Bool(false)),
            ("(set? {})", Bool(false)),
            ("(set? #{})", Bool(true)),
            ("(set? \"a\")", Bool(false)),
            ("(set? :a)", Bool(false)),
            ("(set? 'a)", Bool(false)),
            ("(string? nil)", Bool(false)),
            ("(string? true)", Bool(false)),
            ("(string? false)", Bool(false)),
            ("(string? [1 2 3])", Bool(false)),
            ("(string? 1)", Bool(false)),
            ("(string? :hi)", Bool(false)),
            ("(string? \"hi\")", Bool(true)),
            ("(string? string?)", Bool(false)),
            ("(number? nil)", Bool(false)),
            ("(number? true)", Bool(false)),
            ("(number? false)", Bool(false)),
            ("(number? [1 2 3])", Bool(false)),
            ("(number? 1)", Bool(true)),
            ("(number? -1)", Bool(true)),
            ("(number? :hi)", Bool(false)),
            ("(number? \"hi\")", Bool(false)),
            ("(number? string?)", Bool(false)),
            ("(fn? nil)", Bool(false)),
            ("(fn? true)", Bool(false)),
            ("(fn? false)", Bool(false)),
            ("(fn? [1 2 3])", Bool(false)),
            ("(fn? 1)", Bool(false)),
            ("(fn? -1)", Bool(false)),
            ("(fn? :hi)", Bool(false)),
            ("(fn? \"hi\")", Bool(false)),
            ("(fn? string?)", Bool(true)),
            ("(fn? (fn* [a] a))", Bool(true)),
            ("(def! foo (fn* [a] a)) (fn? foo)", Bool(true)),
            ("(defmacro! foo (fn* [a] a)) (fn? foo)", Bool(true)),
            ("(conj (list) 1)", list_with_values(vec![Number(1)])),
            (
                "(conj (list 1) 2)",
                list_with_values(vec![Number(2), Number(1)]),
            ),
            (
                "(conj (list 1 2) 3)",
                list_with_values(vec![Number(3), Number(1), Number(2)]),
            ),
            (
                "(conj (list 2 3) 4 5 6)",
                list_with_values(vec![Number(6), Number(5), Number(4), Number(2), Number(3)]),
            ),
            (
                "(conj (list 1) (list 2 3))",
                list_with_values(vec![
                    list_with_values(vec![Number(2), Number(3)]),
                    Number(1),
                ]),
            ),
            ("(conj [] 1)", vector_with_values(vec![Number(1)])),
            (
                "(conj [1] 2)",
                vector_with_values(vec![Number(1), Number(2)]),
            ),
            (
                "(conj [1 2 3] 4)",
                vector_with_values(vec![Number(1), Number(2), Number(3), Number(4)]),
            ),
            (
                "(conj [1 2 3] 4 5)",
                vector_with_values(vec![Number(1), Number(2), Number(3), Number(4), Number(5)]),
            ),
            (
                "(conj '(1 2 3) 4 5)",
                list_with_values(vec![Number(5), Number(4), Number(1), Number(2), Number(3)]),
            ),
            (
                "(conj [3] [4 5])",
                vector_with_values(vec![
                    Number(3),
                    vector_with_values(vec![Number(4), Number(5)]),
                ]),
            ),
            (
                "(conj {:c :d} [1 2] {:a :b :c :e})",
                map_with_values(vec![
                    (
                        Keyword("c".to_string(), None),
                        Keyword("e".to_string(), None),
                    ),
                    (
                        Keyword("a".to_string(), None),
                        Keyword("b".to_string(), None),
                    ),
                    (Number(1), Number(2)),
                ]),
            ),
            (
                "(conj #{1 2} 1 3 2 2 2 2 1)",
                set_with_values(vec![Number(1), Number(2), Number(3)]),
            ),
            ("(macro? nil)", Bool(false)),
            ("(macro? true)", Bool(false)),
            ("(macro? false)", Bool(false)),
            ("(macro? [1 2 3])", Bool(false)),
            ("(macro? 1)", Bool(false)),
            ("(macro? -1)", Bool(false)),
            ("(macro? :hi)", Bool(false)),
            ("(macro? \"hi\")", Bool(false)),
            ("(macro? string?)", Bool(false)),
            ("(macro? {})", Bool(false)),
            ("(macro? (fn* [a] a))", Bool(false)),
            ("(def! foo (fn* [a] a)) (macro? foo)", Bool(false)),
            ("(defmacro! foo (fn* [a] a)) (macro? foo)", Bool(true)),
            ("(number? (time-ms))", Bool(true)),
            ("(seq nil)", Nil),
            ("(seq \"\")", Nil),
            (
                "(seq \"ab\")",
                list_with_values(vec![String("a".to_string()), String("b".to_string())]),
            ),
            ("(apply str (seq \"ab\"))", String("ab".to_string())),
            ("(seq '())", Nil),
            ("(seq '(1 2))", list_with_values(vec![Number(1), Number(2)])),
            ("(seq [])", Nil),
            ("(seq [1 2])", list_with_values(vec![Number(1), Number(2)])),
            ("(seq {})", Nil),
            (
                "(seq {1 2})",
                list_with_values(vec![vector_with_values(vec![Number(1), Number(2)])]),
            ),
            ("(seq #{})", Nil),
            ("(= (set '(1 2)) (set (seq #{1 2})))", Bool(true)),
            ("(zero? 0)", Bool(true)),
            ("(zero? 10)", Bool(false)),
            ("(zero? -10)", Bool(false)),
        ];
        run_eval_test(&test_cases);
    }
}
