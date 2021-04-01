use crate::namespace::Namespace;
use crate::prelude::{divide, multiply, plus, subtract};
use crate::value::Value;
use rpds::{
    HashTrieMap as PersistentMap, HashTrieSet as PersistentSet, List as PersistentList,
    Vector as PersistentVector,
};
use std::default::Default;
use std::fmt::Write;
use std::iter::FromIterator;
use std::rc::Rc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SymbolEvaluationError {
    #[error("namespace `{0}` not found for symbol `{1}`")]
    UndefinedNamespace(String, String),
    #[error("var `{0}` not found in namespace `{1}`")]
    MissingVar(String, String),
}

#[derive(Debug, Error)]
pub enum ListEvaluationError {
    #[error("cannot invoke the supplied value {0}")]
    CannotInvoke(Value),
    #[error("some failure...")]
    Failure,
}

#[derive(Debug, Error)]
pub enum PrimitiveEvaluationError {
    #[error("something failed {0}")]
    Failure(String),
}

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("symbol error: {0}")]
    Symbol(SymbolEvaluationError),
    #[error("list error: {0}")]
    List(ListEvaluationError),
    #[error("primitive error: {0}")]
    Primitve(PrimitiveEvaluationError),
}

#[derive(Debug)]
pub struct Interpreter {
    // index into `namespaces`
    current_namespace: usize,
    namespaces: Vec<Namespace>,
}

impl Default for Interpreter {
    fn default() -> Self {
        let bindings = [
            ("+", Value::Primitive(plus)),
            ("-", Value::Primitive(subtract)),
            ("*", Value::Primitive(multiply)),
            ("/", Value::Primitive(divide)),
        ];
        let default_namespace = Namespace::new("sigil", bindings.iter());

        Interpreter {
            current_namespace: 0,
            namespaces: vec![default_namespace],
        }
    }
}

impl Interpreter {
    pub fn current_namespace(&self) -> Namespace {
        self.namespaces[self.current_namespace].clone()
    }

    fn find_namespace(&self, ns_description: &str) -> Option<Namespace> {
        self.namespaces
            .iter()
            .find(|ns| ns.name() == ns_description)
            .map(|ns| ns.clone())
    }

    fn resolve_ns(&self, ns_opt: Option<&String>) -> Option<Namespace> {
        if let Some(ns_desc) = ns_opt {
            self.find_namespace(ns_desc)
        } else {
            Some(self.current_namespace())
        }
    }

    fn intern_var(&mut self, identifier: &str, value: Value) {
        let ns = self.current_namespace();
        ns.intern_value(identifier, value)
    }

    pub fn evaluate(&mut self, form: &Value) -> Result<Value, EvaluationError> {
        let result = match form {
            Value::Nil => Value::Nil,
            Value::Bool(b) => Value::Bool(*b),
            Value::Number(n) => Value::Number(*n),
            Value::String(s) => Value::String(s.to_string()),
            Value::Keyword(id, ns_opt) => {
                Value::Keyword(id.to_string(), ns_opt.as_ref().map(String::from))
            }
            Value::Symbol(id, ns_opt) => {
                if let Some(ns) = self.resolve_ns(ns_opt.as_ref()) {
                    if let Some(mut var) = ns.resolve_identifier(&id) {
                        // temporary
                        Rc::make_mut(&mut var).clone()
                    } else {
                        return Err(EvaluationError::Symbol(SymbolEvaluationError::MissingVar(
                            id.to_string(),
                            ns.name().to_string(),
                        )));
                    }
                } else {
                    let missing_ns = ns_opt.as_ref().unwrap();
                    let mut sym = String::new();
                    let _ = write!(&mut sym, "{}/{}", &missing_ns, id);
                    return Err(EvaluationError::Symbol(
                        SymbolEvaluationError::UndefinedNamespace(missing_ns.to_string(), sym),
                    ));
                }
            }
            Value::List(forms) => {
                if let Some(operator_form) = forms.first() {
                    match operator_form {
                        Value::Symbol(s, None) if s == "def!" => {
                            if let Some(rest) = forms.drop_first() {
                                if let Some(identifier) = rest.first() {
                                    match identifier {
                                        Value::Symbol(id, ns) => {
                                            if ns.is_none() {
                                                if let Some(rest) = rest.drop_first() {
                                                    if let Some(value_form) = rest.first() {
                                                        let value = self.evaluate(value_form)?;
                                                        self.intern_var(id, value.clone());
                                                        return Ok(value);
                                                    }
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure));
                        }
                        _ => match self.evaluate(operator_form)? {
                            Value::Primitive(native_fn) => {
                                let mut operands = vec![];
                                if let Some(rest) = forms.drop_first() {
                                    for operand_form in rest.iter() {
                                        let operand = self.evaluate(operand_form)?;
                                        operands.push(operand);
                                    }
                                }
                                return native_fn(&operands);
                            }
                            v @ _ => {
                                return Err(EvaluationError::List(
                                    ListEvaluationError::CannotInvoke(v),
                                ));
                            }
                        },
                    }
                }
                Value::List(PersistentList::new())
            }
            Value::Vector(forms) => {
                let mut result = vec![];
                for form in forms.into_iter() {
                    let value = self.evaluate(form)?;
                    result.push(value);
                }
                Value::Vector(PersistentVector::from_iter(result.into_iter()))
            }
            Value::Map(forms) => {
                let mut result = vec![];
                for (k, v) in forms.into_iter() {
                    let key = self.evaluate(k)?;
                    let value = self.evaluate(v)?;
                    result.push((key, value));
                }
                Value::Map(PersistentMap::from_iter(result.into_iter()))
            }
            Value::Set(forms) => {
                let mut result = vec![];
                for form in forms.into_iter() {
                    let value = self.evaluate(form)?;
                    result.push(value);
                }
                Value::Set(PersistentSet::from_iter(result.into_iter()))
            }
            Value::Primitive(_) => unreachable!(),
        };
        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::reader::read;

    #[test]
    fn test_basic_eval() {
        use Value::*;

        let mut interpreter = Interpreter::default();
        let test_cases = vec![
            ("nil", Nil),
            ("1337", Number(1337)),
            ("-1337", Number(-1337)),
            ("(+)", Number(0)),
            ("(+ 1)", Number(1)),
            ("(+ 1 10)", Number(11)),
            ("(+ 1 10 2)", Number(13)),
            ("(- 1)", Number(-1)),
            ("(- 10 9)", Number(1)),
            ("(- 10 20)", Number(-10)),
            ("(- 10 20 10)", Number(-20)),
            ("(*)", Number(1)),
            ("(* 2)", Number(2)),
            ("(* 2 3)", Number(6)),
            ("(* 2 3 1 1 1)", Number(6)),
            ("(/ 2)", Number(0)),
            ("(/ 1)", Number(1)),
            ("(/ 22 2)", Number(11)),
            ("(/ 22 2 1 1 1)", Number(11)),
            ("(/ 22 2 1 1 1)", Number(11)),
            ("(+ 2 (* 3 4))", Number(14)),
        ];

        for (input, expected) in test_cases {
            let read_result = read(input).unwrap();
            let form = &read_result[0];
            let result = interpreter.evaluate(form).unwrap();
            assert_eq!(result, expected)
        }
    }
}
