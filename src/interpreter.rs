use crate::namespace::Namespace;
use crate::prelude::{divide, multiply, plus, subtract};
use crate::value::Value;
use itertools::Itertools;
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
    #[error("some failure: {0}")]
    Failure(String),
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
    // stack of namespaces
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
            namespaces: vec![default_namespace],
        }
    }
}

impl Interpreter {
    pub fn current_namespace(&self) -> Namespace {
        // NOTE: error if scopes underflow...
        self.namespaces.last().unwrap().clone()
    }

    fn current_namespace_with_cursor(&self) -> (Namespace, usize) {
        (self.current_namespace(), self.namespaces.len() - 1)
    }

    fn find_namespace(&self, ns_description: &str) -> Option<(Namespace, usize)> {
        self.namespaces
            .iter()
            .enumerate()
            .find(|(_, ns)| ns.name() == ns_description)
            .map(|(cursor, ns)| (ns.clone(), cursor))
    }

    fn intern_var(&mut self, identifier: &str, value: Value) {
        let ns = self.current_namespace();
        ns.intern_value(identifier, value)
    }

    fn resolve_var(&mut self, identifier: &str, namespace: &Namespace) -> Option<Value> {
        namespace
            .resolve_identifier(&identifier)
            .map(|mut var| Rc::make_mut(&mut var).clone())
    }

    fn resolve_symbol(
        &mut self,
        identifier: &str,
        ns_opt: Option<&String>,
    ) -> Result<Value, EvaluationError> {
        let (mut namespace, cursor) = match ns_opt {
            Some(ns_desc) => {
                self.find_namespace(ns_desc).ok_or_else(|| {
                    // ns is `Some` but missing in `self`...
                    let missing_ns = ns_opt.as_ref().unwrap();
                    let mut sym = String::new();
                    let _ = write!(&mut sym, "{}/{}", &missing_ns, identifier);
                    EvaluationError::Symbol(SymbolEvaluationError::UndefinedNamespace(
                        missing_ns.to_string(),
                        sym,
                    ))
                })
            }
            None => Ok(self.current_namespace_with_cursor()),
        }?;
        if let Some(var) = self.resolve_var(&identifier, &namespace) {
            return Ok(var);
        }
        for index in (0..cursor).rev() {
            namespace = self.namespaces[index].clone();
            if let Some(var) = self.resolve_var(&identifier, &namespace) {
                return Ok(var);
            }
        }
        Err(EvaluationError::Symbol(SymbolEvaluationError::MissingVar(
            identifier.to_string(),
            namespace.name().to_string(),
        )))
    }

    fn enter_scope(&mut self) {
        self.namespaces.push(Namespace::default())
    }

    fn leave_scope(&mut self) {
        // NOTE: error if scopes underflow...
        let _ = self.namespaces.pop().unwrap();
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
                return self.resolve_symbol(id, ns_opt.as_ref());
            }
            Value::List(forms) => {
                if let Some(operator_form) = forms.first() {
                    match operator_form {
                        // (def! symbol value)
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
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `def!`".to_string(),
                            )));
                        }
                        // (let* [bindings*] body)
                        Value::Symbol(s, None) if s == "let*" => {
                            if let Some(rest) = forms.drop_first() {
                                if let Some(bindings) = rest.first() {
                                    match bindings {
                                        Value::Vector(elems) => {
                                            if elems.len() % 2 == 0 {
                                                if let Some(body) = rest.drop_first() {
                                                    self.enter_scope();
                                                    for (name, value_form) in elems.iter().tuples()
                                                    {
                                                        match name {
                                                            Value::Symbol(s, None) => {
                                                                let value =
                                                                    self.evaluate(value_form)?;
                                                                self.intern_var(s, value)
                                                            }
                                                            _ => {
                                                                self.leave_scope();
                                                                return Err(EvaluationError::List(
                                                                    ListEvaluationError::Failure(
                                                                        "could not evaluate `let*`"
                                                                            .to_string(),
                                                                    ),
                                                                ));
                                                            }
                                                        }
                                                    }
                                                    let form = body.push_front(Value::Symbol(
                                                        "do".to_string(),
                                                        None,
                                                    ));
                                                    let result = self.evaluate(&Value::List(form));
                                                    self.leave_scope();
                                                    return result;
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `let*`".to_string(),
                            )));
                        }
                        // (do forms*)
                        Value::Symbol(s, None) if s == "do" => {
                            if let Some(rest) = forms.drop_first() {
                                return rest
                                    .iter()
                                    .try_fold(Value::Nil, |_, next| self.evaluate(next));
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `do`".to_string(),
                            )));
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
            ("(do 1 2 3)", Number(3)),
            ("(def! a 3)", Number(3)),
            ("(do (def! a 3) (+ a 1))", Number(4)),
            ("(let* [] )", Nil),
            ("(let* [a 1] )", Nil),
            ("(let* [a 3] a)", Number(3)),
            ("(let* [a 3] (+ a 5))", Number(8)),
            (
                "(let* [a 3 b 33] (+ a (let* [c 4] (+ c 1)) b 5))",
                Number(46),
            ),
            ("(do (def! a 1) (let* [a 3] a))", Number(3)),
            ("(do (def! a 1) (let* [a 3] a) a)", Number(1)),
        ];

        for (input, expected) in test_cases {
            let read_result = read(input).unwrap();
            let form = &read_result[0];
            let result = interpreter.evaluate(form).unwrap();
            assert_eq!(result, expected)
        }
    }
}
