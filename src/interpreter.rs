use crate::namespace::{namespace_with_name, Namespace};
use crate::value::Value;
use rpds::{
    HashTrieMap as PersistentMap, HashTrieSet as PersistentSet, List as PersistentList,
    Vector as PersistentVector,
};
use std::default::Default;
use std::fmt::Write;
use std::iter::FromIterator;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SymbolEvaluationError {
    #[error("namespace `{0}` not found for symbol `{1}`")]
    UndefinedNamespace(String, String),
    #[error("var `{0}` not found in namespace `{1}`")]
    MissingVar(String, String),
}

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("symbol error: {0}")]
    Symbol(SymbolEvaluationError),
}

#[derive(Debug)]
pub struct Interpreter {
    // index into `namespaces`
    current_namespace: usize,
    namespaces: Vec<Namespace>,
}

impl Default for Interpreter {
    fn default() -> Self {
        Interpreter {
            current_namespace: 0,
            namespaces: vec![namespace_with_name("sigil")],
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
            .find(|ns| ns.name == ns_description)
            .map(|ns| ns.clone())
    }

    fn resolve_ns(&self, ns_opt: Option<&String>) -> Option<Namespace> {
        if let Some(ns_desc) = ns_opt {
            self.find_namespace(ns_desc)
        } else {
            Some(self.current_namespace())
        }
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
                    if let Some(value) = ns.resolve_identifier(&id) {
                        value
                    } else {
                        return Err(EvaluationError::Symbol(SymbolEvaluationError::MissingVar(
                            id.to_string(),
                            ns.name.clone(),
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
                let mut result = vec![];
                for form in forms.into_iter() {
                    let value = self.evaluate(form)?;
                    result.push(value);
                }
                Value::List(PersistentList::from_iter(result.into_iter()))
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
        };
        Ok(result)
    }
}
