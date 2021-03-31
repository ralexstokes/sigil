use crate::namespace::{namespace_with_name, Namespace};
use crate::reader::Form;
use crate::value::Value;
use itertools::Itertools;
use rpds::{
    HashTrieMap as PersistentMap, HashTrieSet as PersistentSet, List as PersistentList,
    Vector as PersistentVector,
};
use std::default::Default;
use std::iter::FromIterator;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MapEvaluationError {
    #[error("incorrect arity in {0}")]
    IncorrectArity(String),
}

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("error: {0} for map")]
    Map(MapEvaluationError),
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
            .find(|&ns| ns.name == ns_description)
            .map(|ns| ns.clone())
    }

    fn find_or_create_namespace(&self, ns_description: &str) -> Option<Namespace> {
        self.find_namespace(ns_description)
            .or_else(|| Some(namespace_with_name(ns_description)))
    }

    pub fn evaluate(&mut self, form: Form) -> Result<Value, EvaluationError> {
        let result = match form {
            Form::Nil => Value::Nil,
            Form::Bool(b) => Value::Bool(b),
            Form::Number(n) => Value::Number(n),
            Form::String(s) => Value::String(s.to_string()),
            Form::Keyword(id, ns_opt) => {
                let ns = ns_opt.and_then(|ns| self.find_or_create_namespace(ns));
                Value::Keyword(id.to_string(), ns)
            }
            Form::Symbol(id, ns_opt) => {
                let ns = ns_opt.and_then(|ns| self.find_or_create_namespace(ns));
                Value::Symbol(id.to_string(), ns)
            }
            Form::List(forms) => {
                let mut result = vec![];
                for form in forms.into_iter() {
                    let value = self.evaluate(form)?;
                    result.push(value);
                }
                Value::List(PersistentList::from_iter(result.into_iter()))
            }
            Form::Vector(forms) => {
                let mut result = vec![];
                for form in forms.into_iter() {
                    let value = self.evaluate(form)?;
                    result.push(value);
                }
                Value::Vector(PersistentVector::from_iter(result.into_iter()))
            }
            Form::Map(forms) => {
                if forms.len() % 2 != 0 {
                    return Err(EvaluationError::Map(MapEvaluationError::IncorrectArity(
                        "constructor".to_string(),
                    )));
                }
                let mut result = vec![];
                for (k, v) in forms.into_iter().tuples() {
                    let key = self.evaluate(k)?;
                    let value = self.evaluate(v)?;
                    result.push((key, value));
                }
                Value::Map(PersistentMap::from_iter(result.into_iter()))
            }
            Form::Set(forms) => {
                let mut result = vec![];
                for form in forms.into_iter() {
                    let value = self.evaluate(form)?;
                    result.push(value);
                }
                Value::Set(PersistentSet::from_iter(result.into_iter()))
            }
            _ => Value::Nil,
        };
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Value::*;

    #[test]
    fn test_ord_provided() {
        let ref x = List(PersistentList::from_iter(vec![
            Number(1),
            Number(2),
            Number(3),
        ]));
        let ref y = List(PersistentList::from_iter(vec![
            Number(2),
            Number(3),
            Number(1),
        ]));
        let ref z = List(PersistentList::from_iter(vec![Number(44)]));
        let ref a = List(PersistentList::from_iter(vec![Number(0)]));
        let ref b = List(PersistentList::from_iter(vec![Number(1)]));
        let ref c = List(PersistentList::new());

        assert_eq!(x.cmp(x), Ordering::Equal);
        assert_eq!(x.cmp(y), Ordering::Less);
        assert_eq!(x.cmp(z), Ordering::Less);
        assert_eq!(x.cmp(a), Ordering::Greater);
        assert_eq!(x.cmp(b), Ordering::Greater);
        assert_eq!(x.cmp(c), Ordering::Greater);
        assert_eq!(c.cmp(x), Ordering::Less);
        assert_eq!(c.cmp(y), Ordering::Less);
    }

    #[test]
    fn test_ord_custom() {
        let ref x = Map(PersistentMap::from_iter(vec![
            (Number(1), Number(2)),
            (Number(3), Number(4)),
        ]));
        let ref y = Map(PersistentMap::from_iter(vec![(Number(1), Number(2))]));
        let ref z = Map(PersistentMap::from_iter(vec![
            (Number(4), Number(3)),
            (Number(1), Number(2)),
        ]));
        let ref a = Map(PersistentMap::from_iter(vec![
            (Number(1), Number(444)),
            (Number(3), Number(4)),
        ]));
        let ref b = Map(PersistentMap::new());
        let ref c = Map(PersistentMap::from_iter(vec![
            (Number(1), Number(2)),
            (Number(3), Number(4)),
            (Number(4), Number(8)),
        ]));

        assert_eq!(x.cmp(x), Ordering::Equal);
        assert_eq!(x.cmp(y), Ordering::Greater);
        assert_eq!(x.cmp(z), Ordering::Less);
        assert_eq!(x.cmp(a), Ordering::Less);
        assert_eq!(x.cmp(b), Ordering::Greater);
        assert_eq!(x.cmp(c), Ordering::Less);
        assert_eq!(b.cmp(b), Ordering::Equal);
        assert_eq!(b.cmp(c), Ordering::Less);
        assert_eq!(b.cmp(y), Ordering::Less);
    }
}
