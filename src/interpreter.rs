use crate::reader::Form;
use itertools::join;
use rpds::{HashTrieMap as PMap, HashTrieSet as PSet, List as PList, Vector as PVector};
use std::cmp;
use std::default::Default;
use std::fmt;
use std::iter::FromIterator;
use std::rc::Rc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("unknown")]
    Unknown,
}

type Namespace = Rc<NamespaceInner>;

fn namespace_with_name(name: &str) -> Namespace {
    Rc::new(NamespaceInner {
        name: name.to_string(),
    })
}

#[derive(Debug, Eq, Hash)]
pub struct NamespaceInner {
    name: String,
}

impl cmp::PartialEq for NamespaceInner {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl fmt::Display for NamespaceInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
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

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Nil,
    Bool(bool),
    Number(u64),
    String(String),
    Keyword(String, Option<Namespace>),
    Symbol(String, Option<Namespace>),
    List(PList<Value>),
    Vector(PVector<Value>),
    // Map(PMap<Value, Value>),
    // Set(PSet<Value>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Value::*;

        match self {
            Nil => write!(f, "nil"),
            Bool(ref b) => write!(f, "{}", b),
            Number(ref n) => write!(f, "{}", n),
            String(ref s) => write!(f, "\"{}\"", s),
            Keyword(ref id, ref ns_opt) => {
                write!(f, ":")?;
                if let Some(ns) = ns_opt {
                    write!(f, "{}/", ns)?;
                }
                write!(f, "{}", id)
            }
            Symbol(ref id, ref ns_opt) => {
                if let Some(ns) = ns_opt {
                    write!(f, "{}/", ns)?;
                }
                write!(f, "{}", id)
            }
            List(elems) => write!(f, "({})", join(elems, " ")),
            Vector(elems) => write!(f, "[{}]", join(elems, " ")),
            // Map(elems) => write!(f, "{{{}}}", join(elems, " ")),
            // Set(elems) => write!(f, "#{{{}}}", join(elems, " ")),
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
                Value::List(PList::from_iter(result.into_iter()))
            }
            Form::Vector(forms) => {
                let mut result = vec![];
                for form in forms.into_iter() {
                    let value = self.evaluate(form)?;
                    result.push(value);
                }
                Value::Vector(PVector::from_iter(result.into_iter()))
            }
            // Form::Map(Vec<Form<'a>>) =>,
            // Form::Set(forms) => {
            //     let mut result = vec![];
            //     for form in forms.into_iter() {
            //         let value = self.evaluate(form)?;
            //         result.push(value);
            //     }
            //     Value::Set(PSet::from_iter(result.into_iter()))
            // }
            _ => Value::Nil,
        };
        Ok(result)
    }
}
