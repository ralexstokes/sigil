use crate::reader::Form;
use std::cmp;
use std::default::Default;
use std::fmt;
use std::rc::Rc;
use thiserror::Error;

#[derive(Debug)]
pub struct Environment;

impl Default for Environment {
    fn default() -> Self {
        Environment {}
    }
}

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("unknown")]
    Unknown,
}

type Namespace = Rc<NamespaceInner>;

#[derive(Debug)]
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

impl Default for NamespaceInner {
    fn default() -> Self {
        NamespaceInner {
            name: "sigil".to_string(),
        }
    }
}

#[derive(Debug)]
pub struct Interpreter {
    // index into `namespaces`
    current_namespace: usize,
    namespaces: Vec<Namespace>,
    _env: Environment,
}

impl Default for Interpreter {
    fn default() -> Self {
        let namespaces = vec![Rc::new(NamespaceInner::default())];
        Interpreter {
            current_namespace: 0,
            namespaces: namespaces,
            _env: Environment::default(),
        }
    }
}

impl Interpreter {
    pub fn current_namespace(&self) -> Namespace {
        self.namespaces[self.current_namespace].clone()
    }

    pub fn evaluate<'a>(&mut self, form: Form<'a>) -> Result<Form<'a>, EvaluationError> {
        use Form::*;

        let result = match form {
            Number(n) => {
                // demo name space switch in repl
                self.namespaces.push(Rc::new(NamespaceInner {
                    name: n.to_string(),
                }));
                self.current_namespace += 1;
                Number(n)
            }
            other @ _ => other,
        };
        Ok(result)
    }
}
