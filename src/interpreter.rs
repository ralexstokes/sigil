use crate::reader::Form;
use std::default::Default;
use std::fmt;
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

pub struct Namespace {
    name: String,
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Default for Namespace {
    fn default() -> Self {
        Namespace {
            name: "sigil".to_string(),
        }
    }
}

#[derive(Default)]
pub struct Interpreter {
    pub current_namespace: Namespace,
    _env: Environment,
}

impl Interpreter {
    pub fn evaluate<'a>(&mut self, form: Form<'a>) -> Result<Form<'a>, EvaluationError> {
        let result = match form {
            other @ _ => other,
        };
        Ok(result)
    }
}
