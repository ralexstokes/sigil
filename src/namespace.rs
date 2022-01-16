use crate::value::{unbound_var, var_with_value, Value};
use std::collections::HashMap;
use thiserror::Error;

pub(crate) const DEFAULT_NAME: &str = "core";

#[derive(Debug, Error, Clone)]
pub enum NamespaceError {
    #[error("value found in namespace was not a Value::Var, instead {0}")]
    ValueInNamespaceWasNotVar(Value),
}

#[derive(Debug)]
// map from identifier to Value::Var
pub struct Namespace {
    pub name: String,
    bindings: HashMap<String, Value>,
}

impl Default for Namespace {
    fn default() -> Self {
        Self::new(DEFAULT_NAME)
    }
}

impl Namespace {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            bindings: HashMap::new(),
        }
    }

    pub fn get(&self, identifier: &str) -> Option<&Value> {
        self.bindings.get(identifier)
    }

    // NOTE: `value` must be a `Value::Var`
    fn insert(&mut self, identifier: &str, value: &Value) {
        self.bindings.insert(identifier.to_string(), value.clone());
    }

    // NOTE: `value` will be wrapped in a `Value::Var` which is stored in this namespace
    pub fn intern(&mut self, identifier: &str, value: &Value) -> Result<Value, NamespaceError> {
        match self.get(identifier) {
            Some(Value::Var(var)) => {
                var.update(value.clone());
                Ok(Value::Var(var.clone()))
            }
            Some(other) => Err(NamespaceError::ValueInNamespaceWasNotVar(other.clone())),
            None => {
                let var = var_with_value(value.clone(), &self.name, identifier);
                self.insert(identifier, &var);
                Ok(var)
            }
        }
    }

    pub fn intern_unbound(&mut self, identifier: &str) -> Value {
        let var = unbound_var(&self.name, identifier);
        self.insert(identifier, &var);
        var
    }

    pub fn remove(&mut self, identifier: &str) {
        self.bindings.remove(identifier);
    }

    pub fn merge(&mut self, other: &Namespace) -> Result<(), NamespaceError> {
        for (identifier, value) in &other.bindings {
            self.intern(identifier, value)?;
        }
        Ok(())
    }

    pub fn symbols(&self) -> impl Iterator<Item = &String> {
        self.bindings.keys()
    }
}
