mod var;

use crate::reader::Identifier;
use crate::value::{unbound_var, var_with_value, Value};
use std::collections::HashMap;
use thiserror::Error;

pub use var::Var;

const DEFAULT_NAME: Identifier = Identifier::from("core");

// `Context` maps global names to namespaces
// and also tracks a "current" namespace, the one which
// a non-local identifier is assumed to refer to
#[derive(Debug)]
pub struct Context {
    names: HashMap<Identifier, usize>,
    name_lookup: HashMap<usize, Identifier>,
    current_namespace: usize,
    namespaces: Vec<Namespace>,
}

impl Default for Context {
    fn default() -> Self {
        let current_namespace = 0;
        let names = HashMap::from([(DEFAULT_NAME, current_namespace)]);
        let name_lookup = HashMap::from([(current_namespace, DEFAULT_NAME.clone())]);
        let namespaces = vec![Namespace::default()];

        Self {
            names,
            name_lookup,
            current_namespace,
            namespaces,
        }
    }
}

impl Context {
    pub fn current_namespace(&self) -> &Namespace {
        &self.namespaces[self.current_namespace]
    }

    fn resolve_reference_in_namespace(
        &self,
        namespace: &Namespace,
        identifier: &Identifier,
    ) -> Option<Var> {
        namespace.get(identifier).map(|value| Var(value.clone()))
    }

    pub fn resolve_reference(
        &self,
        name: &Identifier,
        identifier: &Identifier,
    ) -> Result<Var, NamespaceError> {
        let namespace_index = self
            .names
            .get(name)
            .ok_or_else(|| NamespaceError::MissingNamespace(name.clone()))?;

        let namespace = &self.namespaces[*namespace_index];

        self.resolve_reference_in_namespace(namespace, identifier)
            .ok_or_else(|| NamespaceError::MissingIdentifier(identifier.clone(), name.clone()))
    }

    pub fn resolve_reference_in_current_namespace(
        &self,
        identifier: &Identifier,
    ) -> Result<Var, NamespaceError> {
        self.resolve_reference_in_namespace(self.current_namespace(), identifier)
            .ok_or_else(|| {
                let name = self.name_lookup.get(&self.current_namespace).unwrap();
                NamespaceError::MissingIdentifier(identifier.clone(), name.clone())
            })
    }
}

#[derive(Debug, Error, Clone)]
pub enum NamespaceError {
    #[error("value found in namespace was not a Value::Var, instead {0}")]
    ValueInNamespaceWasNotVar(Value),
    #[error("namespace {0} was not found")]
    MissingNamespace(Identifier),
    #[error("identifier {0} was not found in namespace {1}")]
    MissingIdentifier(Identifier, Identifier),
}

#[derive(Debug)]
pub struct Namespace {
    pub name: String,
    bindings: HashMap<Identifier, Value>,
}

impl Default for Namespace {
    fn default() -> Self {
        Self::new(&DEFAULT_NAME)
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
