mod var;

use crate::reader::{Identifier, Symbol};
use crate::value::{unbound_var, var_with_value, RuntimeValue, Value};
use std::collections::HashMap;
use thiserror::Error;
use var::new_var;

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

    fn get_namespace(&self, identifier: &Identifier) -> &Namespace {
        let index = self.names[identifier];
        &self.namespaces[index]
    }

    pub fn resolve_symbol(&self, symbol: &Symbol) -> Result<Var, NamespaceError> {
        match symbol {
            Symbol {
                identifier,
                namespace: Some(name),
            } => self.resolve_reference(name, identifier),
            Symbol {
                identifier,
                namespace: None,
            } => self.resolve_reference_in_current_namespace(identifier),
        }
    }

    fn resolve_reference(
        &self,
        name: &Identifier,
        identifier: &Identifier,
    ) -> Result<Var, NamespaceError> {
        let namespace_index = self
            .names
            .get(name)
            .ok_or_else(|| NamespaceError::MissingNamespace(name.clone()))?;

        let namespace = &self.namespaces[*namespace_index];

        namespace
            .get(identifier)
            .ok_or_else(|| NamespaceError::MissingIdentifier(identifier.clone(), name.clone()))
    }

    fn resolve_reference_in_current_namespace(
        &self,
        identifier: &Identifier,
    ) -> Result<Var, NamespaceError> {
        self.current_namespace().get(identifier).ok_or_else(|| {
            let name = self.name_lookup.get(&self.current_namespace).unwrap();
            NamespaceError::MissingIdentifier(identifier.clone(), name.clone())
        })
    }
}

#[derive(Debug, Error, Clone)]
pub enum NamespaceError {
    #[error("namespace {0} was not found")]
    MissingNamespace(Identifier),
    #[error("identifier {0} was not found in namespace {1}")]
    MissingIdentifier(Identifier, Identifier),
}

#[derive(Debug, Default)]
pub struct Namespace {
    bindings: HashMap<Identifier, Var>,
}

impl Namespace {
    pub fn get(&self, identifier: &Identifier) -> Option<Var> {
        self.bindings.get(identifier).map(|var| var.clone())
    }

    pub fn intern(
        &mut self,
        identifier: &Identifier,
        value: Option<RuntimeValue>,
    ) -> Result<Var, NamespaceError> {
        let var = self
            .bindings
            .entry(identifier.clone())
            .or_insert_with(|| Var::Unbound);

        if let Some(value) = value {
            *var = new_var(value)
        }
        Ok(var.clone())
    }

    pub fn remove(&mut self, identifier: &str) {
        self.bindings.remove(identifier);
    }

    // pub fn merge(&mut self, other: &Namespace) -> Result<(), NamespaceError> {
    //     for (identifier, value) in &other.bindings {
    //         self.intern(identifier, value)?;
    //     }
    //     Ok(())
    // }

    pub fn symbols(&self) -> impl Iterator<Item = &String> {
        self.bindings.keys()
    }
}
