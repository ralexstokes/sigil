use crate::value::Value;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

pub type Var = Rc<Value>;

pub type Namespace = Rc<NamespaceInner>;

pub fn namespace_with_name(name: &str) -> Namespace {
    Rc::new(NamespaceInner {
        name: name.to_string(),
        bindings: HashMap::new(),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamespaceInner {
    pub name: String,
    pub bindings: HashMap<String, Var>,
}

impl NamespaceInner {
    /// `resolve_identifier` returns the `Value` wrapped by the `Var`
    /// referenced in this `Namespace` by `identifier`.
    pub fn resolve_identifier(&self, identifier: &str) -> Option<Value> {
        Some(Value::Symbol(
            identifier.to_string(),
            Some(self.to_string()),
        ))
    }
}

impl fmt::Display for NamespaceInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}
