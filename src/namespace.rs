use crate::value::Value;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

pub type Var = Rc<Value>;

pub fn namespace_with_name(name: &str) -> Namespace {
    Namespace {
        name: name.to_string(),
        bindings: HashMap::new(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Namespace {
    pub name: String,
    pub bindings: HashMap<String, Var>,
}

impl Namespace {
    /// `resolve_identifier` returns the `Value` wrapped by the `Var`
    /// referenced in this `Namespace` by `identifier`.
    pub fn resolve_identifier(&mut self, identifier: &str) -> Option<Value> {
        if let Some(var) = self.bindings.get_mut(identifier) {
            Some(Rc::make_mut(var).clone())
        } else {
            Some(Value::Symbol(
                identifier.to_string(),
                Some(self.name.clone()),
            ))
        }
    }

    pub fn intern_value(&mut self, identifier: &str, value: Value) {
        self.bindings.insert(identifier.to_string(), Rc::new(value));
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}
