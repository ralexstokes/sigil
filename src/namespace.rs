use crate::value::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::iter::FromIterator;
use std::rc::Rc;

pub type Var = Rc<Value>;

#[derive(Debug, PartialEq, Eq)]
pub struct Namespace(Rc<RefCell<NamespaceInner>>);

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.borrow().name)
    }
}

impl Namespace {
    pub fn new<'a>(name: &'a str, values: impl Iterator<Item = &'a (&'a str, Value)>) -> Self {
        let bindings = values
            .into_iter()
            .map(|(name, value)| (name.to_string(), Rc::new(value.clone())));
        Self(Rc::new(RefCell::new(NamespaceInner {
            name: name.to_string(),
            bindings: HashMap::from_iter(bindings),
        })))
    }

    pub fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }

    pub fn name(&self) -> String {
        self.0.borrow().name.clone()
    }

    /// `resolve_identifier` returns the `Value` wrapped by the `Var`
    /// referenced in this `NamespaceInner` by `identifier`.
    pub fn resolve_identifier(&self, identifier: &str) -> Option<Var> {
        self.0.borrow().bindings.get(identifier).map(Rc::clone)
    }

    pub fn intern_value(&self, identifier: &str, value: Value) {
        self.0
            .borrow_mut()
            .bindings
            .insert(identifier.to_string(), Rc::new(value));
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct NamespaceInner {
    pub name: String,
    pub bindings: HashMap<String, Var>,
}
