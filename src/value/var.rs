use crate::reader::Symbol;
use crate::value::RuntimeValue;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::mem::discriminant;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Var {
    Bound(Rc<RefCell<RuntimeValue>>),
    Unbound,
}

impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        discriminant(self).hash(state);

        match self {
            Var::Bound(cell) => {
                let value = cell.borrow();
                value.hash(state);
            }
            Var::Unbound => {}
        }
    }
}

impl Var {
    pub fn new(value: RuntimeValue) -> Self {
        Var::Bound(Rc::new(RefCell::new(value)))
    }

    pub fn inner(&self) -> Option<RuntimeValue> {
        match self {
            Var::Bound(inner) => Some(inner.borrow().clone()),
            Var::Unbound => None,
        }
    }

    pub fn update(&mut self, value: RuntimeValue) {
        match self {
            Var::Bound(inner) => {
                let mut data = inner.borrow_mut();
                *data = value;
            }
            Var::Unbound => *self = Var::new(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocatedVar {
    pub symbol: Symbol,
    data: Var,
}

impl LocatedVar {
    pub fn new(symbol: &Symbol, data: Var) -> Self {
        Self {
            symbol: symbol.clone(),
            data,
        }
    }

    pub fn value(&self) -> RuntimeValue {
        match &self.data {
            Var::Bound(inner) => inner.borrow().clone(),
            Var::Unbound => RuntimeValue::UnboundVar(self.symbol.clone()),
        }
    }
}

impl Deref for LocatedVar {
    type Target = Var;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for LocatedVar {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
