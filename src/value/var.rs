use crate::value::RuntimeValue;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::mem::discriminant;
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

    pub fn value(&self) -> RuntimeValue {
        match self {
            Var::Bound(inner) => inner.borrow().clone(),
            // NOTE: this is a bit of a pun, in lieu of having anything more descriptive to provide for now
            Var::Unbound => RuntimeValue::Var(self.clone()),
        }
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
