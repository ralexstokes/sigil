use crate::value::RuntimeValue;

use std::cell::{Ref, RefCell};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AtomImpl(Rc<RefCell<RuntimeValue>>);

impl Hash for AtomImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let inner = self.0.borrow();
        inner.hash(state);
    }
}

impl AtomImpl {
    pub fn value(&self) -> Ref<'_, RuntimeValue> {
        self.0.borrow()
    }

    pub fn reset(&self, value: &RuntimeValue) -> RuntimeValue {
        let mut inner = self.0.borrow_mut();
        *inner = value.clone();
        value.clone()
    }
}

pub fn new_atom(data: RuntimeValue) -> RuntimeValue {
    RuntimeValue::Atom(AtomImpl(Rc::new(RefCell::new(data))))
}
