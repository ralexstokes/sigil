use crate::value::RuntimeValue;

use std::cell::{Ref, RefCell};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AtomRef(Rc<RefCell<RuntimeValue>>);

impl Hash for AtomRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let inner = self.0.borrow();
        inner.hash(state);
    }
}

impl AtomRef {
    pub fn new(value: RuntimeValue) -> Self {
        AtomRef(Rc::new(RefCell::new(value)))
    }

    pub fn value(&self) -> Ref<'_, RuntimeValue> {
        self.0.borrow()
    }

    pub fn reset(&self, value: &RuntimeValue) -> RuntimeValue {
        let mut inner = self.0.borrow_mut();
        *inner = value.clone();
        value.clone()
    }
}
