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

pub fn new_var(value: RuntimeValue) -> Var {
    Var::Bound(Rc::new(RefCell::new(value)))
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
