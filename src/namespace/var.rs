use crate::value::RuntimeValue;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Var {
    Bound(Rc<RefCell<RuntimeValue>>),
    Unbound,
}

pub fn new_var(value: RuntimeValue) -> Var {
    Var::Bound(Rc::new(RefCell::new(value)))
}
