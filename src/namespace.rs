use std::fmt;
use std::rc::Rc;
pub type Namespace = Rc<NamespaceInner>;

pub fn namespace_with_name(name: &str) -> Namespace {
    Rc::new(NamespaceInner {
        name: name.to_string(),
    })
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NamespaceInner {
    pub name: String,
}

impl fmt::Display for NamespaceInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}
