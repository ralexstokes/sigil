use crate::interpreter::{EvaluationError, Interpreter};
use itertools::{join, sorted};
use rpds::{
    HashTrieMap as PersistentMap, HashTrieSet as PersistentSet, List as PersistentList,
    Vector as PersistentVector,
};
use std::cell::RefCell;
use std::cmp::{Eq, Ord, Ordering, PartialEq};
use std::fmt;
use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::iter::{FromIterator, IntoIterator};
use std::mem::discriminant;
use std::rc::Rc;

pub fn list_with_values(values: impl IntoIterator<Item = Value>) -> Value {
    Value::List(PersistentList::from_iter(values))
}

pub fn vector_with_values(values: impl IntoIterator<Item = Value>) -> Value {
    Value::Vector(PersistentVector::from_iter(values))
}

pub fn map_with_values(values: impl IntoIterator<Item = (Value, Value)>) -> Value {
    Value::Map(PersistentMap::from_iter(values))
}

pub fn set_with_values(values: impl IntoIterator<Item = Value>) -> Value {
    Value::Set(PersistentSet::from_iter(values))
}

pub fn var_with_value(value: Value, namespace: &str, identifier: &str) -> Value {
    Value::Var(VarImpl {
        inner: Rc::new(RefCell::new(value)),
        namespace: namespace.to_string(),
        identifier: identifier.to_string(),
    })
}

pub fn atom_with_value(value: Value) -> Value {
    Value::Atom(Rc::new(RefCell::new(value)))
}

pub fn var_impl_into_inner(var: &VarImpl) -> Value {
    var.inner.borrow().clone()
}

pub fn var_into_inner(value: Value) -> Value {
    match value {
        Value::Var(v) => var_impl_into_inner(&v),
        _ => panic!("called with non Var value"),
    }
}

pub fn atom_impl_into_inner(atom: &AtomImpl) -> Value {
    atom.borrow().clone()
}

pub fn exception(msg: &str, data: &Value) -> Value {
    Value::Exception(ExceptionImpl {
        message: msg.to_string(),
        data: Box::new(data.clone()),
    })
}

pub type NativeFn = fn(&mut Interpreter, &[Value]) -> Result<Value, EvaluationError>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lambda {
    pub body: PersistentList<Value>,
    pub arity: usize,
    // allow for nested fns
    pub level: usize,
}

#[derive(Clone)]
pub struct VarImpl {
    pub inner: Rc<RefCell<Value>>,
    namespace: String,
    identifier: String,
}

type AtomImpl = Rc<RefCell<Value>>;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExceptionImpl {
    message: String,
    data: Box<Value>,
}

#[derive(Clone)]
pub enum Value {
    Nil,
    Bool(bool),
    Number(i64),
    String(String),
    // identifier with optional namespace
    Keyword(String, Option<String>),
    // identifier with optional namespace
    Symbol(String, Option<String>),
    List(PersistentList<Value>),
    Vector(PersistentVector<Value>),
    Map(PersistentMap<Value, Value>),
    Set(PersistentSet<Value>),
    Fn(Lambda),
    Primitive(NativeFn),
    Var(VarImpl),
    Recur(PersistentVector<Value>),
    Atom(AtomImpl),
    Macro(Lambda),
    Exception(ExceptionImpl),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        use Value::*;

        match self {
            Nil => match other {
                Nil => true,
                _ => false,
            },
            Bool(ref x) => match other {
                Bool(ref y) => x == y,
                _ => false,
            },
            Number(ref x) => match other {
                Number(ref y) => x == y,
                _ => false,
            },
            String(ref x) => match other {
                String(ref y) => x == y,
                _ => false,
            },
            Keyword(ref x, ref x_ns_opt) => match other {
                Keyword(ref y, ref y_ns_opt) => (x, x_ns_opt) == (y, y_ns_opt),
                _ => false,
            },
            Symbol(ref x, ref x_ns_opt) => match other {
                Symbol(ref y, ref y_ns_opt) => (x, x_ns_opt) == (y, y_ns_opt),
                _ => false,
            },
            List(ref x) => match other {
                List(ref y) => x == y,
                _ => false,
            },
            Vector(ref x) => match other {
                Vector(ref y) => x == y,
                _ => false,
            },
            Map(ref x) => match other {
                Map(ref y) => x == y,
                _ => false,
            },
            Set(ref x) => match other {
                Set(ref y) => x == y,
                _ => false,
            },
            Fn(ref x) => match other {
                Fn(ref y) => x == y,
                _ => false,
            },
            Primitive(x) => match other {
                Primitive(y) => {
                    let x_ptr = x as *const NativeFn;
                    let x_identifier =
                        unsafe { std::mem::transmute::<*const NativeFn, usize>(x_ptr) };
                    let y_ptr = y as *const NativeFn;
                    let y_identifier =
                        unsafe { std::mem::transmute::<*const NativeFn, usize>(y_ptr) };
                    x_identifier == y_identifier
                }
                _ => false,
            },
            Var(VarImpl {
                namespace: namespace_x,
                identifier: identifier_x,
                ..
            }) => match other {
                Var(VarImpl {
                    namespace: namespace_y,
                    identifier: identifier_y,
                    ..
                }) => (namespace_x, identifier_x) == (namespace_y, identifier_y),
                _ => false,
            },
            Recur(ref x) => match other {
                Recur(ref y) => x == y,
                _ => false,
            },
            Atom(ref x) => match other {
                Atom(ref y) => x == y,
                _ => false,
            },
            Macro(ref x) => match other {
                Macro(ref y) => x == y,
                _ => false,
            },
            Exception(ref x) => match other {
                Exception(ref y) => x == y,
                _ => false,
            },
        }
    }
}

impl Eq for Value {}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// NOTE: `Ord` is implemented to facilitate operations within the `Interpreter`,
// e.g. consistent hashing; this notion of order should not be exposed to users.
impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        use Value::*;

        match self {
            Nil => match other {
                Nil => Ordering::Equal,
                _ => Ordering::Less,
            },
            Bool(ref x) => match other {
                Nil => Ordering::Greater,
                Bool(ref y) => x.cmp(y),
                _ => Ordering::Less,
            },
            Number(ref x) => match other {
                Nil | Bool(_) => Ordering::Greater,
                Number(ref y) => x.cmp(y),
                _ => Ordering::Less,
            },
            String(ref x) => match other {
                Nil | Bool(_) | Number(_) => Ordering::Greater,
                String(ref y) => x.cmp(y),
                _ => Ordering::Less,
            },
            Keyword(ref x, ref x_ns_opt) => match other {
                Nil | Bool(_) | Number(_) | String(_) => Ordering::Greater,
                Keyword(ref y, ref y_ns_opt) => ((x, x_ns_opt)).cmp(&(y, y_ns_opt)),
                _ => Ordering::Less,
            },
            Symbol(ref x, ref x_ns_opt) => match other {
                Nil | Bool(_) | Number(_) | String(_) | Keyword(_, _) => Ordering::Greater,
                Symbol(ref y, ref y_ns_opt) => ((x, x_ns_opt)).cmp(&(y, y_ns_opt)),
                _ => Ordering::Less,
            },
            List(ref x) => match other {
                Nil | Bool(_) | Number(_) | String(_) | Keyword(_, _) | Symbol(_, _) => {
                    Ordering::Greater
                }
                List(ref y) => x.cmp(y),
                _ => Ordering::Less,
            },
            Vector(ref x) => match other {
                Nil | Bool(_) | Number(_) | String(_) | Keyword(_, _) | Symbol(_, _) | List(_) => {
                    Ordering::Greater
                }
                Vector(ref y) => x.cmp(y),
                _ => Ordering::Less,
            },
            Map(ref x) => match other {
                Nil
                | Bool(_)
                | Number(_)
                | String(_)
                | Keyword(_, _)
                | Symbol(_, _)
                | List(_)
                | Vector(_) => Ordering::Greater,
                Map(ref y) => sorted(x).cmp(sorted(y)),
                _ => Ordering::Less,
            },
            Set(ref x) => match other {
                Nil
                | Bool(_)
                | Number(_)
                | String(_)
                | Keyword(_, _)
                | Symbol(_, _)
                | List(_)
                | Vector(_)
                | Map(_) => Ordering::Greater,
                Set(ref y) => sorted(x).cmp(sorted(y)),
                _ => Ordering::Less,
            },
            Fn(ref x) => match other {
                Nil
                | Bool(_)
                | Number(_)
                | String(_)
                | Keyword(_, _)
                | Symbol(_, _)
                | List(_)
                | Vector(_)
                | Map(_)
                | Set(_) => Ordering::Greater,
                Fn(ref y) => x.cmp(y),
                _ => Ordering::Less,
            },
            Primitive(x) => match other {
                Nil
                | Bool(_)
                | Number(_)
                | String(_)
                | Keyword(_, _)
                | Symbol(_, _)
                | List(_)
                | Vector(_)
                | Map(_)
                | Set(_)
                | Fn(_) => Ordering::Greater,
                Primitive(y) => {
                    let x_ptr = x as *const NativeFn;
                    let x_identifier =
                        unsafe { std::mem::transmute::<*const NativeFn, usize>(x_ptr) };
                    let y_ptr = y as *const NativeFn;
                    let y_identifier =
                        unsafe { std::mem::transmute::<*const NativeFn, usize>(y_ptr) };
                    x_identifier.cmp(&y_identifier)
                }
                _ => Ordering::Less,
            },
            Var(VarImpl {
                namespace: namespace_x,
                identifier: identifier_x,
                ..
            }) => match other {
                Nil
                | Bool(_)
                | Number(_)
                | String(_)
                | Keyword(_, _)
                | Symbol(_, _)
                | List(_)
                | Vector(_)
                | Map(_)
                | Set(_)
                | Fn(_)
                | Primitive(_) => Ordering::Greater,
                Var(VarImpl {
                    namespace: namespace_y,
                    identifier: identifier_y,
                    ..
                }) => (namespace_x, identifier_x).cmp(&(namespace_y, identifier_y)),
                _ => Ordering::Less,
            },
            Recur(ref x) => match other {
                Nil
                | Bool(_)
                | Number(_)
                | String(_)
                | Keyword(_, _)
                | Symbol(_, _)
                | List(_)
                | Vector(_)
                | Map(_)
                | Set(_)
                | Fn(_)
                | Primitive(_)
                | Var(_) => Ordering::Greater,
                Recur(ref y) => x.cmp(y),
                _ => Ordering::Less,
            },
            Atom(ref x) => match other {
                Nil
                | Bool(_)
                | Number(_)
                | String(_)
                | Keyword(_, _)
                | Symbol(_, _)
                | List(_)
                | Vector(_)
                | Map(_)
                | Set(_)
                | Fn(_)
                | Primitive(_)
                | Var(_)
                | Recur(_) => Ordering::Greater,
                Atom(ref y) => x.cmp(y),
                _ => Ordering::Less,
            },
            Macro(ref x) => match other {
                Nil
                | Bool(_)
                | Number(_)
                | String(_)
                | Keyword(_, _)
                | Symbol(_, _)
                | List(_)
                | Vector(_)
                | Map(_)
                | Set(_)
                | Fn(_)
                | Primitive(_)
                | Var(_)
                | Recur(_)
                | Atom(_) => Ordering::Greater,
                Macro(ref y) => x.cmp(y),
                _ => Ordering::Less,
            },
            Exception(ref x) => match other {
                Nil
                | Bool(_)
                | Number(_)
                | String(_)
                | Keyword(_, _)
                | Symbol(_, _)
                | List(_)
                | Vector(_)
                | Map(_)
                | Set(_)
                | Fn(_)
                | Primitive(_)
                | Var(_)
                | Recur(_)
                | Atom(_)
                | Macro(_) => Ordering::Greater,
                Exception(ref y) => x.cmp(y),
            },
        }
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use Value::*;

        // mix in the particular variant
        discriminant(self).hash(state);

        match self {
            Nil => {}
            Bool(b) => b.hash(state),
            Number(n) => n.hash(state),
            String(s) => s.hash(state),
            Keyword(s, ns) => {
                s.hash(state);
                ns.hash(state);
            }
            Symbol(s, ns) => {
                s.hash(state);
                ns.hash(state);
            }
            List(l) => l.hash(state),
            Vector(v) => v.hash(state),
            Map(m) => {
                m.size().hash(state);
                sorted(m).for_each(|binding| binding.hash(state));
            }
            Set(s) => {
                s.size().hash(state);
                sorted(s).for_each(|elem| elem.hash(state));
            }
            Fn(lambda) => lambda.hash(state),
            Primitive(f) => {
                let ptr = f as *const NativeFn;
                let identifier = unsafe { std::mem::transmute::<*const NativeFn, usize>(ptr) };
                identifier.hash(state);
            }
            Var(VarImpl {
                inner,
                namespace,
                identifier,
            }) => {
                (*inner.borrow()).hash(state);
                namespace.hash(state);
                identifier.hash(state);
            }
            Recur(v) => v.hash(state),
            Atom(v) => {
                (*v.borrow()).hash(state);
            }
            Macro(lambda) => lambda.hash(state),
            Exception(e) => e.hash(state),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Value::*;

        match self {
            Nil => write!(f, "nil"),
            Bool(ref b) => write!(f, "{}", b),
            Number(ref n) => write!(f, "{}", n),
            String(ref s) => write!(f, "\"{}\"", s),
            Keyword(ref id, ref ns_opt) => {
                write!(f, ":")?;
                if let Some(ns) = ns_opt {
                    write!(f, "{}/", ns)?;
                }
                write!(f, "{}", id)
            }
            Symbol(ref id, ref ns_opt) => {
                if let Some(ns) = ns_opt {
                    write!(f, "{}/", ns)?;
                }
                write!(f, "{}", id)
            }
            List(elems) => write!(f, "({})", join(elems, " ")),
            Vector(elems) => write!(f, "[{}]", join(elems, " ")),
            Map(elems) => {
                let mut inner = vec![];
                for (k, v) in elems {
                    let mut buffer = std::string::String::new();
                    write!(buffer, "{} {}", k, v)?;
                    inner.push(buffer);
                }
                write!(f, "{{{}}}", join(inner, ", "))
            }
            Set(elems) => write!(f, "#{{{}}}", join(elems, " ")),
            Fn(_) => write!(f, "<fn*>"),
            Primitive(_) => write!(f, "<native function>"),
            Var(VarImpl {
                namespace,
                identifier,
                ..
            }) => write!(f, "#'{}/{}", namespace, identifier),
            Recur(elems) => write!(f, "[{}]", join(elems, " ")),
            Atom(v) => write!(f, "(atom {})", *v.borrow()),
            Macro(_) => write!(f, "<macro>"),
            Exception(ExceptionImpl { message, data }) => {
                write!(f, "exception: {}, {}", message, data)
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Value {
    pub fn to_readable_string(&self) -> String {
        use Value::*;

        let mut f = std::string::String::new();

        let _ = match self {
            Nil => write!(&mut f, "nil"),
            Bool(ref b) => write!(&mut f, "{}", b),
            Number(ref n) => write!(&mut f, "{}", n),
            String(ref s) => write!(&mut f, "{}", s),
            Keyword(ref id, ref ns_opt) => {
                let _ = write!(&mut f, ":");
                if let Some(ns) = ns_opt {
                    let _ = write!(&mut f, "{}/", ns);
                }
                write!(&mut f, "{}", id)
            }
            Symbol(ref id, ref ns_opt) => {
                if let Some(ns) = ns_opt {
                    let _ = write!(&mut f, "{}/", ns);
                }
                write!(&mut f, "{}", id)
            }
            List(elems) => write!(&mut f, "({})", join(elems, " ")),
            Vector(elems) => write!(&mut f, "[{}]", join(elems, " ")),
            Map(elems) => {
                let mut inner = vec![];
                for (k, v) in elems {
                    let mut buffer = std::string::String::new();
                    let _ = write!(buffer, "{} {}", k, v);
                    inner.push(buffer);
                }
                write!(&mut f, "{{{}}}", join(inner, ", "))
            }
            Set(elems) => write!(&mut f, "#{{{}}}", join(elems, " ")),
            Fn(_) => write!(&mut f, "<fn*>"),
            Primitive(_) => write!(&mut f, "<native function>"),
            Var(VarImpl {
                namespace,
                identifier,
                ..
            }) => write!(f, "#'{}/{}", namespace, identifier),
            Recur(elems) => write!(&mut f, "[{}]", join(elems, " ")),
            Atom(v) => write!(&mut f, "(atom {})", *v.borrow()),
            Macro(_) => write!(&mut f, "<macro>"),
            Exception(_) => write!(&mut f, "<exception>"),
        };

        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Value::*;

    #[test]
    fn test_ord_provided() {
        let ref x = List(PersistentList::from_iter(vec![
            Number(1),
            Number(2),
            Number(3),
        ]));
        let ref y = List(PersistentList::from_iter(vec![
            Number(2),
            Number(3),
            Number(1),
        ]));
        let ref z = List(PersistentList::from_iter(vec![Number(44)]));
        let ref a = List(PersistentList::from_iter(vec![Number(0)]));
        let ref b = List(PersistentList::from_iter(vec![Number(1)]));
        let ref c = List(PersistentList::new());

        assert_eq!(x.cmp(x), Ordering::Equal);
        assert_eq!(x.cmp(y), Ordering::Less);
        assert_eq!(x.cmp(z), Ordering::Less);
        assert_eq!(x.cmp(a), Ordering::Greater);
        assert_eq!(x.cmp(b), Ordering::Greater);
        assert_eq!(x.cmp(c), Ordering::Greater);
        assert_eq!(c.cmp(x), Ordering::Less);
        assert_eq!(c.cmp(y), Ordering::Less);
    }

    #[test]
    fn test_ord_custom() {
        let ref x = Map(PersistentMap::from_iter(vec![
            (Number(1), Number(2)),
            (Number(3), Number(4)),
        ]));
        let ref y = Map(PersistentMap::from_iter(vec![(Number(1), Number(2))]));
        let ref z = Map(PersistentMap::from_iter(vec![
            (Number(4), Number(3)),
            (Number(1), Number(2)),
        ]));
        let ref a = Map(PersistentMap::from_iter(vec![
            (Number(1), Number(444)),
            (Number(3), Number(4)),
        ]));
        let ref b = Map(PersistentMap::new());
        let ref c = Map(PersistentMap::from_iter(vec![
            (Number(1), Number(2)),
            (Number(3), Number(4)),
            (Number(4), Number(8)),
        ]));

        assert_eq!(x.cmp(x), Ordering::Equal);
        assert_eq!(x.cmp(y), Ordering::Greater);
        assert_eq!(x.cmp(z), Ordering::Less);
        assert_eq!(x.cmp(a), Ordering::Less);
        assert_eq!(x.cmp(b), Ordering::Greater);
        assert_eq!(x.cmp(c), Ordering::Less);
        assert_eq!(b.cmp(b), Ordering::Equal);
        assert_eq!(b.cmp(c), Ordering::Less);
        assert_eq!(b.cmp(y), Ordering::Less);
    }
}
