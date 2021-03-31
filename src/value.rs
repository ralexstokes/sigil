use crate::namespace::Namespace;
use itertools::{join, sorted};
use rpds::{
    HashTrieMap as PersistentMap, HashTrieSet as PersistentSet, List as PersistentList,
    Vector as PersistentVector,
};
use std::fmt;
use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::mem::discriminant;

use std::cmp::{Ord, Ordering, PartialEq};
#[derive(Debug, PartialEq, Eq)]
pub enum Value {
    Nil,
    Bool(bool),
    Number(u64),
    String(String),
    Keyword(String, Option<Namespace>),
    Symbol(String, Option<Namespace>),
    List(PersistentList<Value>),
    Vector(PersistentVector<Value>),
    Map(PersistentMap<Value, Value>),
    Set(PersistentSet<Value>),
}

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
                Set(ref y) => sorted(x).cmp(sorted(y)),
                _ => Ordering::Greater,
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
        }
    }
}

impl fmt::Display for Value {
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
        }
    }
}
