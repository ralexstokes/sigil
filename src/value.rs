use crate::collections::{PersistentList, PersistentMap, PersistentSet, PersistentVector};
use crate::interpreter::{EvaluationError, EvaluationResult, Interpreter};
use crate::namespace::Var;
use crate::reader::{Atom, Form, Identifier, Symbol};
use crate::writer::{
    unescape_string, write_bool, write_fn, write_identifer, write_keyword, write_list, write_map,
    write_nil, write_number, write_primitive, write_set, write_string, write_symbol, write_var,
    write_vector,
};
use itertools::{sorted, Itertools};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};
use std::mem::discriminant;

// pub fn list_with_values(values: impl IntoIterator<Item = RuntimeValue>) -> RuntimeValue {
//     RuntimeValue::List(PersistentList::from_iter(values))
// }

// pub fn vector_with_values(values: impl IntoIterator<Item = RuntimeValue>) -> RuntimeValue {
//     RuntimeValue::Vector(PersistentVector::from_iter(values))
// }

// pub fn map_with_values(
//     values: impl IntoIterator<Item = (RuntimeValue, RuntimeValue)>,
// ) -> RuntimeValue {
//     RuntimeValue::Map(PersistentMap::from_iter(values))
// }

// pub fn set_with_values(values: impl IntoIterator<Item = RuntimeValue>) -> RuntimeValue {
//     RuntimeValue::Set(PersistentSet::from_iter(values))
// }

// pub fn var_with_value(value: RuntimeValue, namespace: &str, identifier: &str) -> RuntimeValue {
//     RuntimeValue::Var(VarImpl {
//         data: Rc::new(RefCell::new(Some(value))),
//         namespace: namespace.to_string(),
//         identifier: identifier.to_string(),
//     })
// }

// pub fn unbound_var(namespace: &str, identifier: &str) -> RuntimeValue {
//     RuntimeValue::Var(VarImpl {
//         data: Rc::new(RefCell::new(None)),
//         namespace: namespace.to_string(),
//         identifier: identifier.to_string(),
//     })
// }

// pub fn atom_with_value(value: RuntimeValue) -> RuntimeValue {
//     RuntimeValue::Atom(Rc::new(RefCell::new(value)))
// }

// pub fn var_impl_into_inner(var: &VarImpl) -> Option<RuntimeValue> {
//     var.data.borrow().clone()
// }

// pub fn atom_impl_into_inner(atom: &AtomImpl) -> RuntimeValue {
//     atom.borrow().clone()
// }

pub fn exception(msg: &str, data: RuntimeValue) -> ExceptionImpl {
    ExceptionImpl::User(UserException {
        message: msg.to_string(),
        data: Box::new(data),
    })
}

pub fn exception_from_system_err(err: EvaluationError) -> RuntimeValue {
    let inner = match err {
        EvaluationError::Exception(exc) => exc,
        err => ExceptionImpl::System(Box::new(err)),
    };
    RuntimeValue::Exception(inner)
}

pub type NativeFn = fn(&mut Interpreter, &[RuntimeValue]) -> EvaluationResult<RuntimeValue>;

#[derive(Clone)]
pub struct Primitive(NativeFn);

impl From<&NativeFn> for Primitive {
    fn from(f: &NativeFn) -> Self {
        Self(*f)
    }
}

impl fmt::Debug for Primitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<primitive fn>")
    }
}

impl PartialEq for Primitive {
    fn eq(&self, other: &Self) -> bool {
        let x = self as *const Primitive as usize;
        let y = other as *const Primitive as usize;
        x == y
    }
}

impl Eq for Primitive {}

impl Hash for Primitive {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let x = self as *const Self as usize;
        x.hash(state);
    }
}

impl PartialOrd for Primitive {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let x = self as *const Primitive as usize;
        let y = other as *const Primitive as usize;
        x.partial_cmp(&y)
    }
}

impl Ord for Primitive {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Primitive {
    pub fn apply(
        &self,
        interpreter: &mut Interpreter,
        args: &[RuntimeValue],
    ) -> EvaluationResult<RuntimeValue> {
        self.0(interpreter, args)
    }
}

// #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
// pub struct FnImpl {
//     pub body: PersistentList<RuntimeValue>,
//     pub arity: usize,
//     // allow for nested fns
//     pub level: usize,
//     pub variadic: bool,
// }

// #[derive(Debug, Clone, Eq)]
// pub struct FnWithCapturesImpl {
//     pub f: FnImpl,
//     pub captures: HashMap<String, Option<RuntimeValue>>,
// }

// impl PartialOrd for FnWithCapturesImpl {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         Some(self.cmp(other))
//     }
// }

// impl Ord for FnWithCapturesImpl {
//     fn cmp(&self, other: &Self) -> Ordering {
//         match self.f.cmp(&other.f) {
//             Ordering::Equal => {
//                 let sorted_pairs = self.captures.iter().sorted();
//                 let other_sorted_pairs = other.captures.iter().sorted();
//                 sorted_pairs.cmp(other_sorted_pairs)
//             }
//             other => other,
//         }
//     }
// }

// impl PartialEq for FnWithCapturesImpl {
//     fn eq(&self, other: &Self) -> bool {
//         if self.f != other.f {
//             return false;
//         }

//         self.captures
//             .iter()
//             .sorted()
//             .zip(other.captures.iter().sorted())
//             .all(|((a, b), (c, d))| a == c && b == d)
//     }
// }

// impl Hash for FnWithCapturesImpl {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.f.hash(state);
//         self.captures.iter().sorted().for_each(|(k, v)| {
//             k.hash(state);
//             v.hash(state);
//         });
//     }
// }

// #[derive(Clone)]
// pub struct VarImpl {
//     data: Rc<RefCell<Option<RuntimeValue>>>,
//     namespace: String,
//     pub identifier: String,
// }

// impl VarImpl {
//     pub fn update(&self, value: RuntimeValue) {
//         *self.data.borrow_mut() = Some(value);
//     }
// }

// type AtomImpl = Rc<RefCell<RuntimeValue>>;

#[derive(Clone, Debug)]
pub struct UserException {
    message: String,
    data: Box<RuntimeValue>,
}

impl UserException {
    fn to_readable_string(&self) -> String {
        let mut result = String::new();
        if !self.message.is_empty() {
            write!(&mut result, "{}, ", self.message).expect("can write to string")
        }
        write!(&mut result, "{}", self.data.to_readable_string()).expect("can write to string");
        result
    }
}

impl fmt::Display for UserException {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.message.is_empty() {
            write!(f, "{}, ", self.message)?;
        }
        write!(f, "{}", self.data)
    }
}

#[derive(Clone, Debug)]
pub enum ExceptionImpl {
    User(UserException),
    System(Box<EvaluationError>),
}

impl ExceptionImpl {
    fn to_readable_string(&self) -> String {
        let mut result = String::new();
        match self {
            ExceptionImpl::User(exc) => {
                write!(&mut result, "{}", exc.to_readable_string()).expect("can write to string")
            }
            ExceptionImpl::System(err) => write!(
                &mut result,
                "{}",
                RuntimeValue::String(err.to_string()).to_readable_string()
            )
            .expect("can write to string"),
        }
        result
    }
}

impl PartialEq for ExceptionImpl {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                ExceptionImpl::User(UserException { message, data }),
                ExceptionImpl::User(UserException {
                    message: other_message,
                    data: other_data,
                }),
            ) => message == other_message && data == other_data,
            _ => false,
        }
    }
}

impl Eq for ExceptionImpl {}

impl PartialOrd for ExceptionImpl {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExceptionImpl {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (
                ExceptionImpl::User(UserException { message, data }),
                ExceptionImpl::User(UserException {
                    message: other_message,
                    data: other_data,
                }),
            ) => (message, data).cmp(&(other_message, other_data)),
            (ExceptionImpl::User(..), ExceptionImpl::System(..)) => Ordering::Less,
            (ExceptionImpl::System(..), ExceptionImpl::User(..)) => Ordering::Greater,
            (ExceptionImpl::System(a), ExceptionImpl::System(b)) => {
                a.to_string().cmp(&b.to_string())
            }
        }
    }
}

impl Hash for ExceptionImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        discriminant(self).hash(state);
        match self {
            ExceptionImpl::User(UserException { message, data }) => {
                message.hash(state);
                data.hash(state);
            }
            ExceptionImpl::System(err) => {
                err.to_string().hash(state);
            }
        }
    }
}

impl fmt::Display for ExceptionImpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExceptionImpl::User(UserException { message, data }) => {
                if !message.is_empty() {
                    write!(f, "{}, ", message)?;
                }
                write!(f, "{}", data)
            }
            ExceptionImpl::System(err) => {
                write!(f, "{}", err)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpecialForm {
    // (def! symbol form?)
    Def(DefForm),
    // (var symbol)
    Var(Symbol),
    // (let* [bindings*] form*)
    Let(LetForm),
    // (loop* [bindings*] form*)
    Loop(LexicalForm),
    // (recur form*)
    Recur(BodyForm),
    // (if predicate consequent alternate?)
    If(IfForm),
    // (do form*)
    Do(BodyForm),
    //(fn* [parameters*] form*)
    Fn(FnForm),
    // (quote form)
    Quote(Box<RuntimeValue>),
    // (quasiquote form)
    Quasiquote(Box<RuntimeValue>),
    // // (unquote form)
    // Unquote(Box<RuntimeValue>),
    // // (splice-unquote form)
    // SpliceUnquote(Box<RuntimeValue>),
    // (defmacro! symbol fn*-form)
    Defmacro(Symbol, FnForm),
    // (macroexpand macro-form)
    Macroexpand(Box<RuntimeValue>),
    // (try* form* catch*-form?)
    Try(TryForm),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DefForm {
    Bound(Symbol, Box<RuntimeValue>),
    Unbound(Symbol),
}

pub type LexicalBinding = (Identifier, Box<RuntimeValue>);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LexicalBindings {
    Bound(Vec<LexicalBinding>),
    Unbound(Vec<Identifier>),
}

impl LexicalBindings {
    pub fn names(&self) -> Vec<Identifier> {
        match self {
            LexicalBindings::Bound(bindings) => {
                bindings.iter().map(|(name, _)| name.clone()).collect()
            }
            LexicalBindings::Unbound(bindings) => bindings.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LexicalForm {
    pub bindings: LexicalBindings,
    pub body: BodyForm,
}

fn is_forward_visible(form: &RuntimeValue) -> bool {
    matches!(form, RuntimeValue::SpecialForm(SpecialForm::Fn(_)))
}

impl LexicalForm {
    pub(super) fn resolve_forward_declarations(&self) -> HashSet<usize> {
        let mut result = HashSet::new();
        match &self.bindings {
            LexicalBindings::Bound(bindings) => {
                for (index, (name, value)) in bindings.iter().enumerate() {
                    if is_forward_visible(value.as_ref()) {
                        result.insert(index);
                    }
                }
            }
            LexicalBindings::Unbound(_) => unreachable!("only relevant for bound symbols"),
        }
        result
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LetForm {
    pub lexical_form: LexicalForm,
    // `let*` can produce "forward declarations" where some names
    // in `scope` can be seen by all other names
    pub forward_declarations: HashSet<usize>,
}

impl Hash for LetForm {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.lexical_form.hash(state);
        let declarations = self.forward_declarations.iter().sorted();
        for declaration in declarations {
            declaration.hash(state);
        }
    }
}

impl PartialOrd for LetForm {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.lexical_form.cmp(&other.lexical_form) {
            Ordering::Equal => Some(
                sorted(self.forward_declarations.iter())
                    .cmp(sorted(other.forward_declarations.iter())),
            ),
            other => Some(other),
        }
    }
}

impl Ord for LetForm {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl LetForm {
    pub fn identifier_for_binding(&self, index: usize) -> Option<&Identifier> {
        match &self.lexical_form.bindings {
            LexicalBindings::Bound(bindings) => bindings.get(index).map(|binding| &binding.0),
            LexicalBindings::Unbound(bindings) => bindings.get(index),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BodyForm {
    pub body: Vec<RuntimeValue>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IfForm {
    pub predicate: Box<RuntimeValue>,
    pub consequent: Box<RuntimeValue>,
    pub alternate: Option<Box<RuntimeValue>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FnForm {
    pub parameters: Vec<Identifier>,
    pub variadic: Option<Identifier>,
    pub body: BodyForm,
}

impl FnForm {
    // `arity` is the number of _fixed_ arguments `self` expects
    pub fn arity(&self) -> usize {
        self.parameters.len()
    }

    pub fn variadic(&self) -> bool {
        self.variadic.is_some()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TryForm {
    pub body: BodyForm,
    pub catch: Option<CatchForm>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CatchForm {
    pub exception_binding: Identifier,
    pub body: BodyForm,
}

impl From<&Atom> for RuntimeValue {
    fn from(atom: &Atom) -> Self {
        match atom {
            Atom::Nil => RuntimeValue::Nil,
            Atom::Bool(b) => RuntimeValue::Bool(*b),
            Atom::Number(n) => RuntimeValue::Number(*n),
            Atom::String(s) => RuntimeValue::String(s.clone()),
            Atom::Keyword(k) => RuntimeValue::Keyword(k.clone()),
            Atom::Symbol(s) => RuntimeValue::Symbol(s.clone()),
        }
    }
}

// this `From` impl provides for a translation from a `Form` to an `RuntimeValue`
// without doing any actual analysis, e.g. when producing a `quote` form.
impl From<&Form> for RuntimeValue {
    fn from(form: &Form) -> Self {
        match form {
            Form::Atom(atom) => atom.into(),
            Form::List(coll) => RuntimeValue::List(coll.iter().map(From::from).collect()),
            Form::Vector(coll) => RuntimeValue::Vector(coll.iter().map(From::from).collect()),
            Form::Map(coll) => {
                RuntimeValue::Map(coll.iter().map(|(k, v)| (k.into(), v.into())).collect())
            }
            Form::Set(coll) => RuntimeValue::Set(coll.iter().map(From::from).collect()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeValue {
    Nil,
    Bool(bool),
    Number(i64),
    String(String),
    Keyword(Symbol),
    Symbol(Symbol),
    LexicalSymbol(Identifier),
    Var(Var),
    List(PersistentList<RuntimeValue>),
    Vector(PersistentVector<RuntimeValue>),
    Map(PersistentMap<RuntimeValue, RuntimeValue>),
    Set(PersistentSet<RuntimeValue>),
    SpecialForm(SpecialForm),
    Fn(FnForm),
    Primitive(Primitive),
    Exception(ExceptionImpl),
    // FnWithCaptures(FnWithCapturesImpl),
    // Atom(AtomImpl),
    // Macro(FnImpl),
}

// #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
// pub struct FnImpl {
//     pub parameters: Vec<Identifier>,
//     pub variadic: Option<Identifier>,
//     pub body: Vec<RuntimeValue>,
// }

// impl PartialEq for RuntimeValue {
//     fn eq(&self, other: &Self) -> bool {
//         use RuntimeValue::*;

//         match self {
//             Nil => matches!(other, Nil),
//             Bool(ref x) => match other {
//                 Bool(ref y) => x == y,
//                 _ => false,
//             },
//             Number(ref x) => match other {
//                 Number(ref y) => x == y,
//                 _ => false,
//             },
//             String(ref x) => match other {
//                 String(ref y) => x == y,
//                 _ => false,
//             },
//             Keyword(x) => match other {
//                 Keyword(y) => x == y,
//                 _ => false,
//             },
//             Symbol(x) => match other {
//                 Symbol(y) => x == y,
//                 _ => false,
//             },
//             List(ref x) => match other {
//                 List(ref y) => x == y,
//                 Vector(ref y) => {
//                     if x.len() == y.len() {
//                         x.iter().zip(y.iter()).map(|(a, b)| a == b).all(|x| x)
//                     } else {
//                         false
//                     }
//                 }
//                 _ => false,
//             },
//             Vector(ref x) => match other {
//                 Vector(ref y) => x == y,
//                 List(ref y) => {
//                     if x.len() == y.len() {
//                         x.iter().zip(y.iter()).map(|(a, b)| a == b).all(|x| x)
//                     } else {
//                         false
//                     }
//                 }
//                 _ => false,
//             },
//             Map(ref x) => match other {
//                 Map(ref y) => x == y,
//                 _ => false,
//             },
//             Set(ref x) => match other {
//                 Set(ref y) => x == y,
//                 _ => false,
//             },
//             Fn(x) => match other {
//                 Fn(y) => x == y,
//                 _ => false,
//             },
//             // FnWithCaptures(ref x) => match other {
//             //     FnWithCaptures(ref y) => x == y,
//             //     _ => false,
//             // },
//             Primitive(x) => match other {
//                 Primitive(y) => {
//                     let x_ptr = x as *const Primitive;
//                     let x_identifier = x_ptr as usize;
//                     let y_ptr = y as *const Primitive;
//                     let y_identifier = y_ptr as usize;
//                     x_identifier == y_identifier
//                 }
//                 _ => false,
//             },
//             Var(VarImpl {
//                 namespace: namespace_x,
//                 identifier: identifier_x,
//                 ..
//             }) => match other {
//                 Var(VarImpl {
//                     namespace: namespace_y,
//                     identifier: identifier_y,
//                     ..
//                 }) => (namespace_x, identifier_x) == (namespace_y, identifier_y),
//                 _ => false,
//             },
//             // Recur(ref x) => match other {
//             //     Recur(ref y) => x == y,
//             //     _ => false,
//             // },
//             // Atom(ref x) => match other {
//             //     Atom(ref y) => x == y,
//             //     _ => false,
//             // },
//             // Macro(ref x) => match other {
//             //     Macro(ref y) => x == y,
//             //     _ => false,
//             // },
//             Exception(ref x) => match other {
//                 Exception(ref y) => x == y,
//                 _ => false,
//             },
//         }
//     }
// }

// impl PartialOrd for RuntimeValue {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         Some(self.cmp(other))
//     }
// }

// NOTE: `Ord` is implemented to facilitate operations within the `Interpreter`,
// e.g. consistent hashing; this notion of order should not be exposed to users.
// impl Ord for RuntimeValue {
//     fn cmp(&self, other: &Self) -> Ordering {
//         use RuntimeValue::*;

//         match self {
//             Nil => match other {
//                 Nil => Ordering::Equal,
//                 _ => Ordering::Less,
//             },
//             Bool(ref x) => match other {
//                 Nil => Ordering::Greater,
//                 Bool(ref y) => x.cmp(y),
//                 _ => Ordering::Less,
//             },
//             Number(ref x) => match other {
//                 Nil | Bool(_) => Ordering::Greater,
//                 Number(ref y) => x.cmp(y),
//                 _ => Ordering::Less,
//             },
//             String(ref x) => match other {
//                 Nil | Bool(_) | Number(_) => Ordering::Greater,
//                 String(ref y) => x.cmp(y),
//                 _ => Ordering::Less,
//             },
//             Keyword(ref x, ref x_ns_opt) => match other {
//                 Nil | Bool(_) | Number(_) | String(_) => Ordering::Greater,
//                 Keyword(ref y, ref y_ns_opt) => (x, x_ns_opt).cmp(&(y, y_ns_opt)),
//                 _ => Ordering::Less,
//             },
//             Symbol(ref x, ref x_ns_opt) => match other {
//                 Nil | Bool(_) | Number(_) | String(_) | Keyword(_, _) => Ordering::Greater,
//                 Symbol(ref y, ref y_ns_opt) => (x, x_ns_opt).cmp(&(y, y_ns_opt)),
//                 _ => Ordering::Less,
//             },
//             List(ref x) => match other {
//                 Nil | Bool(_) | Number(_) | String(_) | Keyword(_, _) | Symbol(_, _) => {
//                     Ordering::Greater
//                 }
//                 List(ref y) => x.cmp(y),
//                 _ => Ordering::Less,
//             },
//             Vector(ref x) => match other {
//                 Nil | Bool(_) | Number(_) | String(_) | Keyword(_, _) | Symbol(_, _) | List(_) => {
//                     Ordering::Greater
//                 }
//                 Vector(ref y) => x.cmp(y),
//                 _ => Ordering::Less,
//             },
//             Map(ref x) => match other {
//                 Nil
//                 | Bool(_)
//                 | Number(_)
//                 | String(_)
//                 | Keyword(_, _)
//                 | Symbol(_, _)
//                 | List(_)
//                 | Vector(_) => Ordering::Greater,
//                 Map(ref y) => sorted(x).cmp(sorted(y)),
//                 _ => Ordering::Less,
//             },
//             Set(ref x) => match other {
//                 Nil
//                 | Bool(_)
//                 | Number(_)
//                 | String(_)
//                 | Keyword(_, _)
//                 | Symbol(_, _)
//                 | List(_)
//                 | Vector(_)
//                 | Map(_) => Ordering::Greater,
//                 Set(ref y) => sorted(x).cmp(sorted(y)),
//                 _ => Ordering::Less,
//             },
//             Fn(ref x) => match other {
//                 Nil
//                 | Bool(_)
//                 | Number(_)
//                 | String(_)
//                 | Keyword(_, _)
//                 | Symbol(_, _)
//                 | List(_)
//                 | Vector(_)
//                 | Map(_)
//                 | Set(_) => Ordering::Greater,
//                 Fn(ref y) => x.cmp(y),
//                 _ => Ordering::Less,
//             },
//             // FnWithCaptures(ref x) => match other {
//             //     Nil
//             //     | Bool(_)
//             //     | Number(_)
//             //     | String(_)
//             //     | Keyword(_, _)
//             //     | Symbol(_, _)
//             //     | List(_)
//             //     | Vector(_)
//             //     | Map(_)
//             //     | Set(_)
//             //     | Fn(_) => Ordering::Greater,
//             //     FnWithCaptures(ref y) => x.cmp(y),
//             //     _ => Ordering::Less,
//             // },
//             Primitive(x) => match other {
//                 Nil
//                 | Bool(_)
//                 | Number(_)
//                 | String(_)
//                 | Keyword(_, _)
//                 | Symbol(_, _)
//                 | List(_)
//                 | Vector(_)
//                 | Map(_)
//                 | Set(_)
//                 | Fn(_)
//                 | FnWithCaptures(_) => Ordering::Greater,
//                 Primitive(y) => {
//                     let x_ptr = x as *const NativeFn;
//                     let x_identifier = x_ptr as usize;
//                     let y_ptr = y as *const NativeFn;
//                     let y_identifier = y_ptr as usize;
//                     x_identifier.cmp(&y_identifier)
//                 }
//                 _ => Ordering::Less,
//             },
//             Var(VarImpl {
//                 namespace: namespace_x,
//                 identifier: identifier_x,
//                 ..
//             }) => match other {
//                 Nil
//                 | Bool(_)
//                 | Number(_)
//                 | String(_)
//                 | Keyword(_, _)
//                 | Symbol(_, _)
//                 | List(_)
//                 | Vector(_)
//                 | Map(_)
//                 | Set(_)
//                 | Fn(_)
//                 | FnWithCaptures(_)
//                 | Primitive(_) => Ordering::Greater,
//                 Var(VarImpl {
//                     namespace: namespace_y,
//                     identifier: identifier_y,
//                     ..
//                 }) => (namespace_x, identifier_x).cmp(&(namespace_y, identifier_y)),
//                 _ => Ordering::Less,
//             },
//             Recur(ref x) => match other {
//                 Nil
//                 | Bool(_)
//                 | Number(_)
//                 | String(_)
//                 | Keyword(_, _)
//                 | Symbol(_, _)
//                 | List(_)
//                 | Vector(_)
//                 | Map(_)
//                 | Set(_)
//                 | Fn(_)
//                 | FnWithCaptures(_)
//                 | Primitive(_)
//                 | Var(_) => Ordering::Greater,
//                 Recur(ref y) => x.cmp(y),
//                 _ => Ordering::Less,
//             },
//             Atom(ref x) => match other {
//                 Nil
//                 | Bool(_)
//                 | Number(_)
//                 | String(_)
//                 | Keyword(_, _)
//                 | Symbol(_, _)
//                 | List(_)
//                 | Vector(_)
//                 | Map(_)
//                 | Set(_)
//                 | Fn(_)
//                 | FnWithCaptures(_)
//                 | Primitive(_)
//                 | Var(_)
//                 | Recur(_) => Ordering::Greater,
//                 Atom(ref y) => x.cmp(y),
//                 _ => Ordering::Less,
//             },
//             Macro(ref x) => match other {
//                 Nil
//                 | Bool(_)
//                 | Number(_)
//                 | String(_)
//                 | Keyword(_, _)
//                 | Symbol(_, _)
//                 | List(_)
//                 | Vector(_)
//                 | Map(_)
//                 | Set(_)
//                 | Fn(_)
//                 | FnWithCaptures(_)
//                 | Primitive(_)
//                 | Var(_)
//                 | Recur(_)
//                 | Atom(_) => Ordering::Greater,
//                 Macro(ref y) => x.cmp(y),
//                 _ => Ordering::Less,
//             },
//             Exception(ref x) => match other {
//                 Nil
//                 | Bool(_)
//                 | Number(_)
//                 | String(_)
//                 | Keyword(_, _)
//                 | Symbol(_, _)
//                 | List(_)
//                 | Vector(_)
//                 | Map(_)
//                 | Set(_)
//                 | Fn(_)
//                 | FnWithCaptures(_)
//                 | Primitive(_)
//                 | Var(_)
//                 | Recur(_)
//                 | Atom(_)
//                 | Macro(_) => Ordering::Greater,
//                 Exception(ref y) => x.cmp(y),
//             },
//         }
//     }
// }

// impl Hash for RuntimeValue {
// fn hash<H: Hasher>(&self, state: &mut H) {
//     use RuntimeValue::*;

//     // mix in the particular variant
//     discriminant(self).hash(state);

//     match self {
//         Nil => {}
//         Bool(b) => b.hash(state),
//         Number(n) => n.hash(state),
//         String(s) => s.hash(state),
//         Keyword(s) => s.hash(state),
//         Symbol(s) => s.hash(state),
//         List(l) => l.hash(state),
//         Vector(v) => v.hash(state),
//         Map(m) => {
//             m.size().hash(state);
//             sorted(m).for_each(|binding| binding.hash(state));
//         }
//         Set(s) => {
//             s.size().hash(state);
//             sorted(s).for_each(|elem| elem.hash(state));
//         }
//         Fn(f) => f.hash(state),
//         // FnWithCaptures(lambda) => lambda.hash(state),
//         Primitive(f) => {
//             let ptr = f as *const Primitive;
//             let identifier = ptr as usize;
//             identifier.hash(state);
//         }
//         Var(var) => {
//             var.hash(state);
//             // data.borrow().hash(state);
//             // namespace.hash(state);
//             // identifier.hash(state);
//         }
//         // Recur(v) => v.hash(state),
//         // Atom(v) => {
//         //     (*v.borrow()).hash(state);
//         // }
//         // Macro(lambda) => lambda.hash(state),
//         Exception(e) => e.hash(state),
//     }
// }
// }

// impl fmt::Debug for RuntimeValue {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         use RuntimeValue::*;

//         match self {
//             Nil => write!(f, "Nil"),
//             Bool(ref b) => write!(f, "Bool({:?})", b),
//             Number(ref n) => write!(f, "Number({:?})", n),
//             String(ref s) => write!(f, "String({:?})", s),
//             Keyword(ref id, ref ns_opt) => {
//                 write!(f, "Keyword(\"")?;
//                 if let Some(ns) = ns_opt {
//                     write!(f, "{}/", ns)?;
//                 }
//                 write!(f, "{}\")", id)
//             }
//             Symbol(ref id, ref ns_opt) => {
//                 write!(f, "Symbol(\"")?;
//                 if let Some(ns) = ns_opt {
//                     write!(f, "{}/", ns)?;
//                 }
//                 write!(f, "{}\")", id)
//             }
//             List(elems) => write!(f, "List({:?})", elems.iter().format(", ")),
//             Vector(elems) => write!(f, "Vector({:?})", elems.iter().format(", ")),
//             Map(elems) => {
//                 let mut inner = vec![];
//                 for (k, v) in elems {
//                     let mut buffer = std::string::String::new();
//                     write!(buffer, "{:?} {:?}", k, v)?;
//                     inner.push(buffer);
//                 }
//                 write!(f, "Map({:?})", inner.iter().format(", "))
//             }
//             Set(elems) => write!(f, "Set({:?})", elems.iter().format(", ")),
//             Fn(_) => write!(f, "Fn(..)"),
//             FnWithCaptures(..) => write!(f, "FnWithCaptures(..)",),
//             Primitive(_) => write!(f, "Primitive(..)"),
//             Var(VarImpl {
//                 data,
//                 namespace,
//                 identifier,
//             }) => match data.borrow().as_ref() {
//                 Some(inner) => {
//                     write!(f, "Var({:?}/{:?}, {:?})", namespace, identifier, inner)
//                 }
//                 None => write!(f, "Var({:?}/{:?}, unbound)", namespace, identifier),
//             },
//             Recur(elems) => write!(f, "Recur({:?})", elems.iter().format(" ")),
//             Atom(v) => write!(f, "Atom({:?})", *v.borrow()),
//             Macro(_) => write!(f, "Macro(..)"),
//             Exception(exception) => {
//                 write!(f, "Exception({:?})", exception)
//             }
//         }
//     }
// }

impl fmt::Display for RuntimeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeValue::Nil => write_nil(f),
            RuntimeValue::Bool(b) => write_bool(f, *b),
            RuntimeValue::Number(n) => write_number(f, *n),
            RuntimeValue::String(s) => write_string(f, s),
            RuntimeValue::LexicalSymbol(s) => write_identifer(f, s),
            RuntimeValue::Keyword(symbol) => write_keyword(f, symbol),
            RuntimeValue::Symbol(symbol) => write_symbol(f, symbol),
            RuntimeValue::Var(var) => write_var(f, var),
            RuntimeValue::List(elems) => write_list(f, elems),
            RuntimeValue::Vector(elems) => write_vector(f, elems),
            RuntimeValue::Map(elems) => write_map(f, elems),
            RuntimeValue::Set(elems) => write_set(f, elems),
            RuntimeValue::SpecialForm(_) => {
                // TODO
                //  write_fn(f),
                write_nil(f)
            }
            // FnWithCaptures(..) => write!(f, "<fn* +captures>",),
            RuntimeValue::Fn(..) => write_fn(f),
            RuntimeValue::Primitive(..) => write_primitive(f),
            // Atom(v) => write!(f, "(atom {})", *v.borrow()),
            // Macro(_) => write!(f, "<macro>"),
            RuntimeValue::Exception(exception) => {
                write!(f, "{}", exception)
            }
        }
    }
}

impl RuntimeValue {
    pub fn to_readable_string(&self) -> String {
        let mut f = String::new();

        let _ = match self {
            RuntimeValue::List(elems) => {
                write!(
                    &mut f,
                    "({})",
                    elems.iter().map(|elem| elem.to_readable_string()).join(" ")
                )
                .expect("can write to string");
            }
            RuntimeValue::Vector(elems) => {
                write!(
                    &mut f,
                    "[{}]",
                    elems.iter().map(|elem| elem.to_readable_string()).join(" ")
                )
                .expect("can write to string");
            }
            RuntimeValue::Map(elems) => {
                let mut inner = vec![];
                for (k, v) in elems {
                    let mut buffer = String::new();
                    write!(
                        buffer,
                        "{} {}",
                        k.to_readable_string(),
                        v.to_readable_string()
                    )
                    .expect("can write to string");
                    inner.push(buffer);
                }
                write!(&mut f, "{{{}}}", inner.iter().format(", ")).expect("can write to string");
            }
            RuntimeValue::Set(elems) => write!(
                &mut f,
                "#{{{}}}",
                elems
                    .iter()
                    .map(|elem| elem.to_readable_string())
                    .format(" ")
            )
            .expect("can write to string"),
            RuntimeValue::String(s) => {
                let unescaped_string = unescape_string(s);
                write!(&mut f, "\"{}\"", unescaped_string).expect("can write to string");
            }
            RuntimeValue::Exception(e) => {
                write!(&mut f, "{}", e.to_readable_string()).expect("can write to string")
            }
            other => {
                write!(&mut f, "{}", other).expect("can write to string");
            }
        };
        f
    }
}

// this `From` impl is to facilitate lang fns like `read-string` so does a translation without any "analysis"
// impl From<&AnalyzedForm> for RuntimeValue {
//     fn from(form: &AnalyzedForm) -> Self {
//         match form {
//             AnalyzedForm::LexicalSymbol(identifier) => {
//                 RuntimeValue::LexicalSymbol(identifier.clone())
//             }
//             AnalyzedForm::Var(var) => RuntimeValue::Var(var.clone()),
//             AnalyzedForm::Atom(atom) => atom.into(),
//             AnalyzedForm::List(coll) => match coll {
//                 AnalyzedList::Def(form) => {}
//                 AnalyzedList::Var(symbol) => {}
//                 AnalyzedList::Let(form) => {}
//                 AnalyzedList::Loop(form) => {}
//                 AnalyzedList::Recur(form) => {}
//                 AnalyzedList::If(form) => {}
//                 AnalyzedList::Do(form) => {}
//                 AnalyzedList::Fn(form) => {}
//                 AnalyzedList::Quote(form) => {}
//                 AnalyzedList::Quasiquote(form) => {}
//                 AnalyzedList::Defmacro(symbol, form) => {}
//                 AnalyzedList::Macroexpand(form) => {}
//                 AnalyzedList::Try(form) => {}
//                 AnalyzedList::Form(coll) => {
//                     RuntimeValue::List(coll.iter().map(From::from).collect())
//                 }
//             },
//             AnalyzedForm::Vector(coll) => {
//                 RuntimeValue::Vector(coll.iter().map(From::from).collect())
//             }
//             AnalyzedForm::Map(coll) => {
//                 RuntimeValue::Map(coll.iter().map(|(k, v)| (k.into(), v.into())).collect())
//             }
//             AnalyzedForm::Set(coll) => RuntimeValue::Set(coll.iter().map(From::from).collect()),
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use RuntimeValue::*;

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
