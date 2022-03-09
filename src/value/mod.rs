mod atom;
mod var;

use crate::collections::{PersistentList, PersistentMap, PersistentSet, PersistentVector};
use crate::interpreter::{EvaluationError, EvaluationResult, Interpreter};
use crate::reader::{Atom, Identifier, Symbol};
use crate::writer::{
    unescape_string, write_bool, write_exception, write_fn, write_identifer, write_keyword,
    write_list, write_macro, write_map, write_nil, write_number, write_primitive, write_set,
    write_special_form, write_string, write_symbol, write_var, write_vector,
};
pub use atom::AtomRef;
use itertools::{sorted, Itertools};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};
use std::mem::discriminant;
use std::ops::Deref;
pub use var::Var;

pub type Scope = HashMap<Identifier, RuntimeValue>;
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
    Loop(LetForm),
    // (recur form*)
    Recur(BodyForm),
    // (if predicate consequent alternate?)
    If(IfForm),
    // (do form*)
    Do(BodyForm),
    //(fn* [parameters*] form*)
    Fn(FnImpl),
    // (quote form)
    Quote(Box<RuntimeValue>),
    // (quasiquote form)
    Quasiquote(Box<RuntimeValue>),
    // (unquote form)
    Unquote(Box<RuntimeValue>),
    // (splice-unquote form)
    SpliceUnquote(Box<RuntimeValue>),
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

pub type LexicalBinding = (Identifier, RuntimeValue);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LexicalForm {
    pub bindings: Vec<LexicalBinding>,
    pub body: BodyForm,
    pub forward_declarations: HashSet<usize>,
}

impl Hash for LexicalForm {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bindings.hash(state);
        self.body.hash(state);
        let ids = self.forward_declarations.iter().sorted();
        for id in ids {
            id.hash(state);
        }
    }
}

impl PartialOrd for LexicalForm {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.bindings.cmp(&other.bindings) {
            Ordering::Equal => match self.body.cmp(&other.body) {
                Ordering::Equal => Some(
                    sorted(self.forward_declarations.iter())
                        .cmp(sorted(other.forward_declarations.iter())),
                ),
                other => Some(other),
            },
            other => Some(other),
        }
    }
}

impl Ord for LexicalForm {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl LexicalForm {
    pub fn identifier_for_binding(&self, index: usize) -> Option<&Identifier> {
        self.bindings.get(index).map(|binding| &binding.0)
    }
}

pub type LetForm = LexicalForm;

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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FnImpl {
    Default(FnForm),
    WithCaptures(FnWithCapturesImpl),
}

impl Deref for FnImpl {
    type Target = FnForm;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Default(form) => form,
            Self::WithCaptures(inner) => &inner.form,
        }
    }
}

pub type Captures = HashMap<Identifier, Var>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FnWithCapturesImpl {
    pub form: FnForm,
    pub captures: Captures,
}

impl Hash for FnWithCapturesImpl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.form.hash(state);
        let captures = self.captures.iter().sorted();
        for (name, value) in captures {
            name.hash(state);
            value.hash(state);
        }
    }
}

impl PartialOrd for FnWithCapturesImpl {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.form.cmp(&other.form) {
            Ordering::Equal => {
                Some(sorted(self.captures.iter()).cmp(sorted(other.captures.iter())))
            }
            other => Some(other),
        }
    }
}

impl Ord for FnWithCapturesImpl {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl FnWithCapturesImpl {
    pub fn update_captures(&mut self, scopes: &Vec<Scope>) {
        for (identifier, capture) in self.captures.iter_mut() {
            for scope in scopes.iter().rev() {
                if let Some(value) = scope.get(identifier) {
                    capture.update(value.clone());
                    break;
                }
            }
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
    Fn(FnImpl),
    Primitive(Primitive),
    Exception(ExceptionImpl),
    Atom(AtomRef),
    Macro(FnImpl),
}

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
            RuntimeValue::SpecialForm(form) => write_special_form(f, form),
            RuntimeValue::Fn(..) => write_fn(f),
            RuntimeValue::Primitive(..) => write_primitive(f),
            RuntimeValue::Exception(exception) => write_exception(f, exception),
            RuntimeValue::Atom(v) => write!(f, "(atom {})", v.value()),
            RuntimeValue::Macro(..) => write_macro(f),
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
