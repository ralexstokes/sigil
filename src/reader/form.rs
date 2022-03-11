use crate::writer::{
    write_bool, write_keyword, write_list, write_map, write_nil, write_number, write_set,
    write_string, write_symbol, write_vector,
};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Form {
    Atom(Atom),
    List(Vec<Form>),
    Vector(Vec<Form>),
    Map(Vec<(Form, Form)>),
    Set(Vec<Form>),
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub enum Atom {
    Nil,
    Bool(bool),
    Number(i64),
    String(String),
    Keyword(Symbol),
    Symbol(Symbol),
}

pub type Identifier = String;

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub identifier: Identifier,
    pub namespace: Option<Identifier>,
}

impl Symbol {
    pub fn simple(&self) -> bool {
        self.namespace.is_none()
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_symbol(f, self)
    }
}

impl fmt::Display for Form {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Form::Atom(atom) => match atom {
                Atom::Nil => write_nil(f),
                Atom::Bool(b) => write_bool(f, *b),
                Atom::Number(n) => write_number(f, *n),
                Atom::String(s) => write_string(f, s),
                Atom::Keyword(symbol) => write_keyword(f, symbol),
                Atom::Symbol(symbol) => write_symbol(f, symbol),
            },
            Form::List(elems) => write_list(f, elems),
            Form::Vector(elems) => write_vector(f, elems),
            Form::Map(elems) => write_map(f, elems.iter().map(|(a, b)| (a, b))),
            Form::Set(elems) => write_set(f, elems),
        }
    }
}
