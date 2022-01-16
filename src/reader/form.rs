use crate::writer::{
    write_bool, write_keyword, write_list, write_map, write_nil, write_number, write_set,
    write_string, write_symbol, write_vector,
};
use std::fmt;

pub(super) fn list_from(elems: Vec<Form>) -> Form {
    Form::List(elems)
}

pub(super) fn vector_from(elems: Vec<Form>) -> Form {
    Form::Vector(elems)
}
pub(super) fn map_from(elems: Vec<(Form, Form)>) -> Form {
    Form::Map(elems)
}

pub(super) fn set_from(elems: Vec<Form>) -> Form {
    Form::Set(elems)
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Form {
    Nil,
    Bool(bool),
    Number(i64),
    String(String),
    // identifier with optional namespace
    Keyword(String, Option<String>),
    // identifier with optional namespace
    Symbol(String, Option<String>),
    List(Vec<Form>),
    Vector(Vec<Form>),
    Map(Vec<(Form, Form)>),
    Set(Vec<Form>),
}

impl fmt::Display for Form {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Form::*;

        match self {
            Nil => write_nil(f),
            Bool(b) => write_bool(f, *b),
            Number(n) => write_number(f, *n),
            String(s) => write_string(f, s),
            Keyword(id, ns_opt) => write_keyword(f, id, ns_opt),
            Symbol(id, ns_opt) => write_symbol(f, id, ns_opt),
            List(elems) => write_list(f, elems),
            Vector(elems) => write_vector(f, elems),
            Map(elems) => write_map(f, elems),
            Set(elems) => write_set(f, elems),
        }
    }
}
