use itertools::join;
use std::fmt::{self, Write};

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

fn write_symbol(f: &mut fmt::Formatter<'_>, id: &str, ns_opt: &Option<String>) -> fmt::Result {
    if let Some(ns) = ns_opt {
        write!(f, "{}/", ns)?;
    }
    write!(f, "{}", id)
}

pub(crate) fn unescape_string(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut iter = input.chars().peekable();
    while let Some(ch) = iter.peek() {
        let ch = *ch;
        match ch {
            '\\' => {
                result.push('\\');
                result.push('\\');
                iter.next().expect("from peek");
            }
            '\n' => {
                result.push('\\');
                result.push('n');
                iter.next().expect("from peek");
            }
            '\"' => {
                result.push('\\');
                result.push('"');
                iter.next().expect("from peek");
            }
            ch => {
                result.push(ch);
                iter.next().expect("from peek");
            }
        };
    }
    result
}

impl fmt::Display for Form {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Form::*;

        match self {
            Nil => write!(f, "nil"),
            Bool(b) => write!(f, "{}", b),
            Number(n) => write!(f, "{}", n),
            String(s) => write!(f, "\"{}\"", unescape_string(s)),
            Keyword(id, ns_opt) => {
                write!(f, ":")?;
                write_symbol(f, id, ns_opt)
            }
            Symbol(id, ns_opt) => write_symbol(f, id, ns_opt),
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
