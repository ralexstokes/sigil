use crate::reader::{Identifier, Symbol};
use crate::value::Var;
use itertools::join;
use std::fmt::{self, Display, Write};
use std::string::String as StdString;

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

pub(crate) fn write_nil(f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "nil")
}

pub(crate) fn write_bool(f: &mut fmt::Formatter<'_>, b: bool) -> fmt::Result {
    write!(f, "{}", b)
}

pub(crate) fn write_number(f: &mut fmt::Formatter<'_>, n: i64) -> fmt::Result {
    write!(f, "{}", n)
}

pub(crate) fn write_string(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
    write!(f, "\"{}\"", unescape_string(s))
}

pub(crate) fn write_identifer(f: &mut fmt::Formatter<'_>, identifier: &Identifier) -> fmt::Result {
    write!(f, "{}", identifier)
}

pub(crate) fn write_keyword(f: &mut fmt::Formatter<'_>, symbol: &Symbol) -> fmt::Result {
    write!(f, ":")?;
    write_symbol(f, symbol)
}

pub(crate) fn write_symbol(
    f: &mut fmt::Formatter<'_>,
    Symbol {
        identifier,
        namespace,
    }: &Symbol,
) -> fmt::Result {
    if let Some(ns) = namespace {
        write!(f, "{}/", ns)?;
    }
    write_identifer(f, identifier)
}

pub(crate) fn write_list<I>(f: &mut fmt::Formatter<'_>, elems: I) -> fmt::Result
where
    I: IntoIterator,
    I::Item: Display,
{
    write!(f, "({})", join(elems, " "))
}

pub(crate) fn write_vector<I>(f: &mut fmt::Formatter<'_>, elems: I) -> fmt::Result
where
    I: IntoIterator,
    I::Item: Display,
{
    write!(f, "[{}]", join(elems, " "))
}

pub(crate) fn write_map<'a, I, E>(f: &'a mut fmt::Formatter<'_>, elems: I) -> fmt::Result
where
    I: IntoIterator<Item = (&'a E, &'a E)>,
    E: Display + 'a,
{
    let mut inner = vec![];
    for (k, v) in elems {
        let mut buffer = StdString::new();
        write!(buffer, "{} {}", k, v)?;
        inner.push(buffer);
    }
    write!(f, "{{{}}}", join(inner, ", "))
}

pub(crate) fn write_set<I>(f: &mut fmt::Formatter<'_>, elems: I) -> fmt::Result
where
    I: IntoIterator,
    I::Item: Display,
{
    write!(f, "#{{{}}}", join(elems, " "))
}

pub(crate) fn write_fn(f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "<fn*>")
}

pub(crate) fn write_primitive(f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "<primitive fn>")
}

pub(crate) fn write_var(f: &mut fmt::Formatter<'_>, var: &Var) -> fmt::Result {
    match var {
        Var::Bound(data) => {
            // TODO how to display name...
            write!(f, "<bound var #'TODO>")
        }
        Var::Unbound => {
            write!(f, "<unbound var #'TODO>")
        }
    }
}
