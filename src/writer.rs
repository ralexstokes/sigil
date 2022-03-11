use crate::reader::{Identifier, Symbol};
use crate::value::{
    BodyForm, DefForm, ExceptionImpl, FnForm, LexicalBinding, LocatedVar, SpecialForm,
};
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

pub(crate) fn write_located_var(f: &mut fmt::Formatter<'_>, var: &LocatedVar) -> fmt::Result {
    write!(f, "#'")?;
    write_symbol(f, &var.symbol)
}

}

pub(crate) fn write_exception(
    f: &mut fmt::Formatter<'_>,
    exception: &ExceptionImpl,
) -> fmt::Result {
    write!(f, "{exception}")
}

pub(crate) fn write_macro(f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "<macro>")
}

fn write_lexical_bindings(
    f: &mut fmt::Formatter<'_>,
    bindings: &Vec<LexicalBinding>,
) -> fmt::Result {
    write!(f, "[")?;
    if let Some(((last_name, last_value), prefix)) = bindings.split_last() {
        for (name, value) in prefix {
            write!(f, "{name} {value} ")?;
        }

        write!(f, "{last_name} {last_value}")?;
    }
    write!(f, "]")
}

fn write_body(f: &mut fmt::Formatter<'_>, body: &BodyForm) -> fmt::Result {
    write!(f, "{}", join(&body.body, " "))
}

fn write_fn_form(f: &mut fmt::Formatter<'_>, fn_form: &FnForm) -> fmt::Result {
    write!(f, "(fn* [{}", join(&fn_form.parameters, " "))?;
    if let Some(var_arg) = &fn_form.variadic {
        write!(f, "& {}", var_arg)?;
    }
    write!(f, "] ")?;
    write_body(f, &fn_form.body)?;
    write!(f, ")")
}

pub(crate) fn write_special_form(f: &mut fmt::Formatter<'_>, form: &SpecialForm) -> fmt::Result {
    match form {
        SpecialForm::Def(form) => {
            write!(f, "(def! ")?;
            match form {
                DefForm::Bound(name, value) => {
                    write_symbol(f, name)?;
                    write!(f, "{value}")?;
                }
                DefForm::Unbound(name) => {
                    write_symbol(f, name)?;
                }
            }
            write!(f, ")")
        }
        SpecialForm::Var(symbol) => {
            write!(f, "(var {symbol})")
        }
        SpecialForm::Let(form) => {
            write!(f, "(let* ")?;
            write_lexical_bindings(f, &form.bindings)?;
            write!(f, " ")?;
            write_body(f, &form.body)?;
            write!(f, ")")
        }
        SpecialForm::Loop(form) => {
            write!(f, "(loop* ")?;
            write_lexical_bindings(f, &form.bindings)?;
            write!(f, " ")?;
            write_body(f, &form.body)?;
            write!(f, ")")
        }
        SpecialForm::Recur(form) => {
            write!(f, "(recur ")?;
            write_body(f, form)?;
            write!(f, ")")
        }
        SpecialForm::If(form) => {
            write!(f, "(if ")?;
            write!(f, "{} ", form.predicate)?;
            write!(f, "{}", form.consequent)?;
            if let Some(form) = &form.alternate {
                write!(f, " {}", form)?;
            }
            write!(f, ")")
        }
        SpecialForm::Do(form) => {
            write!(f, "(do ")?;
            write_body(f, form)?;
            write!(f, ")")
        }
        SpecialForm::Fn(fn_form) => write_fn_form(f, fn_form),
        SpecialForm::Quote(form) => {
            write!(f, "(quote {form})")
        }
        SpecialForm::Quasiquote(form) => {
            write!(f, "(quasiquote {form})")
        }
        SpecialForm::Unquote(form) => {
            write!(f, "(unquote {form})")
        }
        SpecialForm::SpliceUnquote(form) => {
            write!(f, "(splice-unquote {form})")
        }
        SpecialForm::Defmacro(name, fn_form) => {
            write!(f, "(defmacro! {name}")?;
            write_fn_form(f, fn_form)?;
            write!(f, ")")
        }
        SpecialForm::Macroexpand(form) => {
            write!(f, "(macroexpand {form})")
        }
        SpecialForm::Try(form) => {
            write!(f, "(try* ")?;
            write_body(f, &form.body)?;
            if let Some(catch) = &form.catch {
                write!(f, "(catch* ")?;
                write!(f, "{} ", catch.exception_binding)?;
                write_body(f, &catch.body)?;
                write!(f, ")")?;
            }
            write!(f, ")")
        }
    }
}
