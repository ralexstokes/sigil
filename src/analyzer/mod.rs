mod analyzed_form;
mod analyzer;

use crate::reader::{Atom, Form};
use thiserror::Error;

pub use analyzed_form::{AnalyzedForm, List};
pub(crate) use analyzer::{analyze_fn, analyze_let, lambda_parameter_key, LetForm};

#[derive(Debug, Error, Clone)]
pub enum AnalysisError {}

pub type AnalysisResult<T> = Result<T, AnalysisError>;

fn analyze_def(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Def)
}

fn analyze_var(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Var)
}

fn analyze_let_refactor(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Let)
}

fn analyze_loop(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Loop)
}

fn analyze_recur(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Recur)
}

fn analyze_if(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::If)
}

fn analyze_do(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Do)
}

fn analyze_fn_refactor(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Fn)
}

fn analyze_quote(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Quote)
}

fn analyze_quasiquote(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Quasiquote)
}

fn analyze_unquote(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Unquote)
}

fn analyze_splice_unquote(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::SpliceUnquote)
}

fn analyze_defmacro(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Defmacro)
}

fn analyze_macroexpand(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Macroexpand)
}

fn analyze_try(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Try)
}

fn analyze_catch(args: &[Form]) -> AnalysisResult<List<'_>> {
    Ok(List::Catch)
}

fn analyze_list_with_possible_special_form<'a>(
    operator: &'a Atom,
    rest: &'a [Form],
) -> AnalysisResult<List<'a>> {
    match operator {
        Atom::Symbol(s, None) if s == "def!" => analyze_def(rest),
        Atom::Symbol(s, None) if s == "var" => analyze_var(rest),
        Atom::Symbol(s, None) if s == "let*" => analyze_let_refactor(rest),
        Atom::Symbol(s, None) if s == "loop*" => analyze_loop(rest),
        Atom::Symbol(s, None) if s == "recur" => analyze_recur(rest),
        Atom::Symbol(s, None) if s == "if" => analyze_if(rest),
        Atom::Symbol(s, None) if s == "do" => analyze_do(rest),
        Atom::Symbol(s, None) if s == "fn*" => analyze_fn_refactor(rest),
        Atom::Symbol(s, None) if s == "quote" => analyze_quote(rest),
        Atom::Symbol(s, None) if s == "quasiquote" => analyze_quasiquote(rest),
        Atom::Symbol(s, None) if s == "unquote" => analyze_unquote(rest),
        Atom::Symbol(s, None) if s == "splice-unquote" => analyze_splice_unquote(rest),
        Atom::Symbol(s, None) if s == "defmacro!" => analyze_defmacro(rest),
        Atom::Symbol(s, None) if s == "macroexpand" => analyze_macroexpand(rest),
        Atom::Symbol(s, None) if s == "try*" => analyze_try(rest),
        Atom::Symbol(s, None) if s == "catch*" => analyze_catch(rest),
        operator @ Atom::Symbol(_, None) => {
            let mut inner = vec![AnalyzedForm::Atom(operator)];
            inner.extend(rest.iter().map(analyze).collect::<Result<Vec<_>, _>>()?);
            Ok(List::Form(inner))
        }
        _ => unreachable!("only call this function with one of the prior variants"),
    }
}

fn analyze_list(forms: &[Form]) -> AnalysisResult<AnalyzedForm> {
    let inner = match forms.split_first() {
        Some((first, rest)) => match first {
            Form::Atom(atom @ Atom::Symbol(_, None)) => {
                analyze_list_with_possible_special_form(atom, rest)?
            }
            first => {
                let mut inner = vec![analyze(first)?];
                inner.extend(rest.iter().map(analyze).collect::<Result<Vec<_>, _>>()?);
                List::Form(inner)
            }
        },
        None => List::Form(vec![]),
    };
    Ok(AnalyzedForm::List(inner))
}

pub fn analyze(form: &Form) -> AnalysisResult<AnalyzedForm> {
    let inner = match form {
        Form::Atom(a) => AnalyzedForm::Atom(a),
        Form::List(elems) => analyze_list(elems)?,
        Form::Vector(elems) => {
            AnalyzedForm::Vector(elems.iter().map(analyze).collect::<Result<Vec<_>, _>>()?)
        }
        Form::Map(elems) => AnalyzedForm::Map(
            elems
                .iter()
                .map(|(x, y)| -> AnalysisResult<(AnalyzedForm, AnalyzedForm)> {
                    let analyzed_x = analyze(x)?;
                    let analyzed_y = analyze(y)?;
                    Ok((analyzed_x, analyzed_y))
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Form::Set(elems) => {
            AnalyzedForm::Set(elems.iter().map(analyze).collect::<Result<Vec<_>, _>>()?)
        }
    };
    Ok(inner)
}
