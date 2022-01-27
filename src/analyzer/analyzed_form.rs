use crate::namespace::Var;
use crate::reader::{Atom, Identifier, Symbol};

pub enum AnalyzedForm<'a> {
    Symbol(&'a Identifier),
    Var(Var),
    Atom(&'a Atom),
    List(AnalyzedList<'a>),
    Vector(Vec<AnalyzedForm<'a>>),
    Map(Vec<(AnalyzedForm<'a>, AnalyzedForm<'a>)>),
    Set(Vec<AnalyzedForm<'a>>),
}

pub enum AnalyzedList<'a> {
    // (def! symbol form?)
    Def(DefForm<'a>),
    // (var symbol)
    Var(&'a Symbol),
    // (let* [bindings*] form*)
    Let(LetForm<'a>),
    // (loop* [bindings*] form*)
    Loop(LexicalForm<'a>),
    // (recur form*)
    Recur(BodyForm<'a>),
    // (if predicate consequent alternate?)
    If(IfForm<'a>),
    // (do form*)
    Do(BodyForm<'a>),
    //(fn* [parameters*] form*)
    Fn(FnForm<'a>),
    // (quote form)
    Quote(Box<AnalyzedForm<'a>>),
    // (quasiquote form)
    Quasiquote(Box<AnalyzedForm<'a>>),
    // (unquote form)
    Unquote(Box<AnalyzedForm<'a>>),
    // (splice-unquote form)
    SpliceUnquote(Box<AnalyzedForm<'a>>),
    // (defmacro! symbol fn*-form)
    Defmacro(&'a Symbol, FnForm<'a>),
    // (macroexpand macro-form)
    Macroexpand(Box<AnalyzedForm<'a>>),
    // (try* form* catch*-form?)
    Try(TryForm<'a>),
    // (catch* exc-symbol form*)
    Catch(CatchForm<'a>),
    Form(Vec<AnalyzedForm<'a>>),
}

pub enum DefForm<'a> {
    Bound(&'a Symbol, Box<AnalyzedForm<'a>>),
    Unbound(&'a Symbol),
}

pub type LexicalBinding<'a> = (&'a Symbol, Box<AnalyzedForm<'a>>);

pub enum LexicalBindings<'a> {
    Bound(Vec<LexicalBinding<'a>>),
    Unbound(Vec<&'a Symbol>),
}

pub struct LexicalForm<'a> {
    bindings: LexicalBindings<'a>,
    body: BodyForm<'a>,
}

fn is_forward_visible(form: &AnalyzedForm<'_>) -> bool {
    matches!(form, &AnalyzedForm::List(AnalyzedList::Fn(_)))
}

impl<'a> LexicalForm<'a> {
    pub(super) fn resolve_forward_declarations(&self) -> Vec<&'a Symbol> {
        let mut result = vec![];
        match self.bindings {
            LexicalBindings::Bound(bindings) => {
                for (name, value) in bindings {
                    if is_forward_visible(value.as_ref()) {
                        result.push(name);
                    }
                }
            }
            LexicalBindings::Unbound(_) => {}
        }
        result
    }
}

pub struct LetForm<'a> {
    lexical_form: LexicalForm<'a>,
    // `let*` can produce "forward declarations" where some names
    // in `scope` can be seen by all other names
    pub forward_declarations: Vec<&'a Symbol>,
}

#[derive(Default)]
pub struct BodyForm<'a> {
    body: Vec<AnalyzedForm<'a>>,
}

pub struct IfForm<'a> {
    predicate: Box<AnalyzedForm<'a>>,
    consequent: Box<AnalyzedForm<'a>>,
    alternate: Option<Box<AnalyzedForm<'a>>>,
}

pub struct FnForm<'a> {
    parameters: Vec<&'a Symbol>,
    variadic: Option<&'a Symbol>,
    body: BodyForm<'a>,
}

impl<'a> FnForm<'a> {
    pub fn arity(&self) -> usize {
        self.parameters.len()
    }
}

#[derive(Default)]
pub struct TryForm<'a> {
    body: BodyForm<'a>,
    catch: Option<CatchForm<'a>>,
}

pub struct CatchForm<'a> {
    exception_binding: &'a Symbol,
    body: BodyForm<'a>,
}
