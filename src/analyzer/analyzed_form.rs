use crate::reader::Atom;

pub enum AnalyzedForm<'a> {
    Atom(&'a Atom),
    List(List<'a>),
    Vector(Vec<AnalyzedForm<'a>>),
    Map(Vec<(AnalyzedForm<'a>, AnalyzedForm<'a>)>),
    Set(Vec<AnalyzedForm<'a>>),
}

pub enum List<'a> {
    Def,
    Var,
    Let,
    Loop,
    Recur,
    If,
    Do,
    Fn,
    Quote,
    Quasiquote,
    Unquote,
    SpliceUnquote,
    Defmacro,
    Macroexpand,
    Try,
    Catch,
    Form(Vec<AnalyzedForm<'a>>),
}
