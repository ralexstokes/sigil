mod analyzed_form;

use crate::{
    namespace::{Context as NamespaceContext, NamespaceError},
    reader::{Atom, Form, Identifier, Symbol},
};
pub use analyzed_form::{
    AnalyzedForm, AnalyzedList, BodyForm, CatchForm, DefForm, FnForm, IfForm, LetForm,
    LexicalBindings, LexicalForm, TryForm,
};
use itertools::Itertools;
use std::collections::HashSet;
use std::ops::Range;
use thiserror::Error;

// TODO: remove once we can express unbounded range
const MAX_ARGS_BOUND: usize = 129;

const VARIADIC_PARAM_COUNT: usize = 2;
const VARIADIC_ARG: &Symbol = &Symbol {
    identifier: Identifier::from("&"),
    namespace: None,
};

#[derive(Debug, Clone, Error)]
pub enum SymbolError {
    #[error("{0}")]
    NamespaceError(#[from] NamespaceError),
}

#[derive(Debug, Clone, Error)]
#[error("incorrect type of arguments: provided {provided} but expected {expected}")]
pub struct TypeError {
    expected: String,
    provided: Form,
}

#[derive(Debug, Clone, Error)]
#[error("incorrect arity: provided {provided} args, but only accept {acceptable:?} args")]
pub struct ArityError {
    provided: usize,
    acceptable: Range<usize>,
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum LexicalError {
    #[error("bindings in lexical context must be bound")]
    BindingsMustBeBound,
    Arity(#[from] ArityError),
    Type(#[from] TypeError),
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum DefError {
    Arity(#[from] ArityError),
    Type(#[from] TypeError),
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum VarError {
    Arity(#[from] ArityError),
    Type(#[from] TypeError),
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum LetError {
    LexicalError(#[from] LexicalError),
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum LoopError {
    LexicalError(#[from] LexicalError),
}

#[derive(Debug, Clone, Error)]
pub enum RecurError {}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum IfError {
    Arity(#[from] ArityError),
}

#[derive(Debug, Clone, Error)]
pub enum DoError {}

#[derive(Debug, Clone, Error)]
pub enum FnError {
    #[error("{0}")]
    LexicalError(#[from] LexicalError),
    #[error("missing name for variadic form after `&`")]
    MissingVariadicArg,
    #[error("variadic marker `&` used out of position")]
    VariadicArgNotInTailPosition,
    #[error("variadic marker `&` duplicated in parameters")]
    VariadicArgNotUnique,
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum QuoteError {
    Arity(#[from] ArityError),
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum QuasiquoteError {
    Arity(#[from] ArityError),
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum UnquoteError {
    Arity(#[from] ArityError),
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum SpliceUnquoteError {
    Arity(#[from] ArityError),
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum DefmacroError {
    Arity(#[from] ArityError),
    Type(#[from] TypeError),
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum MacroexpandError {
    Arity(#[from] ArityError),
}

#[derive(Debug, Clone, Error)]
pub enum TryError {}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum CatchError {
    Arity(#[from] ArityError),
    Type(#[from] TypeError),
}

#[derive(Debug, Error, Clone)]
pub enum AnalysisError {
    #[error("error analyzing symbol: {0}")]
    SymbolError(#[from] SymbolError),
    #[error("error analyzing `def!`: {0}")]
    DefError(#[from] DefError),
    #[error("error analyzing `var`: {0}")]
    VarError(#[from] VarError),
    #[error("error analyzing `let*`: {0}")]
    LetError(#[from] LetError),
    #[error("error analyzing `loop*`: {0}")]
    LoopError(#[from] LoopError),
    #[error("error analyzing `recur`: {0}")]
    RecurError(#[from] RecurError),
    #[error("error analyzing `if`: {0}")]
    IfError(#[from] IfError),
    #[error("error analyzing `do`: {0}")]
    DoError(#[from] DoError),
    #[error("error analyzing `fn*`: {0}")]
    FnError(#[from] FnError),
    #[error("error analyzing `quote`: {0}")]
    QuoteError(#[from] QuoteError),
    #[error("error analyzing `quasiquote`: {0}")]
    QuasiquoteError(#[from] QuasiquoteError),
    #[error("error analyzing `unquote`: {0}")]
    UnquoteError(#[from] UnquoteError),
    #[error("error analyzing `splice-unquote`: {0}")]
    SpliceUnquoteError(#[from] SpliceUnquoteError),
    #[error("error analyzing `defmacro!`: {0}")]
    DefmacroError(#[from] DefmacroError),
    #[error("error analyzing `macroexpand`: {0}")]
    MacroexpandError(#[from] MacroexpandError),
    #[error("error analyzing `try*`: {0}")]
    TryError(#[from] TryError),
    #[error("error analyzing `catch*`: {0}")]
    CatchError(#[from] CatchError),
}

pub type AnalysisResult<T> = Result<T, AnalysisError>;

fn extract_symbol(form: &Form) -> Result<&Symbol, TypeError> {
    match form {
        Form::Atom(Atom::Symbol(symbol)) => Ok(symbol),
        e => Err(TypeError {
            expected: "symbol".into(),
            provided: e.clone(),
        }),
    }
}

fn extract_symbol_without_namespace(form: &Form) -> Result<&Symbol, TypeError> {
    match extract_symbol(form)? {
        s @ Symbol {
            namespace: None, ..
        } => Ok(s),
        e => Err(TypeError {
            expected: "symbol without namespace".into(),
            provided: Form::Atom(Atom::Symbol(e.clone())),
        }),
    }
}

fn extract_vector(form: &Form) -> Result<&Vec<Form>, TypeError> {
    match form {
        Form::Vector(v) => Ok(v),
        e => Err(TypeError {
            expected: "vector".into(),
            provided: e.clone(),
        }),
    }
}

enum LexicalMode {
    Bound,
    Unbound,
}

fn extract_lexical_form(
    forms: &[Form],
    lexical_mode: LexicalMode,
) -> Result<(&Vec<Form>, &[Form]), LexicalError> {
    verify_arity(forms, 1..MAX_ARGS_BOUND).map_err(|err| LexicalError::Arity(err))?;

    let bindings = extract_vector(&forms[0]).map_err(LexicalError::Type)?;

    if matches!(lexical_mode, LexicalMode::Bound) && bindings.len() % 2 != 0 {
        return Err(LexicalError::BindingsMustBeBound);
    }

    let body = &forms[1..];
    Ok((bindings, body))
}

fn verify_arity(forms: &[Form], range: Range<usize>) -> Result<(), ArityError> {
    if range.contains(&forms.len()) {
        Ok(())
    } else {
        Err(ArityError {
            provided: forms.len(),
            acceptable: range,
        })
    }
}

fn analyze_tail_fn_parameters<'a>(parameters: &'a [&Symbol]) -> AnalysisResult<Option<&'a Symbol>> {
    match parameters {
        &[a, b] => {
            if a == VARIADIC_ARG {
                if b == VARIADIC_ARG {
                    Err(AnalysisError::FnError(FnError::VariadicArgNotUnique))
                } else {
                    Ok(Some(b))
                }
            } else {
                Ok(None)
            }
        }
        _ => unreachable!("only call with a slice of two parameters"),
    }
}

#[derive(Debug)]
pub struct Analyzer<'ana> {
    namespaces: &'ana NamespaceContext,
    scopes: Vec<HashSet<&'ana Identifier>>,
}

impl<'ana> Analyzer<'ana> {
    pub fn new(namespaces: &'ana NamespaceContext) -> Self {
        Self {
            namespaces,
            scopes: vec![],
        }
    }

    // (def! name value?)
    fn analyze_def<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        verify_arity(args, 1..3).map_err(DefError::from)?;

        let name = extract_symbol(&args[0]).map_err(DefError::from)?;
        let form = if args.len() == 2 {
            let analyzed_value = self.analyze(&args[1])?;
            DefForm::Bound(name, Box::new(analyzed_value))
        } else {
            DefForm::Unbound(name)
        };

        Ok(AnalyzedList::Def(form))
    }

    // (var name)
    fn analyze_var<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        verify_arity(args, 1..2).map_err(VarError::from)?;

        let symbol = extract_symbol(&args[0])
            .map(AnalyzedList::Var)
            .map_err(VarError::from)?;

        Ok(symbol)
    }

    fn analyze_lexical_bindings<'a, E>(
        &'a self,
        bindings_form: &'a Vec<Form>,
        lexical_mode: LexicalMode,
    ) -> AnalysisResult<LexicalBindings<'a>>
    where
        E: From<LexicalError>,
        AnalysisError: From<E>,
    {
        match lexical_mode {
            LexicalMode::Bound => {
                let mut bindings = vec![];
                for (name, value) in bindings_form.iter().tuples() {
                    let analyzed_name =
                        extract_symbol_without_namespace(name).map_err(|e| E::from(e.into()))?;
                    let analyzed_value = self.analyze(value)?;
                    bindings.push((analyzed_name, Box::new(analyzed_value)));
                }
                Ok(LexicalBindings::Bound(bindings))
            }
            LexicalMode::Unbound => {
                let parameters = bindings_form
                    .iter()
                    .map(extract_symbol_without_namespace)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| E::from(e.into()))?;
                Ok(LexicalBindings::Unbound(parameters))
            }
        }
    }

    fn analyze_lexical_form<'a: 'ana, E>(
        &'a mut self,
        forms: &'a [Form],
        lexical_mode: LexicalMode,
    ) -> AnalysisResult<LexicalForm<'a>>
    where
        E: From<LexicalError>,
        AnalysisError: From<E>,
    {
        let (bindings_form, body) = extract_lexical_form(forms, lexical_mode).map_err(E::from)?;

        let bindings = self.analyze_lexical_bindings::<E>(bindings_form, lexical_mode)?;

        let mut scope = HashSet::new();
        match bindings {
            LexicalBindings::Bound(bindings) => {
                for (name, _) in bindings {
                    scope.insert(&name.identifier);
                }
            }
            LexicalBindings::Unbound(bindings) => {
                for name in bindings {
                    scope.insert(&name.identifier);
                }
            }
        }
        self.scopes.push(scope);

        let body = body
            .iter()
            .map(|form| self.analyze(form))
            .collect::<Result<Vec<_>, _>>()?;

        self.scopes.pop();

        Ok(LexicalForm {
            bindings,
            body: BodyForm { body },
        })
    }

    // (let* [bindings*] body*)
    fn analyze_let<'a: 'ana>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        let lexical_form = self.analyze_lexical_form::<LetError>(args, LexicalMode::Bound)?;
        let forward_declarations = lexical_form.resolve_forward_declarations();
        Ok(AnalyzedList::Let(LetForm {
            lexical_form,
            forward_declarations,
        }))
    }

    // (loop* [bindings*] body*)
    fn analyze_loop<'a: 'ana>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        let lexical_form = self.analyze_lexical_form::<LoopError>(args, LexicalMode::Bound)?;
        Ok(AnalyzedList::Loop(lexical_form))
    }

    // (recur body*)
    fn analyze_recur<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        let body = args
            .iter()
            .map(|form| self.analyze(form))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AnalyzedList::Recur(BodyForm { body }))
    }

    // (if predicate consequent alternate?)
    fn analyze_if<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        verify_arity(args, 2..4).map_err(IfError::from)?;

        let predicate = Box::new(self.analyze(&args[0])?);
        let consequent = Box::new(self.analyze(&args[1])?);

        let alternate = if args.len() == 3 {
            Some(Box::new(self.analyze(&args[2])?))
        } else {
            None
        };

        Ok(AnalyzedList::If(IfForm {
            predicate,
            consequent,
            alternate,
        }))
    }

    // (do body*)
    fn analyze_do<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        let body = args
            .iter()
            .map(|form| self.analyze(form))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AnalyzedList::Do(BodyForm { body }))
    }

    fn analyze_fn_parameters<'a>(
        &'a self,
        parameters: Vec<&'a Symbol>,
    ) -> AnalysisResult<(Vec<&'a Symbol>, Option<&'a Symbol>)> {
        match parameters.len() {
            0 => Ok((parameters, None)),
            1 => {
                if parameters[0] == VARIADIC_ARG {
                    Err(AnalysisError::FnError(FnError::MissingVariadicArg))
                } else {
                    Ok((parameters, None))
                }
            }
            _ => {
                let variadic_position = parameters.len() - VARIADIC_PARAM_COUNT;
                let (fixed_parameters, possibly_variadic) = parameters.split_at(variadic_position);
                let valid_fixed_parameters = fixed_parameters
                    .iter()
                    .all(|&parameter| parameter != VARIADIC_ARG);
                if !valid_fixed_parameters {
                    return Err(AnalysisError::FnError(
                        FnError::VariadicArgNotInTailPosition,
                    ));
                }

                let variadic_parameter = analyze_tail_fn_parameters(possibly_variadic)?;
                if variadic_parameter.is_some() {
                    parameters.truncate(variadic_position);
                }
                Ok((parameters, variadic_parameter))
            }
        }
    }

    fn extract_fn_form<'a: 'ana>(&'a self, args: &'a [Form]) -> AnalysisResult<FnForm<'a>> {
        let LexicalForm { bindings, body } =
            self.analyze_lexical_form::<FnError>(args, LexicalMode::Unbound)?;
        let parameters = match bindings {
            LexicalBindings::Unbound(parameters) => parameters,
            _ => unreachable!("lexical bindings have been validated to only have unbound symbols"),
        };

        let (parameters, variadic) = self.analyze_fn_parameters(parameters)?;

        Ok(FnForm {
            parameters,
            variadic,
            body,
        })
    }

    // (fn [parameters*] body*)
    fn analyze_fn<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        let fn_form = self.extract_fn_form(args)?;

        Ok(AnalyzedList::Fn(fn_form))
    }

    // (quote form)
    fn analyze_quote<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        verify_arity(args, 1..2).map_err(QuoteError::from)?;

        let form = self.analyze(&args[0])?;

        Ok(AnalyzedList::Quote(Box::new(form)))
    }

    // (quasiquote form)
    fn analyze_quasiquote<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        verify_arity(args, 1..2).map_err(QuasiquoteError::from)?;

        let form = self.analyze(&args[0])?;

        Ok(AnalyzedList::Quasiquote(Box::new(form)))
    }

    // // (unquote form)
    // fn analyze_unquote<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
    //     verify_arity(args, 1..2).map_err(UnquoteError::from)?;

    //     let form = self.analyze(&args[0])?;

    //     Ok(AnalyzedList::Unquote(Box::new(form)))
    // }

    // // (splice-unquote form)
    // fn analyze_splice_unquote<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
    //     verify_arity(args, 1..2).map_err(SpliceUnquoteError::from)?;

    //     let form = self.analyze(&args[0])?;

    //     Ok(AnalyzedList::SpliceUnquote(Box::new(form)))
    // }

    // (defmacro! symbol fn*-form)
    fn analyze_defmacro<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        verify_arity(args, 2..3).map_err(DefmacroError::from)?;

        let name = extract_symbol(&args[0]).map_err(DefmacroError::from)?;
        let body = self.extract_fn_form(&args[1..])?;

        Ok(AnalyzedList::Defmacro(name, body))
    }

    // (macroexpand form)
    fn analyze_macroexpand<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        verify_arity(args, 1..2).map_err(MacroexpandError::from)?;

        let form = self.analyze(&args[0])?;

        Ok(AnalyzedList::Macroexpand(Box::new(form)))
    }

    // (try* form* catch*-form?)
    fn analyze_try<'a>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        let try_form = match args.len() {
            0 => TryForm::default(),
            _ => {
                let (last_form, body_forms) = args.split_last().expect("has at least 1 element");
                let mut body = body_forms
                    .iter()
                    .map(|form| self.analyze(form))
                    .collect::<Result<Vec<_>, _>>()?;
                let catch = match self.analyze(last_form)? {
                    AnalyzedForm::List(AnalyzedList::Catch(catch_form)) => Some(catch_form),
                    form => {
                        body.push(form);
                        None
                    }
                };

                TryForm {
                    body: BodyForm { body },
                    catch,
                }
            }
        };

        Ok(AnalyzedList::Try(try_form))
    }

    // (catch* exc-symbol form*)
    fn analyze_catch<'a: 'ana>(&'a self, args: &'a [Form]) -> AnalysisResult<AnalyzedList<'a>> {
        verify_arity(args, 1..MAX_ARGS_BOUND).map_err(CatchError::from)?;

        let exception_binding =
            extract_symbol_without_namespace(&args[0]).map_err(CatchError::from)?;

        let mut scope = HashSet::from([&exception_binding.identifier]);
        self.scopes.push(scope);

        let body = args[1..]
            .iter()
            .map(|form| self.analyze(form))
            .collect::<Result<Vec<_>, _>>()?;

        self.scopes.pop();

        Ok(AnalyzedList::Catch(CatchForm {
            exception_binding,
            body: BodyForm { body },
        }))
    }

    fn analyze_list_with_possible_special_form<'a>(
        &'a self,
        operator: &'a Atom,
        rest: &'a [Form],
    ) -> AnalysisResult<AnalyzedList<'a>> {
        match operator {
            Atom::Symbol(symbol) => match symbol {
                Symbol {
                    identifier: s,
                    namespace: None,
                } => match s.as_str() {
                    "def!" => self.analyze_def(rest),
                    "var" => self.analyze_var(rest),
                    "let*" => self.analyze_let(rest),
                    "loop*" => self.analyze_loop(rest),
                    "recur" => self.analyze_recur(rest),
                    "if" => self.analyze_if(rest),
                    "do" => self.analyze_do(rest),
                    "fn*" => self.analyze_fn(rest),
                    "quote" => self.analyze_quote(rest),
                    "quasiquote" => self.analyze_quasiquote(rest),
                    // "unquote" => self.analyze_unquote(rest),
                    // "splice-unquote" => self.analyze_splice_unquote(rest),
                    "defmacro!" => self.analyze_defmacro(rest),
                    "macroexpand" => self.analyze_macroexpand(rest),
                    "try*" => self.analyze_try(rest),
                    "catch*" => self.analyze_catch(rest),
                    _ => {
                        let mut inner = vec![AnalyzedForm::Atom(operator)];
                        inner.extend(
                            rest.iter()
                                .map(|f| self.analyze(f))
                                .collect::<Result<Vec<_>, _>>()?,
                        );
                        Ok(AnalyzedList::Form(inner))
                    }
                },
                _ => unreachable!("only call this function with one of the prior variants"),
            },
            _ => unreachable!("only call this function with one of the prior variants"),
        }
    }

    fn analyze_list<'a>(&'a self, forms: &'a [Form]) -> AnalysisResult<AnalyzedForm<'a>> {
        let inner = match forms.split_first() {
            Some((first, rest)) => match first {
                Form::Atom(
                    atom @ Atom::Symbol(Symbol {
                        namespace: None, ..
                    }),
                ) => self.analyze_list_with_possible_special_form(atom, rest)?,
                first => {
                    let mut inner = vec![self.analyze(first)?];
                    inner.extend(
                        rest.iter()
                            .map(|f| self.analyze(f))
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                    AnalyzedList::Form(inner)
                }
            },
            None => AnalyzedList::Form(vec![]),
        };
        Ok(AnalyzedForm::List(inner))
    }

    fn identifier_is_in_lexical_scope(&self, identifier: &Identifier) -> bool {
        for scope in self.scopes.iter().rev() {
            if scope.contains(identifier) {
                return true;
            }
        }
        false
    }

    pub fn analyze_symbol<'a>(&'a self, symbol: &'a Symbol) -> AnalysisResult<AnalyzedForm<'a>> {
        let form = match self.namespaces.resolve_symbol(symbol) {
            Ok(var) => AnalyzedForm::Var(var),
            Err(e @ NamespaceError::MissingIdentifier(..)) => {
                let identifier = &symbol.identifier;
                if self.identifier_is_in_lexical_scope(identifier) {
                    AnalyzedForm::LexicalSymbol(identifier)
                } else {
                    return Err(SymbolError::from(e).into());
                }
            }
            Err(err) => return Err(SymbolError::from(err).into()),
        };
        Ok(form)
    }

    // `analyze` performs static analysis of `form` to provide data optimized for evaluation
    // 1. Syntax: can an `AnalyzedForm` be produced from a `Form`
    // 2. Semantics: some `AnalyzedForm`s have some invariants that can be verified statically
    //    like constraints on special forms
    pub fn analyze<'a>(&'a self, form: &'a Form) -> AnalysisResult<AnalyzedForm<'a>> {
        let analyzed_form = match form {
            Form::Atom(Atom::Symbol(s)) => self.analyze_symbol(s)?,
            Form::Atom(a) => AnalyzedForm::Atom(a),
            Form::List(elems) => self.analyze_list(elems)?,
            Form::Vector(elems) => AnalyzedForm::Vector(
                elems
                    .iter()
                    .map(|f| self.analyze(f))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Form::Map(elems) => AnalyzedForm::Map(
                elems
                    .iter()
                    .map(|(x, y)| -> AnalysisResult<(AnalyzedForm, AnalyzedForm)> {
                        let analyzed_x = self.analyze(x)?;
                        let analyzed_y = self.analyze(y)?;
                        Ok((analyzed_x, analyzed_y))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Form::Set(elems) => AnalyzedForm::Set(
                elems
                    .iter()
                    .map(|f| self.analyze(f))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        };
        Ok(analyzed_form)
    }
}
