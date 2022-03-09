use crate::{
    collections::PersistentList,
    namespace::{Context as NamespaceContext, NamespaceError},
    reader::{Atom, Form, Identifier, Symbol},
    value::{
        BodyForm, Captures, CatchForm, DefForm, FnForm, FnImpl, FnWithCapturesImpl, IfForm,
        LexicalBinding, LexicalForm, RuntimeValue, SpecialForm, TryForm, Var,
    },
};
use itertools::Itertools;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::rc::Rc;
use thiserror::Error;

// TODO: remove once we can express unbounded range
const MAX_ARGS_BOUND: usize = 129;

const VARIADIC_PARAM_COUNT: usize = 2;
const VARIADIC_IDENTIFIER: &str = "&";
const CATCH_IDENTIFIER: &str = "catch*";

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
    UnquoteError(#[from] UnquoteError),
    SpliceUnquoteError(#[from] SpliceUnquoteError),
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum QuotingError {
    #[error("form called outside a `quasiquote` context")]
    OutsideQuotedContext,
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum UnquoteError {
    Quoting(#[from] QuotingError),
    Arity(#[from] ArityError),
}

#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub enum SpliceUnquoteError {
    Quoting(#[from] QuotingError),
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

fn extract_symbol(form: &Form) -> Result<Symbol, TypeError> {
    match form {
        Form::Atom(Atom::Symbol(symbol)) => Ok(symbol.clone()),
        e => Err(TypeError {
            expected: "symbol".into(),
            provided: e.clone(),
        }),
    }
}

fn extract_symbol_without_namespace(form: &Form) -> Result<Identifier, TypeError> {
    match extract_symbol(form)? {
        Symbol {
            identifier,
            namespace: None,
        } => Ok(identifier.clone()),
        e => Err(TypeError {
            expected: "symbol without namespace".into(),
            provided: Form::Atom(Atom::Symbol(e.clone())),
        }),
    }
}

fn extract_vector(form: &Form) -> Result<Vec<Form>, TypeError> {
    match form {
        Form::Vector(v) => Ok(v.clone()),
        e => Err(TypeError {
            expected: "vector".into(),
            provided: e.clone(),
        }),
    }
}

fn extract_lexical_bindings(form: &Form) -> Result<Vec<Form>, LexicalError> {
    extract_vector(form).map_err(LexicalError::Type)
}

fn extract_lexical_form(forms: &[Form]) -> Result<(Vec<Form>, Vec<Form>), LexicalError> {
    verify_arity(forms, 1..MAX_ARGS_BOUND).map_err(|err| LexicalError::Arity(err))?;
    let bindings = extract_lexical_bindings(&forms[0])?;

    if bindings.len() % 2 != 0 {
        return Err(LexicalError::BindingsMustBeBound);
    }

    let body = Vec::from(&forms[1..]);
    Ok((bindings, body))
}

fn index_for_name(bindings: &[LexicalBinding], target: &Identifier) -> usize {
    bindings
        .iter()
        .position(|(name, _)| name == target)
        .unwrap()
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

fn analyze_tail_fn_parameters(parameters: &[Identifier]) -> AnalysisResult<Option<Identifier>> {
    match &parameters {
        &[a, b] => {
            if a == VARIADIC_IDENTIFIER {
                if b == VARIADIC_IDENTIFIER {
                    Err(AnalysisError::FnError(FnError::VariadicArgNotUnique))
                } else {
                    Ok(Some(b.clone()))
                }
            } else {
                Ok(None)
            }
        }
        _ => unreachable!("only call with a slice of two parameters"),
    }
}

fn verify_quasiquoted_context(contexts: &[Context]) -> Result<(), QuotingError> {
    for context in contexts.iter().rev() {
        if matches!(context, Context::Quasiquote) {
            return Ok(());
        }
    }
    Err(QuotingError::OutsideQuotedContext)
}

type LexicalScope = HashSet<Identifier>;

type GlobalScope = HashSet<Symbol>;

type Scopes = Vec<LexicalScope>;

fn lexical_frame_for_identifier(scopes: &Scopes, identifier: &Identifier) -> Option<usize> {
    for (index, scope) in scopes.iter().enumerate().rev() {
        if scope.contains(identifier) {
            return Some(index);
        }
    }
    None
}

fn is_forward_visible(form: &Form) -> bool {
    match form {
        Form::List(elems) => {
            if let Some(first) = elems.first() {
                return matches!(first, Form::Atom(Atom::Symbol(Symbol{
                    identifier,
                    namespace: None
                })) if identifier == "fn*");
            }
            false
        }
        _ => false,
    }
}

#[derive(Debug)]
enum Context {
    Default,
    Quote,
    Quasiquote,
}

#[derive(Debug)]
pub struct Analyzer {
    namespaces: Rc<RefCell<NamespaceContext>>,
    scopes: Scopes,
    // track mutations to namespaces, e.g. via simulated `def!`
    global_scope: GlobalScope,
    // track identifiers that outlive their lexical lifetime
    // set of (identifiers, origin_frame)
    captures: Vec<HashSet<(Identifier, usize)>>,
    forward_declarations: Vec<HashMap<Identifier, bool>>,
    contexts: Vec<Context>,
}

impl Analyzer {
    pub fn new(namespaces: Rc<RefCell<NamespaceContext>>) -> Self {
        Self {
            namespaces,
            scopes: vec![],
            global_scope: GlobalScope::default(),
            captures: vec![],
            forward_declarations: vec![],
            contexts: vec![Context::Default],
        }
    }

    pub fn reset(&mut self) {
        debug_assert!(self.scopes.is_empty());
        debug_assert!(self.contexts.len() == 1);
        debug_assert!(self.captures.is_empty());
        debug_assert!(self.forward_declarations.is_empty());
        self.global_scope.clear();
    }

    fn current_lexical_frame(&self) -> usize {
        self.scopes.len() - 1
    }

    // (def! name value?)
    fn analyze_def(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        verify_arity(args, 1..3).map_err(DefError::from)?;

        let name = extract_symbol(&args[0]).map_err(DefError::from)?;

        // NOTE: insert before analyzing value in the event that it recursively captures `name`
        self.global_scope.insert(name.clone());

        let form = if args.len() == 2 {
            let analyzed_value = self.analyze(&args[1])?;
            DefForm::Bound(name, Box::new(analyzed_value))
        } else {
            DefForm::Unbound(name)
        };

        Ok(RuntimeValue::SpecialForm(SpecialForm::Def(form)))
    }

    // (var name)
    fn analyze_var(&self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        verify_arity(args, 1..2).map_err(VarError::from)?;

        let symbol = extract_symbol(&args[0])
            .map(|identifier| RuntimeValue::SpecialForm(SpecialForm::Var(identifier)))
            .map_err(VarError::from)?;

        Ok(symbol)
    }

    fn analyze_lexical_bindings<E>(
        &mut self,
        bindings_form: &[Form],
    ) -> AnalysisResult<Vec<LexicalBinding>>
    where
        E: From<LexicalError>,
        AnalysisError: From<E>,
    {
        let mut names = vec![];
        let mut values = vec![];
        for (name, value) in bindings_form.iter().tuples() {
            let analyzed_name =
                extract_symbol_without_namespace(name).map_err(|e| E::from(e.into()))?;
            let forward_visible = is_forward_visible(value);
            if forward_visible {
                let forward_decls = self.forward_declarations.last_mut().unwrap();
                forward_decls.insert(analyzed_name.clone(), false);
            }
            names.push((analyzed_name, forward_visible));
            values.push(value);
        }

        let mut bindings = vec![];
        for ((analyzed_name, forward_visible), value) in names.iter().zip(values.into_iter()) {
            let analyzed_value = self.analyze(value)?;
            bindings.push((analyzed_name.clone(), analyzed_value));
            if !forward_visible {
                let current_scope = self.scopes.last_mut().unwrap();
                current_scope.insert(analyzed_name.clone());
            }
        }
        Ok(bindings)
    }

    fn analyze_lexical_form<E>(&mut self, forms: &[Form]) -> AnalysisResult<LexicalForm>
    where
        E: From<LexicalError>,
        AnalysisError: From<E>,
    {
        let (bindings_form, body) = extract_lexical_form(forms).map_err(E::from)?;

        self.scopes.push(LexicalScope::default());
        self.forward_declarations.push(Default::default());

        let bindings = self
            .analyze_lexical_bindings::<E>(&bindings_form)
            .map_err(|err| {
                self.scopes.pop();
                self.forward_declarations.pop();
                err
            })?;

        let body = body
            .iter()
            .map(|form| self.analyze(form))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| {
                self.scopes.pop();
                self.forward_declarations.pop();
                err
            })?;

        self.scopes.pop();
        let possible_forward_declarations = self.forward_declarations.pop().unwrap();
        let mut forward_declarations = HashSet::new();
        for (name, &used) in &possible_forward_declarations {
            if used {
                forward_declarations.insert(index_for_name(&bindings, name));
            }
        }

        Ok(LexicalForm {
            bindings,
            body: BodyForm { body },
            forward_declarations,
        })
    }

    // (let* [bindings*] body*)
    fn analyze_let(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        let lexical_form = self.analyze_lexical_form::<LetError>(args)?;
        Ok(RuntimeValue::SpecialForm(SpecialForm::Let(lexical_form)))
    }

    // (loop* [bindings*] body*)
    fn analyze_loop(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        let lexical_form = self.analyze_lexical_form::<LoopError>(args)?;
        Ok(RuntimeValue::SpecialForm(SpecialForm::Loop(lexical_form)))
    }

    // (recur body*)
    fn analyze_recur(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        let body = args
            .iter()
            .map(|form| self.analyze(form))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RuntimeValue::SpecialForm(SpecialForm::Recur(BodyForm {
            body,
        })))
    }

    // (if predicate consequent alternate?)
    fn analyze_if(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        verify_arity(args, 2..4).map_err(IfError::from)?;

        let predicate = Box::new(self.analyze(&args[0])?);
        let consequent = Box::new(self.analyze(&args[1])?);

        let alternate = if args.len() == 3 {
            Some(Box::new(self.analyze(&args[2])?))
        } else {
            None
        };

        Ok(RuntimeValue::SpecialForm(SpecialForm::If(IfForm {
            predicate,
            consequent,
            alternate,
        })))
    }

    // (do body*)
    fn analyze_do(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        let body = args
            .iter()
            .map(|form| self.analyze(form))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RuntimeValue::SpecialForm(SpecialForm::Do(BodyForm {
            body,
        })))
    }

    fn analyze_fn_parameters(
        &mut self,
        bindings_form: &[Form],
    ) -> AnalysisResult<(Vec<Identifier>, Option<Identifier>)> {
        let mut parameters = bindings_form
            .iter()
            .map(|form| {
                extract_symbol_without_namespace(form).map(|s| {
                    let current_scope = self.scopes.last_mut().unwrap();
                    current_scope.insert(s.clone());
                    s
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| -> FnError { LexicalError::from(err).into() })?;

        match parameters.len() {
            0 => Ok((parameters, None)),
            1 => {
                if parameters[0] == VARIADIC_IDENTIFIER {
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
                    .all(|parameter| parameter != VARIADIC_IDENTIFIER);
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

    fn analyze_fn_body(&mut self, body_forms: &[Form]) -> AnalysisResult<BodyForm> {
        Ok(BodyForm {
            body: body_forms
                .iter()
                .map(|f| self.analyze(f))
                .collect::<Result<_, _>>()?,
        })
    }

    fn analyze_fn_captures(&mut self) -> Captures {
        let analyzed_captures = self.captures.pop().unwrap();
        let mut captures = Captures::default();
        let liveness_threshold = self.current_lexical_frame().saturating_sub(1);
        for (identifier, origin_frame) in analyzed_captures {
            let should_hoist = origin_frame < liveness_threshold;
            if should_hoist {
                let enclosing_captures = self.captures.last_mut().unwrap();
                enclosing_captures.insert((identifier.clone(), origin_frame));
            }
            captures.insert(identifier, Var::Unbound);
        }
        captures
    }

    // (fn* [parameters*] body*)
    fn analyze_fn(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        verify_arity(args, 1..MAX_ARGS_BOUND)
            .map_err(|err| -> FnError { LexicalError::from(err).into() })?;
        let bindings_form = extract_lexical_bindings(&args[0]).map_err(FnError::from)?;

        self.scopes.push(LexicalScope::default());

        let (parameters, variadic) = self.analyze_fn_parameters(&bindings_form).map_err(|err| {
            self.scopes.pop();
            err
        })?;

        self.captures.push(Default::default());
        let body = self.analyze_fn_body(&args[1..]).map_err(|err| {
            self.scopes.pop();
            self.captures.pop();
            err
        })?;

        let fn_form = FnForm {
            parameters,
            variadic,
            body,
        };

        let captures = self.analyze_fn_captures();

        let fn_impl = if captures.is_empty() {
            FnImpl::Default(fn_form)
        } else {
            FnImpl::WithCaptures(FnWithCapturesImpl {
                form: fn_form,
                captures,
            })
        };

        self.scopes.pop();

        Ok(RuntimeValue::SpecialForm(SpecialForm::Fn(fn_impl)))
    }

    // (quote form)
    fn analyze_quote(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        verify_arity(args, 1..2).map_err(QuoteError::from)?;

        self.contexts.push(Context::Quote);

        let analyzed_form = self.analyze(&args[0])?;

        self.contexts.pop();

        Ok(RuntimeValue::SpecialForm(SpecialForm::Quote(Box::new(
            analyzed_form,
        ))))
    }

    // (quasiquote form)
    fn analyze_quasiquote(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        verify_arity(args, 1..2).map_err(QuasiquoteError::from)?;

        self.contexts.push(Context::Quasiquote);

        let analyzed_form = self.analyze(&args[0])?;

        self.contexts.pop();

        Ok(RuntimeValue::SpecialForm(SpecialForm::Quasiquote(
            Box::new(analyzed_form),
        )))
    }

    // (unquote form)
    fn analyze_unquote(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        verify_quasiquoted_context(&self.contexts).map_err(UnquoteError::from)?;

        verify_arity(args, 1..2).map_err(UnquoteError::from)?;

        let form = self.analyze(&args[0])?;

        Ok(RuntimeValue::SpecialForm(SpecialForm::Unquote(Box::new(
            form,
        ))))
    }

    // (splice-unquote form)
    fn analyze_splice_unquote(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        verify_quasiquoted_context(&self.contexts).map_err(SpliceUnquoteError::from)?;

        verify_arity(args, 1..2).map_err(SpliceUnquoteError::from)?;

        let form = self.analyze(&args[0])?;

        Ok(RuntimeValue::SpecialForm(SpecialForm::SpliceUnquote(
            Box::new(form),
        )))
    }

    // (defmacro! symbol fn*-form)
    fn analyze_defmacro(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        verify_arity(args, 2..3).map_err(DefmacroError::from)?;

        let name = extract_symbol(&args[0]).map_err(DefmacroError::from)?;
        let body = match self.analyze(&args[1])? {
            RuntimeValue::SpecialForm(SpecialForm::Fn(fn_form)) => match fn_form {
                FnImpl::Default(fn_form) => fn_form,
                FnImpl::WithCaptures(..) => unimplemented!(
                    "macros cannot captures over their external environment currently"
                ),
            },
            _ => {
                return Err(DefmacroError::Type(TypeError {
                    expected: "fn* form".to_string(),
                    provided: args[1].clone(),
                })
                .into());
            }
        };

        Ok(RuntimeValue::SpecialForm(SpecialForm::Defmacro(name, body)))
    }

    // (macroexpand form)
    fn analyze_macroexpand(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        verify_arity(args, 1..2).map_err(MacroexpandError::from)?;

        let form = self.analyze(&args[0])?;

        Ok(RuntimeValue::SpecialForm(SpecialForm::Macroexpand(
            Box::new(form),
        )))
    }

    // (try* form* catch*-form?)
    fn analyze_try(&mut self, args: &[Form]) -> AnalysisResult<RuntimeValue> {
        let try_form = match args.len() {
            0 => TryForm::default(),
            _ => {
                let (last_form, body_forms) = args.split_last().expect("has at least 1 element");
                let mut body = body_forms
                    .iter()
                    .map(|form| self.analyze(form))
                    .collect::<Result<Vec<_>, _>>()?;
                let catch = match last_form {
                    Form::List(form) => match form.split_first() {
                        Some((first, rest)) => match first {
                            Form::Atom(Atom::Symbol(Symbol {
                                identifier,
                                namespace: None,
                            })) if identifier == CATCH_IDENTIFIER => {
                                let catch = self.analyze_catch(rest)?;
                                Some(catch)
                            }
                            _ => None,
                        },
                        None => None,
                    },
                    _ => None,
                };
                if catch.is_none() {
                    let analyzed_form = self.analyze(last_form)?;
                    body.push(analyzed_form);
                }

                TryForm {
                    body: BodyForm { body },
                    catch,
                }
            }
        };

        Ok(RuntimeValue::SpecialForm(SpecialForm::Try(try_form)))
    }

    // (catch* exc-symbol form*)
    fn analyze_catch(&mut self, args: &[Form]) -> AnalysisResult<CatchForm> {
        verify_arity(args, 1..MAX_ARGS_BOUND).map_err(CatchError::from)?;

        let exception_binding =
            extract_symbol_without_namespace(&args[0]).map_err(CatchError::from)?;

        let scope = HashSet::from([exception_binding.clone()]);
        self.scopes.push(scope);

        let body = args[1..]
            .iter()
            .map(|form| self.analyze(form))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| {
                self.scopes.pop();
                err
            })?;

        self.scopes.pop();

        Ok(CatchForm {
            exception_binding,
            body: BodyForm { body },
        })
    }

    fn analyze_list_with_possible_special_form(
        &mut self,
        operator: &Atom,
        operands: &[Form],
    ) -> AnalysisResult<RuntimeValue> {
        match operator {
            Atom::Symbol(
                symbol @ Symbol {
                    identifier: s,
                    namespace: None,
                },
            ) => match s.as_str() {
                "def!" => self.analyze_def(operands),
                "var" => self.analyze_var(operands),
                "let*" => self.analyze_let(operands),
                "loop*" => self.analyze_loop(operands),
                "recur" => self.analyze_recur(operands),
                "if" => self.analyze_if(operands),
                "do" => self.analyze_do(operands),
                "fn*" => self.analyze_fn(operands),
                "quote" => self.analyze_quote(operands),
                "quasiquote" => self.analyze_quasiquote(operands),
                "unquote" => self.analyze_unquote(operands),
                "splice-unquote" => self.analyze_splice_unquote(operands),
                "defmacro!" => self.analyze_defmacro(operands),
                "macroexpand" => self.analyze_macroexpand(operands),
                "try*" => self.analyze_try(operands),
                _ => {
                    let first = self.analyze_symbol(symbol)?;
                    self.analyze_list_without_special_form(first, operands)
                }
            },
            _ => unreachable!("only call this function with one of the prior variants"),
        }
    }

    fn analyze_list_without_special_form(
        &mut self,
        first: RuntimeValue,
        rest: &[Form],
    ) -> AnalysisResult<RuntimeValue> {
        // TODO evaluate in order to reveal errors left-to-right
        let mut inner = PersistentList::new();
        for form in rest.iter().rev() {
            let analyzed_form = self.analyze(form)?;
            inner.push_front_mut(analyzed_form);
        }
        inner.push_front_mut(first);
        Ok(RuntimeValue::List(inner))
    }

    fn analyze_list(&mut self, forms: &[Form]) -> AnalysisResult<RuntimeValue> {
        match forms.split_first() {
            Some((first, rest)) => match first {
                Form::Atom(
                    atom @ Atom::Symbol(Symbol {
                        namespace: None, ..
                    }),
                ) => self.analyze_list_with_possible_special_form(atom, rest),
                first => {
                    let first = self.analyze(first)?;
                    self.analyze_list_without_special_form(first, rest)
                }
            },
            None => Ok(RuntimeValue::List(PersistentList::new())),
        }
    }

    fn should_not_resolve_symbols(&self) -> bool {
        let current_context = self.contexts.last().expect("at least one global context");
        matches!(current_context, Context::Quote | Context::Quasiquote)
    }

    fn analyze_symbol(&mut self, symbol: &Symbol) -> AnalysisResult<RuntimeValue> {
        if self.should_not_resolve_symbols() {
            return Ok(RuntimeValue::Symbol(symbol.clone()));
        }

        if symbol.simple() {
            // TODO allow for nesting, i.e. not the `last` scope only?
            if let Some(forward_decls) = self.forward_declarations.last_mut() {
                if let Some(used) = forward_decls.get_mut(&symbol.identifier) {
                    *used = true;
                    return Ok(RuntimeValue::LexicalSymbol(symbol.identifier.clone()));
                }
            }

            if let Some(origin_frame) =
                lexical_frame_for_identifier(&self.scopes, &symbol.identifier)
            {
                let identifier = symbol.identifier.clone();

                if origin_frame < self.current_lexical_frame() {
                    let captures = self.captures.last_mut().unwrap();
                    captures.insert((identifier.clone(), origin_frame));
                }
                return Ok(RuntimeValue::LexicalSymbol(identifier));
            }
        }

        let result = self
            .namespaces
            .borrow()
            .resolve_symbol(symbol)
            .map(RuntimeValue::Var);

        match result {
            Ok(value) => Ok(value),
            Err(err) => match err {
                err @ NamespaceError::MissingIdentifier(..) => {
                    if self.global_scope.contains(&symbol) {
                        Ok(RuntimeValue::Symbol(symbol.clone()))
                    } else {
                        Err(SymbolError::from(err).into())
                    }
                }
                _ => Err(SymbolError::from(err).into()),
            },
        }
    }

    // `analyze` performs static analysis of `form` to provide data optimized for evaluation
    // 1. Syntax: can an `RuntimeValue` be produced from a `Form`
    // 2. Semantics: some `RuntimeValue`s have some invariants that can be verified statically
    //    like constraints on special forms
    pub fn analyze(&mut self, form: &Form) -> AnalysisResult<RuntimeValue> {
        let analyzed_form = match form {
            Form::Atom(Atom::Symbol(s)) => self.analyze_symbol(s)?,
            Form::Atom(a) => a.into(),
            Form::List(elems) => self.analyze_list(elems)?,
            Form::Vector(elems) => RuntimeValue::Vector(
                elems
                    .iter()
                    .map(|f| self.analyze(f))
                    .collect::<Result<_, _>>()?,
            ),
            Form::Map(elems) => RuntimeValue::Map(
                elems
                    .iter()
                    .map(|(x, y)| -> AnalysisResult<(RuntimeValue, RuntimeValue)> {
                        let analyzed_x = self.analyze(x)?;
                        let analyzed_y = self.analyze(y)?;
                        Ok((analyzed_x, analyzed_y))
                    })
                    .collect::<Result<_, _>>()?,
            ),
            Form::Set(elems) => RuntimeValue::Set(
                elems
                    .iter()
                    .map(|f| self.analyze(f))
                    .collect::<Result<_, _>>()?,
            ),
        };
        Ok(analyzed_form)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        namespace::Context as NamespaceContext,
        reader::read,
        value::{BodyForm, FnForm, FnImpl, FnWithCapturesImpl, LetForm, RuntimeValue, SpecialForm},
    };

    use super::*;

    #[test]
    fn test_analysis_no_captures() {
        let namespaces = Rc::new(RefCell::new(NamespaceContext::default()));
        let mut analyzer = Analyzer::new(namespaces);

        let source = "(let* [x 3] (fn* [y] y))";
        let form = read(source).expect("is valid source");
        let result = analyzer.analyze(&form[0]).expect("can be analyzed");

        let param = Identifier::from("y");
        let fn_form = RuntimeValue::SpecialForm(SpecialForm::Fn(FnImpl::Default(FnForm {
            parameters: vec![param.clone()],
            variadic: None,
            body: BodyForm {
                body: vec![RuntimeValue::LexicalSymbol(param)],
            },
        })));
        let expected_form = RuntimeValue::SpecialForm(SpecialForm::Let(LetForm {
            bindings: vec![(Identifier::from("x"), RuntimeValue::Number(3))],
            body: BodyForm {
                body: vec![fn_form],
            },
            forward_declarations: HashSet::new(),
        }));

        assert_eq!(result, expected_form);
    }

    #[test]
    fn test_analysis_with_captures() {
        let namespaces = Rc::new(RefCell::new(NamespaceContext::default()));
        let mut analyzer = Analyzer::new(namespaces);

        let source = "(let* [x 3] (fn* [] x))";
        let form = read(source).expect("is valid source");
        let result = analyzer.analyze(&form[0]).expect("can be analyzed");

        let param = Identifier::from("x");
        let fn_form =
            RuntimeValue::SpecialForm(SpecialForm::Fn(FnImpl::WithCaptures(FnWithCapturesImpl {
                form: FnForm {
                    parameters: vec![],
                    variadic: None,
                    body: BodyForm {
                        body: vec![RuntimeValue::LexicalSymbol(param.clone())],
                    },
                },
                captures: HashMap::from_iter([(param.clone(), Var::Unbound)]),
            })));
        let expected_form = RuntimeValue::SpecialForm(SpecialForm::Let(LetForm {
            bindings: vec![(Identifier::from("x"), RuntimeValue::Number(3))],
            body: BodyForm {
                body: vec![fn_form],
            },
            forward_declarations: HashSet::new(),
        }));

        assert_eq!(result, expected_form);
    }

    #[test]
    fn test_analysis_with_nested_captures() {
        let namespaces = Rc::new(RefCell::new(NamespaceContext::default()));
        let mut analyzer = Analyzer::new(namespaces);

        let source = "(let* [x 12] (fn* [] (let* [y 12] (fn* [a] (a x y)))))";
        let form = read(source).expect("is valid source");
        let result = analyzer.analyze(&form[0]).expect("can be analyzed");
        // let form  =SpecialForm(Let(LexicalForm { bindings: [("x", Number(12))], body: BodyForm { body: [SpecialForm(Fn(WithCaptures(FnWithCapturesImpl { form: FnForm { parameters: [], variadic: None, body: BodyForm { body: [SpecialForm(Let(LexicalForm { bindings: [("y", Number(12))], body: BodyForm { body: [SpecialForm(Fn(WithCaptures(FnWithCapturesImpl { form: FnForm { parameters: [], variadic: None, body: BodyForm { body: [List(List { head: Some(Node { value: LexicalSymbol("x"), next: Some(Node { value: LexicalSymbol("y"), next: None }) }), last: Some(LexicalSymbol("y")), length: 2 })] } }, captures: {"x": Unbound, "y": Unbound} })))] }, forward_declarations: {} }))] } }, captures: {"x": Unbound} })))] }, forward_declarations: {} }));

        let x = Identifier::from("x");
        let y = Identifier::from("y");
        let inner_fn_form =
            RuntimeValue::SpecialForm(SpecialForm::Fn(FnImpl::WithCaptures(FnWithCapturesImpl {
                form: FnForm {
                    parameters: vec![Identifier::from("a")],
                    variadic: None,
                    body: BodyForm {
                        body: vec![RuntimeValue::List(PersistentList::from_iter([
                            RuntimeValue::LexicalSymbol("a".into()),
                            RuntimeValue::LexicalSymbol("x".into()),
                            RuntimeValue::LexicalSymbol("y".into()),
                        ]))],
                    },
                },
                captures: HashMap::from_iter([
                    (x.clone(), Var::Unbound),
                    (y.clone(), Var::Unbound),
                ]),
            })));
        let inner_let_form = RuntimeValue::SpecialForm(SpecialForm::Let(LetForm {
            bindings: vec![(y.clone(), RuntimeValue::Number(12))],
            body: BodyForm {
                body: vec![inner_fn_form],
            },
            forward_declarations: Default::default(),
        }));
        let fn_form =
            RuntimeValue::SpecialForm(SpecialForm::Fn(FnImpl::WithCaptures(FnWithCapturesImpl {
                form: FnForm {
                    parameters: vec![],
                    variadic: None,
                    body: BodyForm {
                        body: vec![inner_let_form],
                    },
                },
                captures: HashMap::from_iter([(x.clone(), Var::Unbound)]),
            })));
        let expected_form = RuntimeValue::SpecialForm(SpecialForm::Let(LetForm {
            bindings: vec![(x, RuntimeValue::Number(12))],
            body: BodyForm {
                body: vec![fn_form],
            },
            forward_declarations: HashSet::new(),
        }));

        assert_eq!(result, expected_form);
    }
}
