use crate::prelude;
use crate::reader::{read, ReaderError};
use crate::value::{
    exception_from_thrown, exception_is_thrown, list_with_values, update_var, var_impl_into_inner,
    var_with_value, FnImpl, FnWithCapturesImpl, NativeFn, Value, VarImpl,
};
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use rpds::{
    HashTrieMap as PersistentMap, HashTrieSet as PersistentSet, List as PersistentList,
    Vector as PersistentVector,
};
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::From;
use std::default::Default;
use std::env::Args;
use std::fmt::Write;
use std::iter::FromIterator;
use std::time::SystemTimeError;
use thiserror::Error;

const COMMAND_LINE_ARGS_SYMBOL: &str = "*command-line-args*";
const DEFAULT_NAMESPACE: &str = "core";
const DEFAULT_CORE_FILENAME: &str = "src/core.sigil";
const SPECIAL_FORMS: &[&str] = &[
    "def!",           // (def! symbol form)
    "var",            // (var symbol)
    "let*",           // (let* [bindings*] form*)
    "loop*",          // (loop* [bindings*] form*)
    "recur",          // (recur form*)
    "if",             // (if predicate consequent alternate?)
    "do",             // (do form*)
    "fn*",            // (fn* [parameter*] form*)
    "quote",          // (quote form)
    "quasiquote",     // (quasiquote form)
    "unquote",        // (unquote form)
    "splice-unquote", // (splice-unquote form)
    "defmacro!",      // (defmacro! symbol fn*-form)
    "macroexpand",    // (macroexpand macro-form)
    "try*",           // (try* form* catch*-form?)
    "catch*",         // (catch* exc-symbol form*)
];

#[derive(Debug, Error)]
pub enum SymbolEvaluationError {
    #[error("var `{0}` not found in namespace `{1}`")]
    MissingVar(String, String),
    #[error("symbol {0} could not be resolved")]
    UnableToResolveSymbolToValue(String),
}

#[derive(Debug, Error)]
pub enum ListEvaluationError {
    #[error("cannot invoke the supplied value {0}")]
    CannotInvoke(Value),
    #[error("some failure: {0}")]
    Failure(String),
    #[error("error evaluating quasiquote: {0}")]
    QuasiquoteError(String),
    #[error("missing value for captured symbol {0}")]
    MissingCapturedValue(String),
}

#[derive(Debug, Error)]
pub enum PrimitiveEvaluationError {
    #[error("something failed {0}")]
    Failure(String),
}

#[derive(Debug, Error)]
pub enum InterpreterError {
    #[error("requested the {0}th command line arg but only {1} supplied")]
    MissingCommandLineArg(usize, usize),
    #[error("namespace {0} not found")]
    MissingNamespace(String),
    #[error("system time error: {0}")]
    SystemTimeError(#[from] SystemTimeError),
}

#[derive(Debug, Error)]
pub enum SyntaxError {
    #[error("bindings in `let` form must be pairs of names and values")]
    BindingsInLetMustBePaired(PersistentVector<Value>),
    #[error("expected further forms when parsing data")]
    MissingForms,
    #[error("expected vector of lexical bindings instead of {0}")]
    LexicalBindingsMustBeVector(Value),
    #[error("names in `let` form must not be namespaced like {0}")]
    NamesInLetMustNotBeNamespaced(Value),
    #[error("names in `let` form must be symbols unlike {0}")]
    NamesInLetMustBeSymbols(Value),
}

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("symbol error: {0}")]
    Symbol(SymbolEvaluationError),
    #[error("list error: {0}")]
    List(ListEvaluationError),
    #[error("syntax error: {0}")]
    Syntax(#[from] SyntaxError),
    #[error("primitive error: {0}")]
    Primitive(PrimitiveEvaluationError),
    #[error("reader error: {0}")]
    ReaderError(#[from] ReaderError),
    #[error("interpreter error: {0}")]
    Interpreter(#[from] InterpreterError),
}

pub type EvaluationResult<T> = Result<T, EvaluationError>;

type Scope = HashMap<String, Value>;

// map from identifier to Value::Var
type Namespace = HashMap<String, Value>;

fn lambda_parameter_key(index: usize, level: usize) -> String {
    let mut key = String::new();
    let _ = write!(&mut key, ":system-fn-%{}/{}", index, level);
    key
}

fn get_lambda_parameter_level(key: &str) -> Option<usize> {
    if key.starts_with(":system-fn-%") {
        return key
            .chars()
            .last()
            .and_then(|level| level.to_digit(10))
            .map(|level| level as usize);
    }
    None
}

// `scopes` from most specific to least specific
fn resolve_symbol_in_scopes<'a>(
    scopes: impl Iterator<Item = &'a Scope>,
    identifier: &str,
) -> Option<&'a Value> {
    for scope in scopes {
        if let Some(value) = scope.get(identifier) {
            return Some(value);
        }
    }
    None
}

fn get_var_in_namespace(var_desc: &str, namespace: &Namespace) -> Option<Value> {
    namespace.get(var_desc).map(|v| match v {
        Value::Var(inner) => Value::Var(inner.clone()),
        _ => unreachable!("only vars should be in namespaces"),
    })
}

fn intern_value_in_namespace(
    var_desc: &str,
    value: Value,
    namespace: &mut Namespace,
    namespace_desc: &str,
) -> Value {
    match namespace.get(var_desc) {
        // NOTE: must be Some(Value::Var)
        Some(var) => {
            update_var(var, value);
            var.clone()
        }
        None => {
            let var = var_with_value(value, namespace_desc, var_desc);
            namespace.insert(var_desc.to_string(), var.clone());
            var
        }
    }
}

fn unintern_value_in_namespace(identifier: &str, namespace: &mut Namespace) {
    let _ = namespace.remove(identifier);
}

fn eval_quasiquote_list_inner<'a>(
    elems: impl Iterator<Item = &'a Value>,
) -> EvaluationResult<Value> {
    let mut result = Value::List(PersistentList::new());
    for form in elems {
        match form {
            Value::List(inner) => {
                if let Some(first_inner) = inner.first() {
                    match first_inner {
                        Value::Symbol(s, None) if s == "splice-unquote" => {
                            if let Some(rest) = inner.drop_first() {
                                if let Some(second) = rest.first() {
                                    result = list_with_values(vec![
                                        Value::Symbol(
                                            "concat".to_string(),
                                            Some("core".to_string()),
                                        ),
                                        second.clone(),
                                        result,
                                    ]);
                                }
                            } else {
                                return Err(EvaluationError::List(
                                    ListEvaluationError::QuasiquoteError(
                                        "type error to `splice-unquote`".to_string(),
                                    ),
                                ));
                            }
                        }
                        _ => {
                            result = list_with_values(vec![
                                Value::Symbol("cons".to_string(), Some("core".to_string())),
                                eval_quasiquote(form)?,
                                result,
                            ]);
                        }
                    }
                } else {
                    result = list_with_values(vec![
                        Value::Symbol("cons".to_string(), Some("core".to_string())),
                        Value::List(PersistentList::new()),
                        result,
                    ]);
                }
            }
            form => {
                result = list_with_values(vec![
                    Value::Symbol("cons".to_string(), Some("core".to_string())),
                    eval_quasiquote(form)?,
                    result,
                ]);
            }
        }
    }
    Ok(result)
}

fn eval_quasiquote_list(elems: &PersistentList<Value>) -> EvaluationResult<Value> {
    if let Some(first) = elems.first() {
        match first {
            Value::Symbol(s, None) if s == "unquote" => {
                if let Some(rest) = elems.drop_first() {
                    if let Some(argument) = rest.first() {
                        return Ok(argument.clone());
                    }
                }
                return Err(EvaluationError::List(ListEvaluationError::QuasiquoteError(
                    "type error to `unquote`".to_string(),
                )));
            }
            _ => return eval_quasiquote_list_inner(elems.reverse().iter()),
        }
    }
    Ok(Value::List(PersistentList::new()))
}

fn eval_quasiquote_vector(elems: &PersistentVector<Value>) -> EvaluationResult<Value> {
    Ok(list_with_values(vec![
        Value::Symbol("vec".to_string(), Some("core".to_string())),
        eval_quasiquote_list_inner(elems.iter().rev())?,
    ]))
}

fn eval_quasiquote(value: &Value) -> EvaluationResult<Value> {
    match value {
        Value::List(elems) => eval_quasiquote_list(elems),
        Value::Vector(elems) => eval_quasiquote_vector(elems),
        elem @ Value::Map(_) | elem @ Value::Symbol(..) => {
            let args = vec![Value::Symbol("quote".to_string(), None), elem.clone()];
            Ok(list_with_values(args.into_iter()))
        }
        v => Ok(v.clone()),
    }
}

struct LetForm<'a> {
    bindings: Vec<(&'a String, &'a Value)>,
    body: PersistentList<Value>,
}

fn parse_let_bindings(bindings_form: &Value) -> EvaluationResult<Vec<(&String, &Value)>> {
    match bindings_form {
        Value::Vector(bindings) => {
            let bindings_count = bindings.len();
            if bindings_count % 2 == 0 {
                let mut validated_bindings = Vec::with_capacity(bindings_count);
                for (name, value_form) in bindings.iter().tuples() {
                    match name {
                        Value::Symbol(s, None) => {
                            validated_bindings.push((s, value_form));
                        }
                        s @ Value::Symbol(_, Some(_)) => {
                            return Err(SyntaxError::NamesInLetMustNotBeNamespaced(s.clone()).into())
                        }
                        other => {
                            return Err(SyntaxError::NamesInLetMustBeSymbols(other.clone()).into());
                        }
                    }
                }
                Ok(validated_bindings)
            } else {
                return Err(SyntaxError::BindingsInLetMustBePaired(bindings.clone()).into());
            }
        }
        other => return Err(SyntaxError::LexicalBindingsMustBeVector(other.clone()).into()),
    }
}

fn parse_let(forms: &PersistentList<Value>) -> EvaluationResult<LetForm> {
    let bindings_form = forms
        .first()
        .ok_or_else(|| -> EvaluationError { SyntaxError::MissingForms.into() })?;
    let body = forms
        .drop_first()
        .ok_or_else(|| -> EvaluationError { SyntaxError::MissingForms.into() })?;
    let bindings = parse_let_bindings(bindings_form)?;
    Ok(LetForm { bindings, body })
}

fn analyze_let(let_forms: &PersistentList<Value>) -> EvaluationResult<LetForm> {
    let let_form = parse_let(&let_forms)?;
    Ok(let_form)
}

#[derive(Debug)]
pub struct Interpreter {
    current_namespace: String,
    namespaces: HashMap<String, Namespace>,

    // stack of scopes
    // contains at least one scope, the "default" scope
    scopes: Vec<Scope>,
}

impl Default for Interpreter {
    fn default() -> Self {
        // build the "core" namespace
        let mut default_namespace = Namespace::default();
        for (symbol, value) in prelude::BINDINGS {
            intern_value_in_namespace(
                symbol,
                value.clone(),
                &mut default_namespace,
                DEFAULT_NAMESPACE,
            );
        }

        let mut default_namespaces = HashMap::new();
        default_namespaces.insert(DEFAULT_NAMESPACE.to_string(), default_namespace);

        // build the default scope, which resolves special forms to themselves
        // so that they fall through to the interpreter's evaluation
        let mut default_scope = Scope::new();
        for form in SPECIAL_FORMS {
            default_scope.insert(form.to_string(), Value::Symbol(form.to_string(), None));
        }

        let mut interpreter = Interpreter {
            current_namespace: DEFAULT_NAMESPACE.to_string(),
            namespaces: default_namespaces,
            scopes: vec![default_scope],
        };

        // load the "prelude" source
        for line in prelude::SOURCE.lines() {
            let forms = read(line).expect("prelude source has no reader errors");
            for form in forms.iter() {
                let _ = interpreter
                    .evaluate(form)
                    .expect("prelude forms have no evaluation errors");
            }
        }

        let mut core_boot_form_source = String::from("(load-file \"");
        core_boot_form_source += DEFAULT_CORE_FILENAME;
        core_boot_form_source += "\")";
        let core_boot_result = read(&core_boot_form_source).expect("core boot is well-formed");
        let core_boot_form = core_boot_result.get(0).expect("core boot is well-formed");
        let _ = interpreter
            .evaluate(&core_boot_form)
            .expect("core boot is well-formed");

        let mut command_line_args_form_source = String::from("(def! ");
        command_line_args_form_source += COMMAND_LINE_ARGS_SYMBOL;
        command_line_args_form_source += " '())";
        let command_line_args_def_result =
            read(&command_line_args_form_source).expect("environment boot is well-formed");
        let command_line_args_def_form = command_line_args_def_result
            .get(0)
            .expect("environment boot is well-formed");
        let _ = interpreter
            .evaluate(&command_line_args_def_form)
            .expect("environment boot is well-formed");

        interpreter
    }
}

impl Interpreter {
    /// Store `args` in the var referenced by `COMMAND_LINE_ARGS_SYMBOL`.
    pub fn intern_args(&mut self, args: Args) {
        let form = args.map(Value::String).collect();
        self.intern_var(COMMAND_LINE_ARGS_SYMBOL, Value::List(form));
    }

    /// Read the interned command line argument at position `n` in the collection.
    pub fn command_line_arg(&mut self, n: usize) -> EvaluationResult<String> {
        match self.resolve_symbol(COMMAND_LINE_ARGS_SYMBOL, None)? {
            Value::List(args) => match args.iter().nth(n) {
                Some(value) => match value {
                    Value::String(arg) => Ok(arg.clone()),
                    _ => unreachable!(),
                },
                None => Err(EvaluationError::Interpreter(
                    InterpreterError::MissingCommandLineArg(n, args.len()),
                )),
            },
            _ => panic!("error to not intern command line args as a list"),
        }
    }

    pub fn current_namespace(&self) -> &str {
        &self.current_namespace
    }

    fn intern_var(&mut self, identifier: &str, value: Value) -> Value {
        let current_namespace = self.current_namespace().to_string();

        let ns = self
            .namespaces
            .get_mut(&current_namespace)
            .expect("current namespace always resolves");
        intern_value_in_namespace(identifier, value, ns, &current_namespace)
    }

    fn unintern_var(&mut self, identifier: &str) {
        let current_namespace = self.current_namespace().to_string();

        let ns = self
            .namespaces
            .get_mut(&current_namespace)
            .expect("current namespace always resolves");
        unintern_value_in_namespace(identifier, ns);
    }

    // return a ref to some var in the current namespace
    fn resolve_var_in_current_namespace(&self, identifier: &str) -> EvaluationResult<Value> {
        let ns_desc = self.current_namespace();
        self.resolve_var_in_namespace(identifier, ns_desc)
    }

    // namespace -> var
    fn resolve_var_in_namespace(&self, identifier: &str, ns_desc: &str) -> EvaluationResult<Value> {
        self.namespaces
            .get(ns_desc)
            .ok_or_else(|| {
                EvaluationError::Interpreter(InterpreterError::MissingNamespace(
                    ns_desc.to_string(),
                ))
            })
            .and_then(|ns| {
                get_var_in_namespace(identifier, ns).ok_or_else(|| {
                    EvaluationError::Symbol(SymbolEvaluationError::MissingVar(
                        identifier.to_string(),
                        ns_desc.to_string(),
                    ))
                })
            })
    }

    // symbol -> namespace -> var
    fn resolve_symbol_to_var(
        &self,
        identifier: &str,
        ns_opt: Option<&String>,
    ) -> EvaluationResult<Value> {
        // if namespaced, check there
        if let Some(ns_desc) = ns_opt {
            return self.resolve_var_in_namespace(identifier, ns_desc);
        }
        // else resolve in lexical scopes
        if let Some(value) = resolve_symbol_in_scopes(self.scopes.iter().rev(), identifier) {
            return Ok(value.clone());
        }
        // otherwise check current namespace
        self.resolve_var_in_current_namespace(identifier)
    }

    // symbol -> namespace -> var -> value
    fn resolve_symbol(&self, identifier: &str, ns_opt: Option<&String>) -> EvaluationResult<Value> {
        match self.resolve_symbol_to_var(identifier, ns_opt)? {
            Value::Var(v) => Ok(var_impl_into_inner(&v)),
            other => Ok(other),
        }
    }

    fn enter_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    fn insert_value_in_current_scope(&mut self, identifier: &str, value: Value) {
        let scope = self.scopes.last_mut().expect("always one scope");
        scope.insert(identifier.to_string(), value);
    }

    fn leave_scope(&mut self) {
        let _ = self.scopes.pop().expect("no underflow in scope stack");
    }

    // Analyze symbols (recursively) in `form`:
    // 1. Rewrite lambda parameters
    // 2. Capture references to external vars
    fn analyze_form_in_lambda(
        &mut self,
        form: &Value,
        scopes: &mut Vec<Scope>,
        captures: &mut Vec<HashSet<String>>,
        // `level` helps implement lifetime analysis for captured parameters
        level: usize,
    ) -> EvaluationResult<Value> {
        match form {
            Value::Symbol(identifier, ns_opt) => {
                if let Some(value) = resolve_symbol_in_scopes(scopes.iter().rev(), identifier) {
                    if let Value::Symbol(resolved_identifier, None) = value {
                        if let Some(requested_level) =
                            get_lambda_parameter_level(resolved_identifier)
                        {
                            if requested_level < level {
                                let captures_at_level = captures
                                    .last_mut()
                                    .expect("already pushed scope for captures");
                                // TODO: work through lifetimes here to avoid cloning...
                                captures_at_level.insert(resolved_identifier.to_string());
                            }
                        }
                    }
                    return Ok(value.clone());
                }
                self.resolve_symbol_to_var(identifier, ns_opt.as_ref())
            }
            Value::List(elems) => {
                // if first elem introduces a new lexical scope...
                let mut iter = elems.iter();
                let scopes_len = scopes.len();
                let mut analyzed_elems = vec![];
                match iter.next() {
                    Some(Value::Symbol(s, None)) if s == "let*" || s == "loop*" => {
                        analyzed_elems.push(Value::Symbol(s.to_string(), None));
                        if let Some(Value::Vector(bindings)) = iter.next() {
                            let mut scope = Scope::new();
                            if bindings.len() % 2 != 0 {
                                return Err(EvaluationError::List(ListEvaluationError::Failure(
                                    "could not evaluate `let*`".to_string(),
                                )));
                            }
                            let mut analyzed_bindings = PersistentVector::new();
                            for (name, value) in bindings.iter().tuples() {
                                scope.insert(name.to_string(), name.clone());
                                let analyzed_value =
                                    self.analyze_form_in_lambda(value, scopes, captures, level)?;
                                analyzed_bindings.push_back_mut(name.clone());
                                analyzed_bindings.push_back_mut(analyzed_value);
                            }
                            analyzed_elems.push(Value::Vector(analyzed_bindings));
                            scopes.push(scope);
                        }
                    }
                    Some(Value::Symbol(s, None)) if s == "fn*" => {
                        if let Some(Value::Vector(bindings)) = iter.next() {
                            let rest = iter.cloned().collect();
                            // Note: can only have captures over enclosing fns if we have recursive nesting of fns
                            let current_fn_level = captures.len();
                            captures.push(HashSet::new());
                            let analyzed_fn =
                                self.analyze_symbols_in_lambda(rest, bindings, scopes, captures)?;
                            let captures_at_this_level = captures.pop().expect("did push");
                            if !captures_at_this_level.is_empty() {
                                if let Value::Fn(f) = analyzed_fn {
                                    // Note: need to hoist captures if there are intervening functions along the way...
                                    for capture in &captures_at_this_level {
                                        if let Some(level) = get_lambda_parameter_level(&capture) {
                                            if level < current_fn_level {
                                                let captures_at_hoisted_level = captures
                                                    .get_mut(level)
                                                    .expect("already pushed scope");
                                                captures_at_hoisted_level
                                                    .insert(capture.to_string());
                                            }
                                        }
                                    }
                                    let captures = captures_at_this_level
                                        .iter()
                                        .map(|capture| (capture.to_string(), None))
                                        .collect();
                                    return Ok(Value::FnWithCaptures(FnWithCapturesImpl {
                                        f,
                                        captures,
                                    }));
                                }
                            }
                            return Ok(analyzed_fn);
                        }
                    }
                    Some(Value::Symbol(s, None)) if s == "catch*" => {
                        if let Some(Value::Symbol(s, None)) = iter.next() {
                            // Note: to allow for captures inside `catch*`,
                            // treat the form as a lambda of one parameter
                            let mut bindings = PersistentVector::new();
                            bindings.push_back_mut(Value::Symbol(s.clone(), None));

                            let rest = iter.cloned().collect();
                            // Note: can only have captures over enclosing fns if we have recursive nesting of fns
                            let current_fn_level = captures.len();
                            captures.push(HashSet::new());
                            let analyzed_fn =
                                self.analyze_symbols_in_lambda(rest, &bindings, scopes, captures)?;
                            let captures_at_this_level = captures.pop().expect("did push");
                            if !captures_at_this_level.is_empty() {
                                if let Value::Fn(f) = analyzed_fn {
                                    // Note: need to hoist captures if there are intervening functions along the way...
                                    for capture in &captures_at_this_level {
                                        if let Some(level) = get_lambda_parameter_level(&capture) {
                                            if level < current_fn_level {
                                                let captures_at_hoisted_level = captures
                                                    .get_mut(level)
                                                    .expect("already pushed scope");
                                                captures_at_hoisted_level
                                                    .insert(capture.to_string());
                                            }
                                        }
                                    }
                                    let captures = captures_at_this_level
                                        .iter()
                                        .map(|capture| (capture.to_string(), None))
                                        .collect();
                                    return Ok(Value::FnWithCaptures(FnWithCapturesImpl {
                                        f,
                                        captures,
                                    }));
                                }
                            }
                            return Ok(analyzed_fn);
                        }
                    }
                    Some(Value::Symbol(s, None)) if s == "quote" => {
                        if let Some(Value::Symbol(s, None)) = iter.next() {
                            let mut scope = Scope::new();
                            scope.insert(s.to_string(), Value::Symbol(s.to_string(), None));
                            scopes.push(scope);
                        }
                    }
                    _ => {}
                }
                for elem in elems.iter().skip(analyzed_elems.len()) {
                    let analyzed_elem =
                        self.analyze_form_in_lambda(elem, scopes, captures, level)?;
                    analyzed_elems.push(analyzed_elem);
                }
                if scopes_len != scopes.len() {
                    let _ = scopes
                        .pop()
                        .expect("only pop if we pushed local to this function");
                }
                Ok(Value::List(PersistentList::from_iter(analyzed_elems)))
            }
            Value::Vector(elems) => {
                let mut analyzed_elems = PersistentVector::new();
                for elem in elems.iter() {
                    let analyzed_elem =
                        self.analyze_form_in_lambda(elem, scopes, captures, level)?;
                    analyzed_elems.push_back_mut(analyzed_elem);
                }
                Ok(Value::Vector(analyzed_elems))
            }
            Value::Map(elems) => {
                let mut analyzed_elems = PersistentMap::new();
                for (k, v) in elems.iter() {
                    let analyzed_k = self.analyze_form_in_lambda(k, scopes, captures, level)?;
                    let analyzed_v = self.analyze_form_in_lambda(v, scopes, captures, level)?;
                    analyzed_elems.insert_mut(analyzed_k, analyzed_v);
                }
                Ok(Value::Map(analyzed_elems))
            }
            Value::Set(elems) => {
                let mut analyzed_elems = PersistentSet::new();
                for elem in elems.iter() {
                    let analyzed_elem =
                        self.analyze_form_in_lambda(elem, scopes, captures, level)?;
                    analyzed_elems.insert_mut(analyzed_elem);
                }
                Ok(Value::Set(analyzed_elems))
            }
            Value::Fn(_) => unreachable!(),
            Value::FnWithCaptures(_) => unreachable!(),
            Value::Primitive(_) => unreachable!(),
            Value::Recur(_) => unreachable!(),
            Value::Macro(_) => unreachable!(),
            Value::Exception(_) => unreachable!(),
            // Nil, Bool, Number, String, Keyword, Var, Atom
            other => Ok(other.clone()),
        }
    }

    // Non-local symbols should:
    // 1. resolve to a parameter
    // 2. resolve to a value in the enclosing environment, which is captured
    // otherwise, the lambda is an error
    //
    // Note: parameters are resolved to (ordinal) reserved symbols
    fn analyze_symbols_in_lambda(
        &mut self,
        body: PersistentList<Value>,
        params: &PersistentVector<Value>,
        lambda_scopes: &mut Vec<Scope>,
        // record any values captured from the environment that would outlive the lifetime of this particular lambda
        captures: &mut Vec<HashSet<String>>,
    ) -> EvaluationResult<Value> {
        // level of lambda nesting
        let level = lambda_scopes.len();
        // build parameter index
        let mut variadic = false;
        let mut parameters = Scope::new();
        let param_count = params.len();
        let mut iter = params.iter().enumerate();
        while let Some((index, param)) = iter.next() {
            if param_count >= 2 && index == param_count - 2 {
                match param {
                    Value::Symbol(s, None) if s == "&" => {
                        if let Some((index, last_symbol)) = iter.next() {
                            match last_symbol {
                                Value::Symbol(s, None) => {
                                    variadic = true;
                                    let parameter = lambda_parameter_key(index - 1, level);
                                    parameters
                                        .insert(s.to_string(), Value::Symbol(parameter, None));
                                    break;
                                }
                                _ => {
                                    return Err(EvaluationError::List(ListEvaluationError::Failure("could not evaluate `fn*`: variadic binding must be a symbol".to_string())));
                                }
                            }
                        }
                        return Err(EvaluationError::List(ListEvaluationError::Failure(
                            "could not evaluate `fn*`: variadic binding missing after `&`"
                                .to_string(),
                        )));
                    }
                    _ => {}
                }
            }
            match param {
                Value::Symbol(s, None) => {
                    let parameter = lambda_parameter_key(index, level);
                    parameters.insert(s.to_string(), Value::Symbol(parameter, None));
                }
                _ => {
                    return Err(EvaluationError::List(ListEvaluationError::Failure(
                        "could not evaluate `fn*`: parameters must be non-namespaced symbols"
                            .to_string(),
                    )));
                }
            }
        }
        let arity = if variadic {
            parameters.len() - 1
        } else {
            parameters.len()
        };
        // walk the `body`, resolving symbols where possible...
        lambda_scopes.push(parameters);
        let mut analyzed_body = Vec::with_capacity(body.len());
        for form in body.iter() {
            let analyzed_form =
                self.analyze_form_in_lambda(form, lambda_scopes, captures, level)?;
            analyzed_body.push(analyzed_form);
        }
        lambda_scopes.pop();
        Ok(Value::Fn(FnImpl {
            body: analyzed_body.into_iter().collect(),
            arity,
            level,
            variadic,
        }))
    }

    fn apply_macro(
        &mut self,
        FnImpl {
            body,
            arity,
            level,
            variadic,
        }: &FnImpl,
        forms: &PersistentList<Value>,
    ) -> EvaluationResult<Value> {
        if let Some(forms) = forms.drop_first() {
            let result = match self.apply_fn_inner(body, *arity, *level, *variadic, forms, false) {
                Ok(Value::List(forms)) => self.macroexpand(&forms),
                Ok(result) => Ok(result),
                Err(e) => {
                    let mut err = String::from("could not apply macro: ");
                    err += &e.to_string();
                    Err(EvaluationError::List(ListEvaluationError::Failure(err)))
                }
            };
            return result;
        }
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not apply macro".to_string(),
        )));
    }

    fn macroexpand(&mut self, forms: &PersistentList<Value>) -> EvaluationResult<Value> {
        match forms.first() {
            Some(Value::Symbol(identifier, ns_opt)) => {
                if let Ok(Value::Macro(f)) = self.resolve_symbol(identifier, ns_opt.as_ref()) {
                    return self.apply_macro(&f, forms);
                }
            }
            Some(Value::Var(v)) => {
                if let Value::Macro(f) = var_impl_into_inner(v) {
                    return self.apply_macro(&f, forms);
                }
            }
            _ => {}
        }
        Ok(Value::List(forms.clone()))
    }

    /// Apply the given `Fn` to the supplied `args`.
    /// Exposed for various `prelude` functions.
    pub fn apply_fn_inner(
        &mut self,
        body: &PersistentList<Value>,
        arity: usize,
        level: usize,
        variadic: bool,
        args: PersistentList<Value>,
        should_evaluate: bool,
    ) -> EvaluationResult<Value> {
        let correct_arity = if variadic {
            args.len() >= arity
        } else {
            args.len() == arity
        };
        if !correct_arity {
            return Err(EvaluationError::List(ListEvaluationError::Failure(
                "could not apply `fn*`: incorrect arity".to_string(),
            )));
        }
        self.enter_scope();
        let mut iter = args.iter().enumerate();
        if arity > 0 {
            while let Some((index, operand_form)) = iter.next() {
                let operand = if should_evaluate {
                    match self.evaluate(operand_form) {
                        Ok(operand) => operand,
                        Err(e) => {
                            self.leave_scope();
                            let mut error = String::from("could not apply `fn*`: ");
                            error += &e.to_string();
                            return Err(EvaluationError::List(ListEvaluationError::Failure(error)));
                        }
                    }
                } else {
                    operand_form.clone()
                };
                let parameter = lambda_parameter_key(index, level);
                self.insert_value_in_current_scope(&parameter, operand);

                if index == arity - 1 {
                    break;
                }
            }
        }
        if variadic {
            let mut variadic_args = vec![];
            for (_, elem_form) in iter {
                let elem = if should_evaluate {
                    match self.evaluate(elem_form) {
                        Ok(elem) => elem,
                        Err(e) => {
                            self.leave_scope();
                            let mut error = String::from("could not apply `fn*`: ");
                            error += &e.to_string();
                            return Err(EvaluationError::List(ListEvaluationError::Failure(error)));
                        }
                    }
                } else {
                    elem_form.clone()
                };
                variadic_args.push(elem);
            }
            let operand = Value::List(variadic_args.into_iter().collect());
            let parameter = lambda_parameter_key(arity, level);
            self.insert_value_in_current_scope(&parameter, operand);
        }
        let mut result = self.eval_do_inner(body);
        if let Ok(Value::FnWithCaptures(FnWithCapturesImpl { f, mut captures })) = result {
            for (capture, value) in &mut captures {
                let captured_value = resolve_symbol_in_scopes(self.scopes.iter().rev(), capture)
                    .ok_or_else(|| {
                        EvaluationError::Symbol(
                            SymbolEvaluationError::UnableToResolveSymbolToValue(
                                capture.to_string(),
                            ),
                        )
                    })?;
                *value = Some(captured_value.clone());
            }
            result = Ok(Value::FnWithCaptures(FnWithCapturesImpl { f, captures }))
        }
        self.leave_scope();
        result
    }

    fn apply_fn(
        &mut self,
        FnImpl {
            body,
            arity,
            level,
            variadic,
        }: &FnImpl,
        args: &PersistentList<Value>,
    ) -> EvaluationResult<Value> {
        if let Some(args) = args.drop_first() {
            return self.apply_fn_inner(body, *arity, *level, *variadic, args, true);
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not apply `fn*`".to_string(),
        )))
    }

    fn apply_primitive(
        &mut self,
        native_fn: &NativeFn,
        args: &PersistentList<Value>,
    ) -> EvaluationResult<Value> {
        let mut operands = vec![];
        if let Some(rest) = args.drop_first() {
            for operand_form in rest.iter() {
                let operand = self.evaluate(operand_form)?;
                operands.push(operand);
            }
        }
        native_fn(self, &operands)
    }

    fn eval_def(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        if let Some(rest) = forms.drop_first() {
            if let Some(Value::Symbol(id, None)) = rest.first() {
                if let Some(rest) = rest.drop_first() {
                    if let Some(value_form) = rest.first() {
                        // need to only adjust var if this `def!` is successful
                        // also optimistically allocate in the interpreter so that
                        // the def body can capture references to itself (e.g. for recursive fn)
                        //
                        // to address this:
                        // get the existing var, or intern a sentinel value if it is missing
                        let (var, var_already_exists) =
                            match self.resolve_var_in_current_namespace(id) {
                                Ok(v @ Value::Var(..)) => (v, true),
                                Err(EvaluationError::Symbol(
                                    SymbolEvaluationError::MissingVar(..),
                                )) => (self.intern_var(id, Value::Nil), false),
                                e @ Err(_) => return e,
                                _ => unreachable!(),
                            };
                        match self.evaluate(value_form) {
                            Ok(value) => {
                                // and if the evaluation is ok, unconditionally update the var
                                update_var(&var, value);
                                return Ok(var);
                            }
                            Err(e) => {
                                // and if the evaluation is not ok,
                                if !var_already_exists {
                                    // and the var did not already exist, unintern the sentinel allocation
                                    self.unintern_var(id);
                                }
                                // (if the var did already exist, then simply leave alone)

                                let mut error = String::from("could not evaluate `def!`: ");
                                error += &e.to_string();
                                return Err(EvaluationError::List(ListEvaluationError::Failure(
                                    error,
                                )));
                            }
                        }
                    }
                }
            }
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not evaluate `def!`".to_string(),
        )))
    }

    fn eval_var(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        if let Some(rest) = forms.drop_first() {
            if let Some(Value::Symbol(s, ns_opt)) = rest.first() {
                if let Some(ns_desc) = ns_opt {
                    return self.resolve_var_in_namespace(s, ns_desc);
                } else {
                    return self.resolve_var_in_current_namespace(s);
                }
            }
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not evaluate `var`".to_string(),
        )))
    }

    fn eval_let(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        let let_forms = forms
            .drop_first()
            .ok_or_else(|| -> EvaluationError { SyntaxError::MissingForms.into() })?;
        let LetForm { bindings, body } = analyze_let(&let_forms)?;
        self.enter_scope();
        for (name, value_form) in bindings {
            match self.evaluate(value_form) {
                Ok(value) => self.insert_value_in_current_scope(name, value),
                e @ Err(_) => {
                    self.leave_scope();
                    return e;
                }
            }
        }
        let result = self.eval_do_inner(&body);
        self.leave_scope();
        return result;
    }

    fn eval_loop(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        if let Some(rest) = forms.drop_first() {
            if let Some(Value::Vector(elems)) = rest.first() {
                if elems.len() % 2 == 0 {
                    if let Some(body) = rest.drop_first() {
                        if body.is_empty() {
                            return Ok(Value::Nil);
                        }
                        // TODO: analyze loop*
                        // if recur, must be in tail position
                        self.enter_scope();
                        let mut bindings_keys = vec![];
                        for (name, value_form) in elems.iter().tuples() {
                            match name {
                                Value::Symbol(s, None) => {
                                    let value = self.evaluate(value_form)?;
                                    bindings_keys.push(s.clone());
                                    self.insert_value_in_current_scope(s, value)
                                }
                                _ => {
                                    self.leave_scope();
                                    return Err(EvaluationError::List(
                                        ListEvaluationError::Failure(
                                            "could not evaluate `loop*`".to_string(),
                                        ),
                                    ));
                                }
                            }
                        }
                        let mut result = self.eval_do_inner(&body);
                        while let Ok(Value::Recur(next_bindings)) = result {
                            if next_bindings.len() != bindings_keys.len() {
                                self.leave_scope();
                                return Err(EvaluationError::List(
                                                                    ListEvaluationError::Failure(
                                                                        "could not evaluate `loop*`: recur with incorrect number of bindings"
                                                                            .to_string(),
                                                                    ),
                                                                ));
                            }
                            for (key, value) in bindings_keys.iter().zip(next_bindings.iter()) {
                                self.insert_value_in_current_scope(key, value.clone());
                            }
                            result = self.eval_do_inner(&body);
                        }
                        self.leave_scope();
                        return result;
                    }
                }
            }
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not evaluate `loop*`".to_string(),
        )))
    }

    fn eval_recur(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        if let Some(rest) = forms.drop_first() {
            let mut result = vec![];
            for form in rest.into_iter() {
                let value = self.evaluate(form)?;
                result.push(value);
            }
            return Ok(Value::Recur(result.into_iter().collect()));
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not evaluate `recur`".to_string(),
        )))
    }

    fn eval_if(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        if let Some(rest) = forms.drop_first() {
            if let Some(predicate_form) = rest.first() {
                if let Some(rest) = rest.drop_first() {
                    if let Some(consequent_form) = rest.first() {
                        match self.evaluate(predicate_form)? {
                            Value::Bool(predicate) => {
                                if predicate {
                                    return self.evaluate(consequent_form);
                                } else if let Some(rest) = rest.drop_first() {
                                    if let Some(alternate) = rest.first() {
                                        return self.evaluate(alternate);
                                    } else {
                                        return Ok(Value::Nil);
                                    }
                                }
                            }
                            Value::Nil => {
                                if let Some(rest) = rest.drop_first() {
                                    if let Some(alternate) = rest.first() {
                                        return self.evaluate(alternate);
                                    } else {
                                        return Ok(Value::Nil);
                                    }
                                }
                            }
                            _ => return self.evaluate(consequent_form),
                        }
                    }
                }
            }
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not evaluate `if`".to_string(),
        )))
    }

    fn eval_do_inner(&mut self, forms: &PersistentList<Value>) -> EvaluationResult<Value> {
        forms
            .iter()
            .fold_while(Ok(Value::Nil), |_, next| match self.evaluate(next) {
                Ok(e @ Value::Exception(_)) if exception_is_thrown(&e) => Done(Ok(e)),
                e @ Err(_) => Done(e),
                value => Continue(value),
            })
            .into_inner()
    }

    fn eval_do(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        if let Some(rest) = forms.drop_first() {
            return self.eval_do_inner(&rest);
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not evaluate `do`".to_string(),
        )))
    }

    fn eval_fn(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        if let Some(rest) = forms.drop_first() {
            if let Some(Value::Vector(params)) = rest.first() {
                if let Some(body) = rest.drop_first() {
                    let mut scopes = vec![];
                    let mut captures = vec![];
                    return self.analyze_symbols_in_lambda(
                        body,
                        &params,
                        &mut scopes,
                        &mut captures,
                    );
                }
            }
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not evaluate `fn*`".to_string(),
        )))
    }

    fn eval_quote(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        if let Some(rest) = forms.drop_first() {
            if rest.len() == 1 {
                if let Some(form) = rest.first() {
                    return Ok(form.clone());
                }
            }
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not evaluate `quote`".to_string(),
        )))
    }

    fn eval_quasiquote(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        if let Some(rest) = forms.drop_first() {
            if let Some(second) = rest.first() {
                let expansion = eval_quasiquote(second)?;
                return self.evaluate(&expansion);
            }
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not evaluate `quasiquote`".to_string(),
        )))
    }

    fn eval_defmacro(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        match self.eval_def(forms) {
            Ok(Value::Var(v @ VarImpl { .. })) => match var_impl_into_inner(&v) {
                Value::Fn(f) => {
                    let var = Value::Var(v);
                    update_var(&var, Value::Macro(f));
                    Ok(var)
                }
                _ => {
                    self.unintern_var(&v.identifier);
                    let error = String::from(
                        "could not evaluate `defmacro!`: body must be `fn*` without captures",
                    );
                    Err(EvaluationError::List(ListEvaluationError::Failure(error)))
                }
            },
            Err(e) => {
                let mut error = String::from("could not evaluate `defmacro!`: ");
                error += &e.to_string();
                Err(EvaluationError::List(ListEvaluationError::Failure(error)))
            }
            _ => unreachable!(),
        }
    }

    fn eval_macroexpand(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        if let Some(rest) = forms.drop_first() {
            if let Some(Value::List(value)) = rest.first() {
                return self.macroexpand(value);
            }
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not evaluate `macroexpand`".to_string(),
        )))
    }

    fn eval_try(&mut self, forms: PersistentList<Value>) -> EvaluationResult<Value> {
        if let Some(rest) = forms.drop_first() {
            let catch_form = match rest.last() {
                Some(Value::List(last_form)) => match last_form.first() {
                    Some(Value::Symbol(s, None)) if s == "catch*" => {
                        // FIXME: deduplicate analysis of `catch*` here...
                        if let Some(catch_form) = last_form.drop_first() {
                            if let Some(exception_symbol) = catch_form.first() {
                                match exception_symbol {
                                    s @ Value::Symbol(_, None) => {
                                        if let Some(exception_body) = catch_form.drop_first() {
                                            let mut exception_binding = PersistentVector::new();
                                            exception_binding.push_back_mut(s.clone());
                                            let mut scopes = vec![];
                                            let mut captures = vec![];
                                            let body = self.analyze_symbols_in_lambda(
                                                exception_body,
                                                &exception_binding,
                                                &mut scopes,
                                                &mut captures,
                                            )?;
                                            Some(body)
                                        } else {
                                            None
                                        }
                                    }
                                    _ => {
                                        return Err(EvaluationError::List(
                                            ListEvaluationError::Failure(
                                                "could not evaluate `catch*`".to_string(),
                                            ),
                                        ));
                                    }
                                }
                            } else {
                                None
                            }
                        } else {
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `catch*`".to_string(),
                            )));
                        }
                    }
                    _ => None,
                },
                // FIXME: avoid clones here...
                Some(f @ Value::Fn(..)) => Some(f.clone()),
                Some(f @ Value::FnWithCaptures(..)) => Some(f.clone()),
                _ => None,
            };
            let forms_to_eval = if catch_form.is_none() {
                rest
            } else {
                let mut forms_to_eval = vec![];
                for (index, form) in rest.iter().enumerate() {
                    if index == rest.len() - 1 {
                        break;
                    }
                    forms_to_eval.push(form.clone());
                }
                PersistentList::from_iter(forms_to_eval)
            };
            match self.eval_do_inner(&forms_to_eval)? {
                e @ Value::Exception(_) if exception_is_thrown(&e) => match catch_form {
                    Some(Value::Fn(FnImpl { body, level, .. })) => {
                        self.enter_scope();
                        let parameter = lambda_parameter_key(0, level);
                        self.insert_value_in_current_scope(&parameter, exception_from_thrown(&e));
                        let result = self.eval_do_inner(&body);
                        self.leave_scope();
                        return result;
                    }
                    Some(Value::FnWithCaptures(FnWithCapturesImpl {
                        f: FnImpl { body, level, .. },
                        mut captures,
                    })) => {
                        for (capture, value) in &mut captures {
                            let captured_value =
                                resolve_symbol_in_scopes(self.scopes.iter().rev(), capture)
                                    .ok_or_else(|| {
                                        EvaluationError::Symbol(
                                            SymbolEvaluationError::UnableToResolveSymbolToValue(
                                                capture.to_string(),
                                            ),
                                        )
                                    })?;
                            *value = Some(captured_value.clone());
                        }
                        self.enter_scope();
                        for (capture, value) in captures {
                            if let Some(value) = value {
                                self.insert_value_in_current_scope(&capture, value);
                            } else {
                                return Err(EvaluationError::List(
                                    ListEvaluationError::MissingCapturedValue(capture),
                                ));
                            }
                        }
                        self.enter_scope();
                        let parameter = lambda_parameter_key(0, level);
                        self.insert_value_in_current_scope(&parameter, exception_from_thrown(&e));
                        let result = self.eval_do_inner(&body);
                        self.leave_scope();
                        self.leave_scope();
                        return result;
                    }
                    _ => return Ok(e),
                },
                result => return Ok(result),
            }
        }
        Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not evaluate `try*`".to_string(),
        )))
    }

    fn eval_list(&mut self, forms: &PersistentList<Value>) -> EvaluationResult<Value> {
        match self.macroexpand(forms)? {
            // macroexpand to list, interpret...
            Value::List(forms) => {
                if let Some(operator_form) = forms.first() {
                    // handle special forms or any apply-phase constructs
                    match operator_form {
                        Value::Symbol(s, None) if s == "def!" => return self.eval_def(forms),
                        Value::Symbol(s, None) if s == "var" => return self.eval_var(forms),
                        Value::Symbol(s, None) if s == "let*" => return self.eval_let(forms),
                        Value::Symbol(s, None) if s == "loop*" => return self.eval_loop(forms),
                        Value::Symbol(s, None) if s == "recur" => return self.eval_recur(forms),
                        Value::Symbol(s, None) if s == "if" => return self.eval_if(forms),
                        Value::Symbol(s, None) if s == "do" => return self.eval_do(forms),
                        Value::Symbol(s, None) if s == "fn*" => return self.eval_fn(forms),
                        Value::Symbol(s, None) if s == "quote" => return self.eval_quote(forms),
                        Value::Symbol(s, None) if s == "quasiquote" => {
                            return self.eval_quasiquote(forms)
                        }
                        Value::Symbol(s, None) if s == "defmacro!" => {
                            return self.eval_defmacro(forms)
                        }
                        Value::Symbol(s, None) if s == "macroexpand" => {
                            return self.eval_macroexpand(forms)
                        }
                        Value::Symbol(s, None) if s == "try*" => return self.eval_try(forms),
                        // apply phase when operator is already evaluated:
                        Value::Fn(f) => return self.apply_fn(f, &forms),
                        Value::FnWithCaptures(FnWithCapturesImpl { f, .. }) => {
                            return self.apply_fn(f, &forms)
                        }
                        Value::Primitive(native_fn) => {
                            return self.apply_primitive(native_fn, &forms)
                        }
                        _ => match self.evaluate(operator_form)? {
                            Value::Fn(f) => return self.apply_fn(&f, &forms),
                            Value::FnWithCaptures(FnWithCapturesImpl { f, captures }) => {
                                self.enter_scope();
                                for (capture, value) in captures {
                                    if let Some(value) = value {
                                        self.insert_value_in_current_scope(&capture, value);
                                    } else {
                                        return Err(EvaluationError::List(
                                            ListEvaluationError::MissingCapturedValue(capture),
                                        ));
                                    }
                                }
                                let result = self.apply_fn(&f, &forms);
                                self.leave_scope();
                                return result;
                            }
                            Value::Primitive(native_fn) => {
                                return self.apply_primitive(&native_fn, &forms);
                            }
                            v => {
                                return Err(EvaluationError::List(
                                    ListEvaluationError::CannotInvoke(v),
                                ));
                            }
                        },
                    }
                }
                // else, empty list
                Ok(Value::List(forms))
            }
            // macroexpand to value other than list, just evaluate
            expansion => self.evaluate(&expansion),
        }
    }

    /// Evaluate the `form` according to the semantics of the language.
    pub fn evaluate(&mut self, form: &Value) -> EvaluationResult<Value> {
        match form {
            Value::Nil => Ok(Value::Nil),
            Value::Bool(b) => Ok(Value::Bool(*b)),
            Value::Number(n) => Ok(Value::Number(*n)),
            Value::String(s) => Ok(Value::String(s.to_string())),
            Value::Keyword(id, ns_opt) => Ok(Value::Keyword(
                id.to_string(),
                ns_opt.as_ref().map(String::from),
            )),
            Value::Symbol(id, ns_opt) => self.resolve_symbol(id, ns_opt.as_ref()),
            Value::List(forms) => self.eval_list(forms),
            Value::Vector(forms) => {
                let mut result = PersistentVector::new();
                for form in forms {
                    let value = self.evaluate(form)?;
                    result.push_back_mut(value);
                }
                Ok(Value::Vector(result))
            }
            Value::Map(forms) => {
                let mut result = PersistentMap::new();
                for (k, v) in forms {
                    let key = self.evaluate(k)?;
                    let value = self.evaluate(v)?;
                    result.insert_mut(key, value);
                }
                Ok(Value::Map(result))
            }
            Value::Set(forms) => {
                let mut result = PersistentSet::new();
                for form in forms {
                    let value = self.evaluate(form)?;
                    result.insert_mut(value);
                }
                Ok(Value::Set(result))
            }
            Value::Var(v) => Ok(var_impl_into_inner(v)),
            f @ Value::Fn(_) => Ok(f.clone()),
            f @ Value::FnWithCaptures(_) => Ok(f.clone()),
            Value::Primitive(_) => unreachable!(),
            Value::Recur(_) => unreachable!(),
            a @ Value::Atom(_) => Ok(a.clone()),
            Value::Macro(_) => unreachable!(),
            Value::Exception(_) => unreachable!(),
        }
    }

    /// Evaluate `form` in the global scope of the interpreter.
    /// This method is exposed for the `eval` primitive which
    /// has these semantics.
    pub fn evaluate_in_global_scope(&mut self, form: &Value) -> EvaluationResult<Value> {
        let mut child_scopes: Vec<_> = self.scopes.drain(1..).collect();
        let result = self.evaluate(form);
        self.scopes.append(&mut child_scopes);
        result
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::reader::read;
    use crate::value::{
        atom_with_value, exception, list_with_values, map_with_values, set_with_values,
        vector_with_values,
    };
    use Value::*;

    fn run_eval_test(test_cases: &[(&str, Value)]) {
        let mut has_err = false;
        for (input, expected) in test_cases {
            let forms = match read(input) {
                Ok(forms) => forms,
                Err(e) => {
                    has_err = true;
                    println!("failure: reading `{}` failed: {}", input, e);
                    continue;
                }
            };
            let mut interpreter = Interpreter::default();
            let mut final_result: Option<Value> = None;
            for form in &forms {
                match interpreter.evaluate(form) {
                    Ok(result) => {
                        final_result = Some(result);
                    }
                    Err(e) => {
                        has_err = true;
                        println!(
                            "failure: evaluating `{}` should give `{}` but errored: {}",
                            input, expected, e
                        );
                    }
                }
            }
            if let Some(final_result) = final_result {
                if final_result != *expected {
                    has_err = true;
                    println!(
                        "failure: evaluating `{}` should give `{}` but got: {}",
                        input, expected, final_result
                    );
                }
            }
        }
        assert!(!has_err);
    }

    #[test]
    fn test_self_evaluating() {
        let test_cases = vec![
            ("nil", Nil),
            ("true", Bool(true)),
            ("false", Bool(false)),
            ("1337", Number(1337)),
            ("-1337", Number(-1337)),
            ("\"hi\"", String("hi".to_string())),
            (r#""""#, String("".to_string())),
            ("\"abc\"", String("abc".to_string())),
            ("\"abc   def\"", String("abc   def".to_string())),
            ("\"abc\\ndef\\nghi\"", String("abc\\ndef\\nghi".to_string())),
            ("\"abc\\def\\ghi\"", String("abc\\def\\ghi".to_string())),
            ("\"\\\\n\"", String("\\\\n".to_string())),
            (":hi", Keyword("hi".to_string(), None)),
            (
                ":foo/hi",
                Keyword("hi".to_string(), Some("foo".to_string())),
            ),
            ("()", List(PersistentList::new())),
            ("[]", Vector(PersistentVector::new())),
            ("{}", Map(PersistentMap::new())),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_apply() {
        let test_cases = vec![
            ("(+)", Number(0)),
            ("(+ 1)", Number(1)),
            ("(+ 1 10)", Number(11)),
            ("(+ 1 10 2)", Number(13)),
            ("(- 1)", Number(-1)),
            ("(- 10 9)", Number(1)),
            ("(- 10 20)", Number(-10)),
            ("(- 10 20 10)", Number(-20)),
            ("(*)", Number(1)),
            ("(* 2)", Number(2)),
            ("(* 2 3)", Number(6)),
            ("(* 2 3 1 1 1)", Number(6)),
            ("(/ 2)", Number(0)),
            ("(/ 1)", Number(1)),
            ("(/ 22 2)", Number(11)),
            ("(/ 22 2 1 1 1)", Number(11)),
            ("(/ 22 2 1 1 1)", Number(11)),
            ("(+ 2 (* 3 4))", Number(14)),
            ("(+ 5 (* 2 3))", Number(11)),
            ("(- (+ 5 (* 2 3)) 3)", Number(8)),
            ("(/ (- (+ 5 (* 2 3)) 3) 4)", Number(2)),
            ("(/ (- (+ 515 (* 87 311)) 302) 27)", Number(1010)),
            ("(* -3 6)", Number(-18)),
            ("(/ (- (+ 515 (* -87 311)) 296) 27)", Number(-994)),
            (
                "[1 2 (+ 1 2)]",
                vector_with_values(vec![Number(1), Number(2), Number(3)]),
            ),
            (
                "{\"a\" (+ 7 8)}",
                map_with_values(vec![(String("a".to_string()), Number(15))]),
            ),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_do() {
        let test_cases = vec![
            ("(do )", Nil),
            ("(do 1 2 3)", Number(3)),
            ("(do (do 1 2))", Number(2)),
            ("(do (prn 101))", Nil),
            ("(do (prn 101) 7)", Number(7)),
            ("(do (prn 101) (prn 102) (+ 1 2))", Number(3)),
            ("(do (def! a 6) 7 (+ a 8))", Number(14)),
            ("(do (def! a 6) 7 (+ a 8) a)", Number(6)),
            ("(def! DO (fn* [a] 7)) (DO 3)", Number(7)),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_def() {
        let test_cases = vec![
            (
                "(def! a 3)",
                var_with_value(Number(3), DEFAULT_NAMESPACE, "a"),
            ),
            ("(def! a 3) (+ a 1)", Number(4)),
            ("(def! a (+ 1 7)) (+ a 1)", Number(9)),
            (
                "(def! some-num 3)",
                var_with_value(Number(3), DEFAULT_NAMESPACE, "some-num"),
            ),
            (
                "(def! SOME-NUM 4)",
                var_with_value(Number(4), DEFAULT_NAMESPACE, "SOME-NUM"),
            ),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_let() {
        let test_cases = vec![
            ("(let* [] )", Nil),
            ("(let* [a 1] )", Nil),
            ("(let* [a 3] a)", Number(3)),
            ("(let* [b 3] (+ b 5))", Number(8)),
            ("(let* [a 3] (+ a (let* [c 5] c)))", Number(8)),
            ("(let* [a (+ 1 2)] (+ a (let* [c 5] c)))", Number(8)),
            ("(let* [a (+ 1 2)] (+ a (let* [a 5] a)))", Number(8)),
            ("(let* [p (+ 2 3) q (+ 2 p)] (+ p q))", Number(12)),
            ("(let* [a 3] (+ a (let* [a 5] a)))", Number(8)),
            ("(let* [a 3 b a] (+ b 5))", Number(8)),
            (
                "(let* [a 3 b 33] (+ a (let* [c 4] (+ c 1)) b 5))",
                Number(46),
            ),
            ("(def! a 1) (let* [a 3] a)", Number(3)),
            ("(def! a (let* [z 33] z)) a", Number(33)),
            ("(def! a (let* [z 33] z)) (let* [a 3] a)", Number(3)),
            ("(def! a (let* [z 33] z)) (let* [a 3] a) a", Number(33)),
            ("(def! a 1) (let* [a 3] a) a", Number(1)),
            ("(def! b 1) (let* [a 3] (+ a b))", Number(4)),
            (
                "(let* [a 5 b 6] [3 4 a [b 7] 8])",
                vector_with_values(vec![
                    Number(3),
                    Number(4),
                    Number(5),
                    vector_with_values(vec![Number(6), Number(7)]),
                    Number(8),
                ]),
            ),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_if() {
        let test_cases = vec![
            ("(if true 1 2)", Number(1)),
            ("(if true 1)", Number(1)),
            ("(if false 1 2)", Number(2)),
            ("(if false 7 false)", Bool(false)),
            ("(if true (+ 1 7) (+ 1 8))", Number(8)),
            ("(if false (+ 1 7) (+ 1 8))", Number(9)),
            ("(if false 2)", Nil),
            ("(if false (+ 1 7))", Nil),
            ("(if false (/ 1 0))", Nil),
            ("(if nil 1 2)", Number(2)),
            ("(if 0 1 2)", Number(1)),
            ("(if (list) 1 2)", Number(1)),
            ("(if (list 1 2 3) 1 2)", Number(1)),
            ("(= (list) nil)", Bool(false)),
            ("(if nil 2)", Nil),
            ("(if true 2)", Number(2)),
            ("(if false (/ 1 0))", Nil),
            ("(if nil (/ 1 0))", Nil),
            ("(let* [b nil] (if b 2 3))", Number(3)),
            ("(if (> (count (list 1 2 3)) 3) 89 78)", Number(78)),
            ("(if (>= (count (list 1 2 3)) 3) 89 78)", Number(89)),
            ("(if \"\" 7 8)", Number(7)),
            ("(if [] 7 8)", Number(7)),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_fn() {
        let test_cases = vec![
            ("((fn* [a] (+ a 1)) 23)", Number(24)),
            ("((fn* [a b] (+ a b)) 23 1)", Number(24)),
            ("((fn* [] (+ 4 3)) )", Number(7)),
            ("((fn* [f x] (f x)) (fn* [a] (+ 1 a)) 7)", Number(8)),
            ("((fn* [a] (+ a 1) 25) 23)", Number(25)),
            ("((fn* [a] (let* [b 2] (+ a b))) 23)", Number(25)),
            ("((fn* [a] (let* [a 2] (+ a a))) 23)", Number(4)),
            (
                "(def! inc (fn* [a] (+ a 1))) ((fn* [a] (inc a)) 1)",
                Number(2),
            ),
            ("((fn* [a] ((fn* [b] (+ b 1)) a)) 1)", Number(2)),
            ("((fn* [a] ((fn* [a] (+ a 1)) a)) 1)", Number(2)),
            ("((fn* [] ((fn* [] ((fn* [] 13))))))", Number(13)),
            (
                "(def! factorial (fn* [n] (if (< n 2) 1 (* n (factorial (- n 1)))))) (factorial 8)",
                Number(40320),
            ),
            (
                "(def! fibo (fn* [N] (if (= N 0) 1 (if (= N 1) 1 (+ (fibo (- N 1)) (fibo (- N 2))))))) (fibo 1)",
                Number(1),
            ),
            (
                "(def! fibo (fn* [N] (if (= N 0) 1 (if (= N 1) 1 (+ (fibo (- N 1)) (fibo (- N 2))))))) (fibo 2)",
                Number(2),
            ),
            (
                "(def! fibo (fn* [N] (if (= N 0) 1 (if (= N 1) 1 (+ (fibo (- N 1)) (fibo (- N 2))))))) (fibo 4)",
                Number(5),
            ),
            ("(def! f (fn* [a] (+ a 1))) (f 23)", Number(24)),
            (
                "(def! b 12) (def! f (fn* [a] (+ a b))) (def! b 22) (f 1)",
                Number(23),
            ),
            (
                "(def! b 12) (def! f (fn* [a] ((fn* [] (+ a b))))) (def! b 22) (f 1)",
                Number(23),
            ),
            (
                "(def! gen-plus5 (fn* [] (fn* [b] (+ 5 b)))) (def! plus5 (gen-plus5)) (plus5 7)",
                Number(12),
            ),
            ("(((fn* [a] (fn* [b] (+ a b))) 5) 7)", Number(12)),
            ("(def! gen-plus-x (fn* [x] (fn* [b] (+ x b)))) (def! plus7 (gen-plus-x 7)) (plus7 8)", Number(15)),
            ("((((fn* [a] (fn* [b] (fn* [c] (+ a b c)))) 1) 2) 3)", Number(6)),
            ("(((fn* [a] (fn* [b] (* b ((fn* [c] (+ a c)) 32)))) 1) 2)", Number(66)),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_loop_recur() {
        let test_cases = vec![
            ("(loop* [i 12] i)", Number(12)),
            ("(loop* [i 12])", Nil),
            ("(loop* [i 0] (if (< i 5) (recur (+ i 1)) i))", Number(5)),
            ("(def! factorial (fn* [n] (loop* [n n acc 1] (if (< n 1) acc (recur (- n 1) (* acc n)))))) (factorial 20)", Number(2432902008176640000)),
            (
                "(def! inc (fn* [a] (+ a 1))) (loop* [i 0] (if (< i 5) (recur (inc i)) i))",
                Number(5),
            ),
            // // NOTE: the following will overflow the stack
            // (
            //     "(def! f (fn* [i] (if (< i 400) (f (+ 1 i)) i))) (f 0)",
            //     Number(400),
            // ),
            // // but, the loop/recur form is stack efficient
            (
                "(def! f (fn* [i] (loop* [n i] (if (< n 400) (recur (+ 1 n)) n)))) (f 0)",
                Number(400),
            ),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_atoms() {
        let test_cases = vec![
            ("(atom 5)", atom_with_value(Number(5))),
            ("(atom? (atom 5))", Bool(true)),
            ("(atom? nil)", Bool(false)),
            ("(atom? 1)", Bool(false)),
            ("(def! a (atom 5)) (deref a)", Number(5)),
            ("(def! a (atom 5)) @a", Number(5)),
            ("(def! a (atom (fn* [a] (+ a 1)))) (@a 4)", Number(5)),
            ("(def! a (atom 5)) (reset! a 10)", Number(10)),
            ("(def! a (atom 5)) (reset! a 10) @a", Number(10)),
            (
                "(def! a (atom 5)) (def! inc (fn* [x] (+ x 1))) (swap! a inc)",
                Number(6),
            ),
            ("(def! a (atom 5)) (swap! a + 1 2 3 4 5)", Number(20)),
            (
                "(def! a (atom 5)) (swap! a + 1 2 3 4 5) (deref a)",
                Number(20),
            ),
            (
                "(def! a (atom 5)) (swap! a + 1 2 3 4 5) (reset! a 10)",
                Number(10),
            ),
            (
                "(def! a (atom 5)) (swap! a + 1 2 3 4 5) (reset! a 10) (deref a)",
                Number(10),
            ),
            (
                "(def! a (atom 7)) (def! f (fn* [] (swap! a inc))) (f) (f)",
                Number(9),
            ),
            (
                "(def! g (let* [a (atom 0)] (fn* [] (swap! a inc)))) (def! a (atom 1)) (g) (g) (g)",
                Number(3),
            ),
            (
                "(def! e (atom {:+ +})) (swap! e assoc :- -) ((get @e :+) 7 8)",
                Number(15),
            ),
            (
                "(def! e (atom {:+ +})) (swap! e assoc :- -) ((get @e :-) 11 8)",
                Number(3),
            ),
            (
                "(def! e (atom {:+ +})) (swap! e assoc :- -) (swap! e assoc :foo ()) (get @e :foo)",
                list_with_values(vec![]),
            ),
            (
                "(def! e (atom {:+ +})) (swap! e assoc :- -) (swap! e assoc :bar '(1 2 3)) (get @e :bar)",
                list_with_values(vec![Number(1), Number(2), Number(3)]),
            ),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_quote() {
        let test_cases = vec![
            ("(quote 5)", Number(5)),
            (
                "(quote (1 2 3))",
                list_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(quote (1 2 (into+ [] foo :baz/bar)))",
                list_with_values(
                    [
                        Number(1),
                        Number(2),
                        list_with_values(
                            [
                                Symbol("into+".to_string(), None),
                                Vector(PersistentVector::new()),
                                Symbol("foo".to_string(), None),
                                Keyword("bar".to_string(), Some("baz".to_string())),
                            ]
                            .iter()
                            .cloned(),
                        ),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_quasiquote() {
        let test_cases = vec![
            ("(quasiquote nil)", Nil),
            ("(quasiquote ())", list_with_values(vec![])),
            ("(quasiquote 7)", Number(7)),
            ("(quasiquote a)", Symbol("a".to_string(), None)),
            (
                "(quasiquote {:a b})",
                map_with_values(vec![(
                    Keyword("a".to_string(), None),
                    Symbol("b".to_string(), None),
                )]),
            ),
            (
                "(def! lst '(b c)) `(a lst d)",
                read("(a lst d)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`(1 2 (3 4))",
                read("(1 2 (3 4))")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`(nil)",
                read("(nil)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`(1 ())",
                read("(1 ())")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`(() 1)",
                read("(() 1)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`(2 () 1)",
                read("(2 () 1)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`(())",
                read("(())")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`(f () g (h) i (j k) l)",
                read("(f () g (h) i (j k) l)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            ("`~7", Number(7)),
            ("(def! a 8) `a", Symbol("a".to_string(), None)),
            ("(def! a 8) `~a", Number(8)),
            (
                "`(1 a 3)",
                read("(1 a 3)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! a 8) `(1 ~a 3)",
                read("(1 8 3)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! b '(1 :b :d)) `(1 b 3)",
                read("(1 b 3)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! b '(1 :b :d)) `(1 ~b 3)",
                read("(1 (1 :b :d) 3)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`(~1 ~2)",
                read("(1 2)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            ("(let* [x 0] `~x)", Number(0)),
            (
                "(def! lst '(b c)) `(a ~lst d)",
                read("(a (b c) d)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! lst '(b c)) `(a ~@lst d)",
                read("(a b c d)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! lst '(b c)) `(a ~@lst)",
                read("(a b c)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! lst '(b c)) `(~@lst 2)",
                read("(b c 2)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! lst '(b c)) `(~@lst ~@lst)",
                read("(b c b c)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "((fn* [q] (quasiquote ((unquote q) (quote (unquote q))))) (quote (fn* [q] (quasiquote ((unquote q) (quote (unquote q)))))))",
                read("((fn* [q] (quasiquote ((unquote q) (quote (unquote q))))) (quote (fn* [q] (quasiquote ((unquote q) (quote (unquote q)))))))")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`[]",
                read("[]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`[[]]",
                read("[[]]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`[()]",
                read("[()]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`([])",
                read("([])")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! a 8) `[1 a 3]",
                read("[1 a 3]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`[a [] b [c] d [e f] g]",
                read("[a [] b [c] d [e f] g]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! a 8) `[~a]",
                read("[8]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! a 8) `[(~a)]",
                read("[(8)]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! a 8) `([~a])",
                read("([8])")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! a 8) `[a ~a a]",
                read("[a 8 a]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! a 8) `([a ~a a])",
                read("([a 8 a])")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! a 8) `[(a ~a a)]",
                read("[(a 8 a)]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! c '(1 :b :d)) `[~@c]",
                read("[1 :b :d]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! c '(1 :b :d)) `[(~@c)]",
                read("[(1 :b :d)]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! c '(1 :b :d)) `([~@c])",
                read("([1 :b :d])")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! c '(1 :b :d)) `[1 ~@c 3]",
                read("[1 1 :b :d 3]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! c '(1 :b :d)) `([1 ~@c 3])",
                read("([1 1 :b :d 3])")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "(def! c '(1 :b :d)) `[(1 ~@c 3)]",
                read("[(1 1 :b :d 3)]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`(0 unquote)",
                read("(0 unquote)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`(0 splice-unquote)",
                read("(0 splice-unquote)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`[unquote 0]",
                read("[unquote 0]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
            (
                "`[splice-unquote 0]",
                read("[splice-unquote 0]")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_macros() {
        let test_cases = vec![
            ("(defmacro! one (fn* [] 1)) (one)", Number(1)),
            ("(defmacro! two (fn* [] 2)) (two)", Number(2)),
            ("(defmacro! unless (fn* [pred a b] `(if ~pred ~b ~a))) (unless false 7 8)", Number(7)),
            ("(defmacro! unless (fn* [pred a b] `(if ~pred ~b ~a))) (unless true 7 8)", Number(8)),
            ("(defmacro! unless (fn* [pred a b] (list 'if (list 'not pred) a b))) (unless false 7 8)", Number(7)),
            ("(defmacro! unless (fn* [pred a b] (list 'if (list 'not pred) a b))) (unless true 7 8)", Number(8)),
            ("(defmacro! one (fn* [] 1)) (macroexpand (one))", Number(1)),
            ("(defmacro! unless (fn* [pred a b] `(if ~pred ~b ~a))) (macroexpand (unless PRED A B))",
                read("(if PRED B A)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some")
            ),
            ("(defmacro! unless (fn* [pred a b] (list 'if (list 'not pred) a b))) (macroexpand (unless PRED A B))",
                read("(if (not PRED) A B)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some")
            ),
            ("(defmacro! unless (fn* [pred a b] (list 'if (list 'not pred) a b))) (macroexpand (unless 2 3 4))",
                read("(if (not 2) 3 4)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some")
            ),
            ("(defmacro! identity (fn* [x] x)) (let* [a 123] (macroexpand (identity a)))",
                read("a")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some")
            ),
            ("(defmacro! identity (fn* [x] x)) (let* [a 123] (identity a))",
                Number(123),
            ),
            ("(macroexpand (cond))", Nil),
            ("(cond)", Nil),
            ("(macroexpand (cond X Y))",
                read("(if X Y (cond))")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some")
            ),
            ("(cond true 7)", Number(7)),
            ("(cond true 7 true 8)", Number(7)),
            ("(cond false 7)", Nil),
            ("(cond false 7 true 8)", Number(8)),
            ("(cond false 7 false 8 :else 9)", Number(9)),
            ("(cond false 7 (= 2 2) 8 :else 9)", Number(8)),
            ("(cond false 7 false 8 false 9)", Nil),
            ("(let* [x (cond false :no true :yes)] x)", Keyword("yes".to_string(), None)),
            ("(macroexpand (cond X Y Z T))",
                read("(if X Y (cond Z T))")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some")
            ),
            ("(def! x 2) (defmacro! a (fn* [] x)) (a)", Number(2)),
            ("(def! x 2) (defmacro! a (fn* [] x)) (let* [x 3] (a))", Number(2)),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_try_catch() {
        let basic_exc = exception("", &Value::String("test".to_string()));
        let exc = exception(
            "test",
            &map_with_values(vec![(
                Keyword("cause".to_string(), None),
                String("no memory".to_string()),
            )]),
        );
        let test_cases = vec![
            ( "(throw \"test\")", basic_exc),
            ( "(throw {:msg :foo})", exception("", &map_with_values(vec![(Keyword("msg".to_string(), None), Keyword("foo".to_string(), None))]))),
            ( "(try* (throw '(1 2 3)) (catch* e e))", exception("", &list_with_values(vec![Number(1), Number(2), Number(3)]))),
            (
                "(try* 22)",
                Number(22),
            ),
            (
                "(try* (prn 222) 22)",
                Number(22),
            ),
            (
                "(try* (ex-info \"test\" {:cause \"no memory\"}))",
                exc.clone(),
            ),
            (
                "(try* 123 (catch* e 0))",
                Number(123),
            ),
            (
                "(try* (ex-info \"test\" {:cause \"no memory\"}) (catch* e 0))",
                exc,
            ),
            (
                "(try* (throw (ex-info \"test\" {:cause \"no memory\"})) (catch* e (str e)))",
                String("exception: test, {:cause \"no memory\"}".to_string()),
            ),
            (
                "(try* (throw (ex-info \"test\" {:cause \"no memory\"})) (catch* e 999))",
                Number(999),
            ),
            (
                // must throw exception to change control flow
                "(try* (ex-info \"first\" {}) (ex-info \"test\" {:cause \"no memory\"}) 22 (catch* e e))",
                Number(22),
            ),
            (
                // must throw exception to change control flow
                "(try* (ex-info \"first\" {}) (ex-info \"test\" {:cause \"no memory\"}) (catch* e 22))",
                exception(
                    "test",
                    &map_with_values(
                        [(
                            Keyword("cause".to_string(), None),
                            String("no memory".to_string()),
                        )]
                        .iter()
                        .cloned(),
                    ),
                ),
            ),
            (
                "(try* (throw (ex-info \"first\" {})) (ex-info \"test\" {:cause \"no memory\"}) (catch* e e))",
                exception(
                    "first",
                    &Map(PersistentMap::new()),
                ),
            ),
            (
                "(try* (throw (ex-info \"first\" {})) (ex-info \"test\" {:cause \"no memory\"}) (catch* e (prn e) 22))",
                Number(22),
            ),
            (
                "(def! f (fn* [] (try* (throw (ex-info \"test\" {:cause 22})) (catch* e (prn e) 22)))) (f)",
                Number(22),
            ),
            (
                "(def! f (fn* [] (try* (throw (ex-info \"test\" {:cause 'foo})) (catch* e (prn e) 22)))) (f)",
                Number(22),
            ),
            (
                "(try* (do 1 2 (try* (do 3 4 (throw :e1)) (catch* e (throw (ex-info \"foo\" :bar))))) (catch* e :outer))",
                Keyword("outer".to_string(), None),
            ),
            (
                "(try* (do (try* \"t1\" (catch* e \"c1\")) (throw \"e1\")) (catch* e \"c2\"))",
                String("c2".to_string()),
            ),
            (
                "(try* (try* (throw \"e1\") (catch* e (throw \"e2\"))) (catch* e \"c2\"))",
                String("c2".to_string()),
            ),
            (
                "(def! f (fn* [a] ((fn* [] (try* (throw (ex-info \"test\" {:cause 22})) (catch* e (prn e) a)))))) (f 2222)",
                Number(2222),
            ),
            (
                "(((fn* [a] (fn* [] (try* (throw (ex-info \"\" {:foo 2})) (catch* e (prn e) a)))) 2222))",
                Number(2222),
            ),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_var_args() {
        let test_cases = vec![
            ("((fn* [& rest] (first rest)))", Nil),
            ("((fn* [& rest] (first rest)) 5)", Number(5)),
            ("((fn* [& rest] (first rest)) 5 6 7)", Number(5)),
            ("((fn* [& rest] (last rest)) 5 6 7)", Number(7)),
            ("((fn* [& rest] (nth rest 1)) 5 6 7)", Number(6)),
            ("((fn* [& rest] (count rest)))", Number(0)),
            ("((fn* [& rest] (count rest)) 1)", Number(1)),
            ("((fn* [& rest] (count rest)) 1 2 3)", Number(3)),
            ("((fn* [& rest] (list? rest)) 1 2 3)", Bool(true)),
            ("((fn* [& rest] (list? rest)))", Bool(true)),
            ("((fn* [a & rest] (count rest)) 1 2 3)", Number(2)),
            ("((fn* [a & rest] (count rest)) 3)", Number(0)),
            ("((fn* [a & rest] (list? rest)) 3)", Bool(true)),
            ("((fn* [a b & rest] (apply + a b rest)) 1 2 3)", Number(6)),
            (
                "(def! f (fn* [a & rest] (count rest))) (f 1 2 3)",
                Number(2),
            ),
            (
                "(def! f (fn* [a b & rest] (apply + a b rest))) (f 1 2 3 4)",
                Number(10),
            ),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_interpreter() {
        let test_cases = vec![
            ("(list? *command-line-args*)", Bool(true)),
            ("*command-line-args*", list_with_values(vec![])),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_prelude() {
        let test_cases = vec![
            ("(list)", list_with_values(vec![])),
            (
                "(list 1 2)",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            ("(list? (list 1))", Bool(true)),
            ("(list? (list))", Bool(true)),
            ("(list? [1 2])", Bool(false)),
            ("(empty? (list))", Bool(true)),
            ("(empty? (list 1))", Bool(false)),
            ("(empty? [1 2 3])", Bool(false)),
            ("(empty? [])", Bool(true)),
            ("(count nil)", Number(0)),
            ("(count \"hi\")", Number(2)),
            ("(count \"\")", Number(0)),
            ("(count (list))", Number(0)),
            ("(count (list 44 42 41))", Number(3)),
            ("(count [])", Number(0)),
            ("(count [1 2 3])", Number(3)),
            ("(count {})", Number(0)),
            ("(count {:a 1 :b 2})", Number(2)),
            ("(count #{})", Number(0)),
            ("(count #{:a 1 :b 2})", Number(4)),
            ("(if (< 2 3) 12 13)", Number(12)),
            ("(> 13 12)", Bool(true)),
            ("(> 13 13)", Bool(false)),
            ("(> 12 13)", Bool(false)),
            ("(< 13 12)", Bool(false)),
            ("(< 13 13)", Bool(false)),
            ("(< 12 13)", Bool(true)),
            ("(<= 12 12)", Bool(true)),
            ("(<= 13 12)", Bool(false)),
            ("(<= 12 13)", Bool(true)),
            ("(>= 13 12)", Bool(true)),
            ("(>= 13 13)", Bool(true)),
            ("(>= 13 14)", Bool(false)),
            ("(= 12 12)", Bool(true)),
            ("(= 12 13)", Bool(false)),
            ("(= 13 12)", Bool(false)),
            ("(= 0 0)", Bool(true)),
            ("(= 1 0)", Bool(false)),
            ("(= true true)", Bool(true)),
            ("(= true false)", Bool(false)),
            ("(= false false)", Bool(true)),
            ("(= nil nil)", Bool(true)),
            ("(= (list) (list))", Bool(true)),
            ("(= (list) ())", Bool(true)),
            ("(= (list 1 2) '(1 2))", Bool(true)),
            ("(= (list 1 ) ())", Bool(false)),
            ("(= (list ) '(1))", Bool(false)),
            ("(= 0 (list))", Bool(false)),
            ("(= (list) 0)", Bool(false)),
            ("(= (list nil) (list))", Bool(false)),
            ("(= 1 (+ 1 1))", Bool(false)),
            ("(= 2 (+ 1 1))", Bool(true)),
            ("(= nil (+ 1 1))", Bool(false)),
            ("(= nil nil)", Bool(true)),
            ("(= \"\" \"\")", Bool(true)),
            ("(= \"abc\" \"abc\")", Bool(true)),
            ("(= \"\" \"abc\")", Bool(false)),
            ("(= \"abc\" \"\")", Bool(false)),
            ("(= \"abc\" \"def\")", Bool(false)),
            ("(= \"abc\" \"ABC\")", Bool(false)),
            ("(= (list) \"\")", Bool(false)),
            ("(= \"\" (list))", Bool(false)),
            ("(= :abc :abc)", Bool(true)),
            ("(= :abc :def)", Bool(false)),
            ("(= :abc \":abc\")", Bool(false)),
            ("(= (list :abc) (list :abc))", Bool(true)),
            ("(= [] (list))", Bool(true)),
            ("(= [7 8] [7 8])", Bool(true)),
            ("(= [:abc] [:abc])", Bool(true)),
            ("(= (list 1 2) [1 2])", Bool(true)),
            ("(= (list 1) [])", Bool(false)),
            ("(= [] (list 1))", Bool(false)),
            ("(= [] [1])", Bool(false)),
            ("(= 0 [])", Bool(false)),
            ("(= [] 0)", Bool(false)),
            ("(= [] \"\")", Bool(false)),
            ("(= \"\" [])", Bool(false)),
            ("(= [(list)] (list []))", Bool(true)),
            ("(= 'abc 'abc)", Bool(true)),
            ("(= 'abc 'abdc)", Bool(false)),
            ("(= 'abc \"abc\")", Bool(false)),
            ("(= \"abc\" 'abc)", Bool(false)),
            ("(= \"abc\" (str 'abc))", Bool(true)),
            ("(= 'abc nil)", Bool(false)),
            ("(= nil 'abc)", Bool(false)),
            ("(= {} {})", Bool(true)),
            ("(= {} (hash-map))", Bool(true)),
            ("(= {:a 11 :b 22} (hash-map :b 22 :a 11))", Bool(true)),
            (
                "(= {:a 11 :b [22 33]} (hash-map :b [22 33] :a 11))",
                Bool(true),
            ),
            (
                "(= {:a 11 :b {:c 22}} (hash-map :b (hash-map :c 22) :a 11))",
                Bool(true),
            ),
            ("(= {:a 11 :b 22} (hash-map :b 23 :a 11))", Bool(false)),
            ("(= {:a 11 :b 22} (hash-map :a 11))", Bool(false)),
            ("(= {:a [11 22]} {:a (list 11 22)})", Bool(true)),
            ("(= {:a 11 :b 22} (list :a 11 :b 22))", Bool(false)),
            ("(= {} [])", Bool(false)),
            ("(= [] {})", Bool(false)),
            (
                "(= [1 2 (list 3 4 [5 6])] (list 1 2 [3 4 (list 5 6)]))",
                Bool(true),
            ),
            (
                "(read-string \"(+ 1 2)\")",
                List(PersistentList::from_iter(vec![
                    Symbol("+".to_string(), None),
                    Number(1),
                    Number(2),
                ])),
            ),
            (
                "(read-string \"(1 2 (3 4) nil)\")",
                List(PersistentList::from_iter(vec![
                    Number(1),
                    Number(2),
                    List(PersistentList::from_iter(vec![Number(3), Number(4)])),
                    Nil,
                ])),
            ),
            ("(= nil (read-string \"nil\"))", Bool(true)),
            ("(read-string \"7 ;; comment\")", Number(7)),
            ("(read-string \"7;;!\")", Number(7)),
            ("(read-string \"7;;#\")", Number(7)),
            ("(read-string \"7;;$\")", Number(7)),
            ("(read-string \"7;;%\")", Number(7)),
            ("(read-string \"7;;'\")", Number(7)),
            ("(read-string \"7;;\\\\\")", Number(7)),
            ("(read-string \"7;;////////\")", Number(7)),
            ("(read-string \"7;;`\")", Number(7)),
            ("(read-string \"7;; &()*+,-./:;<=>?@[]^_{|}~\")", Number(7)),
            ("(read-string \";; comment\")", Nil),
            ("(eval (list + 1 2 3))", Number(6)),
            ("(eval (read-string \"(+ 2 3)\"))", Number(5)),
            (
                "(def! a 1) (let* [a 12] (eval (read-string \"a\")))",
                Number(1),
            ),
            (
                "(let* [b 12] (do (eval (read-string \"(def! aa 7)\")) aa))",
                Number(7),
            ),
            ("(str)", String("".to_string())),
            ("(str \"\")", String("".to_string())),
            ("(str \"hi\" 3 :foo)", String("hi3:foo".to_string())),
            ("(str \"hi   \" 3 :foo)", String("hi   3:foo".to_string())),
            ("(str [])", String("[]".to_string())),
            (
                "(cons 1 (list))",
                list_with_values([Number(1)].iter().cloned()),
            ),
            ("(cons 1 [])", list_with_values([Number(1)].iter().cloned())),
            (
                "(cons 1 (list 2))",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(cons 1 (list 2 3))",
                list_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(cons 1 [2 3])",
                list_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(cons [1] [2 3])",
                list_with_values(
                    [vector_with_values(vec![Number(1)]), Number(2), Number(3)]
                        .iter()
                        .cloned(),
                ),
            ),
            (
                "(def! a [2 3]) (cons 1 a)",
                list_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(def! a [2 3]) (cons 1 a) a",
                vector_with_values(vec![Number(2), Number(3)]),
            ),
            (
                "(cons (list 1) (list 2 3))",
                list_with_values(
                    [
                        list_with_values([Number(1)].iter().cloned()),
                        Number(2),
                        Number(3),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            ("(concat)", List(PersistentList::new())),
            ("(concat (concat))", List(PersistentList::new())),
            ("(concat (list) (list))", List(PersistentList::new())),
            ("(= () (concat))", Bool(true)),
            (
                "(concat (list 1 2))",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(concat (list 1) (list 2 3))",
                list_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(concat (list 1) [3 3] (list 2 3))",
                list_with_values(
                    [Number(1), Number(3), Number(3), Number(2), Number(3)]
                        .iter()
                        .cloned(),
                ),
            ),
            (
                "(concat [1 2] '(3 4) [5 6])",
                list_with_values(
                    [
                        Number(1),
                        Number(2),
                        Number(3),
                        Number(4),
                        Number(5),
                        Number(6),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            (
                "(concat (list 1) (list 2 3) (list (list 4 5) 6))",
                list_with_values(
                    [
                        Number(1),
                        Number(2),
                        Number(3),
                        list_with_values([Number(4), Number(5)].iter().cloned()),
                        Number(6),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            (
                "(def! a (list 1 2)) (def! b (list 3 4)) (concat a b (list 5 6))",
                list_with_values(
                    [
                        Number(1),
                        Number(2),
                        Number(3),
                        Number(4),
                        Number(5),
                        Number(6),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            (
                "(def! a (list 1 2)) (def! b (list 3 4)) (concat a b (list 5 6)) a",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(def! a (list 1 2)) (def! b (list 3 4)) (concat a b (list 5 6)) b",
                list_with_values([Number(3), Number(4)].iter().cloned()),
            ),
            (
                "(concat [1 2])",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(vec '(1 2 3))",
                vector_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(vec [1 2 3])",
                vector_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            ("(vec nil)", vector_with_values([].iter().cloned())),
            ("(vec '())", vector_with_values([].iter().cloned())),
            ("(vec [])", vector_with_values([].iter().cloned())),
            (
                "(def! a '(1 2)) (vec a)",
                vector_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(def! a '(1 2)) (vec a) a",
                list_with_values([Number(1), Number(2)].iter().cloned()),
            ),
            (
                "(vec '(1))",
                vector_with_values([Number(1)].iter().cloned()),
            ),
            ("(nth [1 2 3] 2)", Number(3)),
            ("(nth [1] 0)", Number(1)),
            ("(nth [1 2 nil] 2)", Nil),
            ("(nth '(1 2 3) 1)", Number(2)),
            ("(nth '(1 2 3) 0)", Number(1)),
            ("(nth '(1 2 nil) 2)", Nil),
            ("(first '(1 2 3))", Number(1)),
            ("(first '())", Nil),
            ("(first [1 2 3])", Number(1)),
            ("(first [10])", Number(10)),
            ("(first [])", Nil),
            ("(first nil)", Nil),
            (
                "(rest '(1 2 3))",
                list_with_values([Number(2), Number(3)].iter().cloned()),
            ),
            ("(rest '(1))", list_with_values(vec![])),
            ("(rest '())", List(PersistentList::new())),
            (
                "(rest [1 2 3])",
                list_with_values([Number(2), Number(3)].iter().cloned()),
            ),
            ("(rest [])", List(PersistentList::new())),
            ("(rest nil)", List(PersistentList::new())),
            ("(rest [10])", List(PersistentList::new())),
            (
                "(rest [10 11 12])",
                list_with_values(vec![Number(11), Number(12)]),
            ),
            (
                "(rest (cons 10 [11 12]))",
                list_with_values(vec![Number(11), Number(12)]),
            ),
            ("(apply str [1 2 3])", String("123".to_string())),
            ("(apply str '(1 2 3))", String("123".to_string())),
            ("(apply str 0 1 2 '(1 2 3))", String("012123".to_string())),
            ("(apply + '(2 3))", Number(5)),
            ("(apply + 4 '(5))", Number(9)),
            ("(apply + 4 [5])", Number(9)),
            ("(apply list ())", list_with_values(vec![])),
            ("(apply list [])", list_with_values(vec![])),
            ("(apply symbol? (list 'two))", Bool(true)),
            ("(apply (fn* [a b] (+ a b)) '(2 3))", Number(5)),
            ("(apply (fn* [a b] (+ a b)) 4 '(5))", Number(9)),
            ("(apply (fn* [a b] (+ a b)) [2 3])", Number(5)),
            ("(apply (fn* [a b] (+ a b)) 4 [5])", Number(9)),
            ("(apply (fn* [& rest] (list? rest)) [1 2 3])", Bool(true)),
            ("(apply (fn* [& rest] (list? rest)) [])", Bool(true)),
            ("(apply (fn* [a & rest] (list? rest)) [1])", Bool(true)),
            (
                "(def! inc (fn* [a] (+ a 1))) (map inc [1 2 3])",
                list_with_values(vec![Number(2), Number(3), Number(4)]),
            ),
            (
                "(map inc '(1 2 3))",
                list_with_values(vec![Number(2), Number(3), Number(4)]),
            ),
            (
                "(map (fn* [x] (* 2 x)) [1 2 3])",
                list_with_values(vec![Number(2), Number(4), Number(6)]),
            ),
            (
                "(map (fn* [& args] (list? args)) [1 2])",
                list_with_values(vec![Bool(true), Bool(true)]),
            ),
            (
                "(map symbol? '(nil false true))",
                list_with_values(vec![Bool(false), Bool(false), Bool(false)]),
            ),
            ("(= () (map str ()))", Bool(true)),
            ("(nil? nil)", Bool(true)),
            ("(nil? true)", Bool(false)),
            ("(nil? false)", Bool(false)),
            ("(nil? [1 2 3])", Bool(false)),
            ("(true? true)", Bool(true)),
            ("(true? nil)", Bool(false)),
            ("(true? false)", Bool(false)),
            ("(true? true?)", Bool(false)),
            ("(true? [1 2 3])", Bool(false)),
            ("(false? false)", Bool(true)),
            ("(false? nil)", Bool(false)),
            ("(false? true)", Bool(false)),
            ("(false? [1 2 3])", Bool(false)),
            ("(symbol? 'a)", Bool(true)),
            ("(symbol? 'foo/a)", Bool(true)),
            ("(symbol? :foo/a)", Bool(false)),
            ("(symbol? :a)", Bool(false)),
            ("(symbol? false)", Bool(false)),
            ("(symbol? true)", Bool(false)),
            ("(symbol? nil)", Bool(false)),
            ("(symbol? (symbol \"abc\"))", Bool(true)),
            ("(symbol? [1 2 3])", Bool(false)),
            ("(symbol \"hi\")", Symbol("hi".to_string(), None)),
            ("(keyword \"hi\")", Keyword("hi".to_string(), None)),
            ("(keyword :hi)", Keyword("hi".to_string(), None)),
            ("(keyword? :a)", Bool(true)),
            ("(keyword? false)", Bool(false)),
            ("(keyword? 'abc)", Bool(false)),
            ("(keyword? \"hi\")", Bool(false)),
            ("(keyword? \"\")", Bool(false)),
            ("(keyword? (keyword \"abc\"))", Bool(true)),
            (
                "(keyword? (first (keys {\":abc\" 123 \":def\" 456})))",
                Bool(false),
            ),
            ("(vector)", Vector(PersistentVector::new())),
            (
                "(vector 1)",
                vector_with_values([Number(1)].iter().cloned()),
            ),
            (
                "(vector 1 2 3)",
                vector_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            ("(vector? [1 2])", Bool(true)),
            ("(vector? '(1 2))", Bool(false)),
            ("(vector? :hi)", Bool(false)),
            ("(= [] (vector))", Bool(true)),
            ("(sequential? '(1 2))", Bool(true)),
            ("(sequential? [1 2])", Bool(true)),
            ("(sequential? :hi)", Bool(false)),
            ("(sequential? nil)", Bool(false)),
            ("(sequential? \"abc\")", Bool(false)),
            ("(sequential? sequential?)", Bool(false)),
            ("(hash-map)", Map(PersistentMap::new())),
            (
                "(hash-map :a 2)",
                map_with_values(
                    [(Keyword("a".to_string(), None), Number(2))]
                        .iter()
                        .cloned(),
                ),
            ),
            ("(map? {:a 1 :b 2})", Bool(true)),
            ("(map? {})", Bool(true)),
            ("(map? '())", Bool(false)),
            ("(map? [])", Bool(false)),
            ("(map? 'abc)", Bool(false)),
            ("(map? :abc)", Bool(false)),
            ("(map? [1 2])", Bool(false)),
            (
                "(assoc {} :a 1)",
                map_with_values(
                    [(Keyword("a".to_string(), None), Number(1))]
                        .iter()
                        .cloned(),
                ),
            ),
            (
                "(assoc {} :a 1 :b 3)",
                map_with_values(
                    [
                        (Keyword("a".to_string(), None), Number(1)),
                        (Keyword("b".to_string(), None), Number(3)),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            (
                "(assoc {:a 1} :b 3)",
                map_with_values(
                    [
                        (Keyword("a".to_string(), None), Number(1)),
                        (Keyword("b".to_string(), None), Number(3)),
                    ]
                    .iter()
                    .cloned(),
                ),
            ),
            (
                "(assoc {:a 1} :a 3 :c 33)",
                map_with_values(vec![
                    (Keyword("a".to_string(), None), Number(3)),
                    (Keyword("c".to_string(), None), Number(33)),
                ]),
            ),
            (
                "(assoc {} :a nil)",
                map_with_values(vec![(Keyword("a".to_string(), None), Nil)]),
            ),
            ("(dissoc {})", map_with_values([].iter().cloned())),
            ("(dissoc {} :a)", map_with_values([].iter().cloned())),
            (
                "(dissoc {:a 1 :b 3} :a)",
                map_with_values(
                    [(Keyword("b".to_string(), None), Number(3))]
                        .iter()
                        .cloned(),
                ),
            ),
            (
                "(dissoc {:a 1 :b 3} :a :b :c)",
                map_with_values([].iter().cloned()),
            ),
            ("(count (keys (assoc {} :b 2 :c 3)))", Number(2)),
            ("(get {:a 1} :a)", Number(1)),
            ("(get {:a 1} :b)", Nil),
            ("(get nil :b)", Nil),
            ("(contains? {:a 1} :b)", Bool(false)),
            ("(contains? {:a 1} :a)", Bool(true)),
            ("(contains? {:abc nil} :abc)", Bool(true)),
            ("(keyword? (nth (keys {:abc 123 :def 456}) 0))", Bool(true)),
            ("(keyword? (nth (vals {123 :abc 456 :def}) 0))", Bool(true)),
            ("(keys {})", Nil),
            (
                "(= (set '(:a :b :c)) (set (keys {:a 1 :b 2 :c 3})))",
                Bool(true),
            ),
            (
                "(= (set '(:a :c)) (set (keys {:a 1 :b 2 :c 3})))",
                Bool(false),
            ),
            ("(vals {})", Nil),
            (
                "(= (set '(1 2 3)) (set (vals {:a 1 :b 2 :c 3})))",
                Bool(true),
            ),
            (
                "(= (set '(1 2)) (set (vals {:a 1 :b 2 :c 3})))",
                Bool(false),
            ),
            ("(last '(1 2 3))", Number(3)),
            ("(last [1 2 3])", Number(3)),
            ("(last '())", Nil),
            ("(last [])", Nil),
            ("(not [])", Bool(false)),
            ("(not nil)", Bool(true)),
            ("(not true)", Bool(false)),
            ("(not false)", Bool(true)),
            ("(not 1)", Bool(false)),
            ("(not 0)", Bool(false)),
            ("(not \"a\")", Bool(false)),
            ("(not \"\")", Bool(false)),
            ("(not (= 1 1))", Bool(false)),
            ("(not (= 1 2))", Bool(true)),
            ("(set nil)", Set(PersistentSet::new())),
            // NOTE: these all rely on an _unguaranteed_ insertion order...
            (
                "(set \"hi\")",
                set_with_values(vec![String("h".to_string()), String("i".to_string())]),
            ),
            ("(set '(1 2))", set_with_values(vec![Number(1), Number(2)])),
            (
                "(set '(1 2 1 2 1 2 2 2 2))",
                set_with_values(vec![Number(1), Number(2)]),
            ),
            (
                "(set [1 2 1 2 1 2 2 2 2])",
                set_with_values(vec![Number(1), Number(2)]),
            ),
            (
                "(set {1 2 3 4})",
                set_with_values(vec![
                    vector_with_values(vec![Number(1), Number(2)]),
                    vector_with_values(vec![Number(3), Number(4)]),
                ]),
            ),
            (
                "(set #{1 2 3 4})",
                set_with_values(vec![Number(1), Number(2), Number(3), Number(4)]),
            ),
            ("(set? #{1 2 3 4})", Bool(true)),
            ("(set? nil)", Bool(false)),
            ("(set? '())", Bool(false)),
            ("(set? [])", Bool(false)),
            ("(set? {})", Bool(false)),
            ("(set? #{})", Bool(true)),
            ("(set? \"a\")", Bool(false)),
            ("(set? :a)", Bool(false)),
            ("(set? 'a)", Bool(false)),
            ("(string? nil)", Bool(false)),
            ("(string? true)", Bool(false)),
            ("(string? false)", Bool(false)),
            ("(string? [1 2 3])", Bool(false)),
            ("(string? 1)", Bool(false)),
            ("(string? :hi)", Bool(false)),
            ("(string? \"hi\")", Bool(true)),
            ("(string? string?)", Bool(false)),
            ("(number? nil)", Bool(false)),
            ("(number? true)", Bool(false)),
            ("(number? false)", Bool(false)),
            ("(number? [1 2 3])", Bool(false)),
            ("(number? 1)", Bool(true)),
            ("(number? -1)", Bool(true)),
            ("(number? :hi)", Bool(false)),
            ("(number? \"hi\")", Bool(false)),
            ("(number? string?)", Bool(false)),
            ("(fn? nil)", Bool(false)),
            ("(fn? true)", Bool(false)),
            ("(fn? false)", Bool(false)),
            ("(fn? [1 2 3])", Bool(false)),
            ("(fn? 1)", Bool(false)),
            ("(fn? -1)", Bool(false)),
            ("(fn? :hi)", Bool(false)),
            ("(fn? \"hi\")", Bool(false)),
            ("(fn? string?)", Bool(true)),
            ("(fn? (fn* [a] a))", Bool(true)),
            ("(def! foo (fn* [a] a)) (fn? foo)", Bool(true)),
            ("(defmacro! foo (fn* [a] a)) (fn? foo)", Bool(true)),
            (
                "(conj [1 2 3] 4)",
                vector_with_values(vec![Number(1), Number(2), Number(3), Number(4)]),
            ),
            (
                "(conj [1 2 3] 4 5)",
                vector_with_values(vec![Number(1), Number(2), Number(3), Number(4), Number(5)]),
            ),
            (
                "(conj '(1 2 3) 4 5)",
                list_with_values(vec![Number(5), Number(4), Number(1), Number(2), Number(3)]),
            ),
            (
                "(conj [3] [4 5])",
                vector_with_values(vec![
                    Number(3),
                    vector_with_values(vec![Number(4), Number(5)]),
                ]),
            ),
            (
                "(conj {:c :d} [1 2] {:a :b :c :e})",
                map_with_values(vec![
                    (
                        Keyword("c".to_string(), None),
                        Keyword("e".to_string(), None),
                    ),
                    (
                        Keyword("a".to_string(), None),
                        Keyword("b".to_string(), None),
                    ),
                    (Number(1), Number(2)),
                ]),
            ),
            (
                "(conj #{1 2} 1 3 2 2 2 2 1)",
                set_with_values(vec![Number(1), Number(2), Number(3)]),
            ),
            ("(macro? nil)", Bool(false)),
            ("(macro? true)", Bool(false)),
            ("(macro? false)", Bool(false)),
            ("(macro? [1 2 3])", Bool(false)),
            ("(macro? 1)", Bool(false)),
            ("(macro? -1)", Bool(false)),
            ("(macro? :hi)", Bool(false)),
            ("(macro? \"hi\")", Bool(false)),
            ("(macro? string?)", Bool(false)),
            ("(macro? {})", Bool(false)),
            ("(macro? (fn* [a] a))", Bool(false)),
            ("(def! foo (fn* [a] a)) (macro? foo)", Bool(false)),
            ("(defmacro! foo (fn* [a] a)) (macro? foo)", Bool(true)),
            ("(number? (time-ms))", Bool(true)),
            ("(seq nil)", Nil),
            ("(seq \"\")", Nil),
            (
                "(seq \"ab\")",
                list_with_values(vec![String("a".to_string()), String("b".to_string())]),
            ),
            ("(apply str (seq \"ab\"))", String("ab".to_string())),
            ("(seq '())", Nil),
            ("(seq '(1 2))", list_with_values(vec![Number(1), Number(2)])),
            ("(seq [])", Nil),
            ("(seq [1 2])", list_with_values(vec![Number(1), Number(2)])),
            ("(seq {})", Nil),
            (
                "(seq {1 2})",
                list_with_values(vec![vector_with_values(vec![Number(1), Number(2)])]),
            ),
            ("(seq #{})", Nil),
            ("(= (set '(1 2)) (set (seq #{1 2})))", Bool(true)),
        ];
        run_eval_test(&test_cases);
    }
}
