use crate::prelude;
use crate::reader::{read, ReaderError};
use crate::value::{
    exception_from_thrown, exception_is_thrown, list_with_values, var_impl_into_inner,
    var_into_inner, var_with_value, FnImpl, Value,
};
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use rpds::{
    HashTrieMap as PersistentMap, HashTrieSet as PersistentSet, List as PersistentList,
    Vector as PersistentVector,
};
use std::collections::HashMap;
use std::convert::From;
use std::default::Default;
use std::env::Args;
use std::fmt::Write;
use std::iter::FromIterator;
use thiserror::Error;

const COMMAND_LINE_ARGS_SYMBOL: &str = "*command-line-args*";
const DEFAULT_NAMESPACE: &str = "core";
const DEFAULT_CORE_FILENAME: &str = "src/core.sigil";
const SPECIAL_FORMS: &[&str] = &[
    "def!",           // (def! symbol form)
    "let*",           // (let* [bindings*] form*)
    "loop*",          // (loop* [bindings*] form*)
    "recur",          // (recur form*)
    "if",             // (if predicate consequent alternate)
    "do",             // (do forms*)
    "fn*",            // (fn* [args] forms*)
    "quote",          // (quote form)
    "quasiquote",     // (quasiquote form)
    "unquote",        // (unquote form)
    "splice-unquote", // (splice-unquote form)
    "defmacro!",      // (defmacro! name fn*-form)
    "macroexpand",    // (macroexpand macro)
    "try*",           // (try* forms* catch*-form?)
    "catch*",         // (catch* exc-binding forms*)
];

pub type EvaluationResult<T> = Result<T, EvaluationError>;

fn lambda_parameter_key(index: usize, level: usize) -> String {
    let mut key = String::new();
    let _ = write!(&mut key, ":system-lambda-%{}/{}", index, level);
    key
}

#[derive(Debug, Error)]
pub enum SymbolEvaluationError {
    #[error("var `{0}` not found in namespace `{1}`")]
    MissingVar(String, String),
}

#[derive(Debug, Error)]
pub enum ListEvaluationError {
    #[error("cannot invoke the supplied value {0}")]
    CannotInvoke(Value),
    #[error("some failure: {0}")]
    Failure(String),
    #[error("error evaluating quasiquote: {0}")]
    QuasiquoteError(String),
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
}

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("symbol error: {0}")]
    Symbol(SymbolEvaluationError),
    #[error("list error: {0}")]
    List(ListEvaluationError),
    #[error("primitive error: {0}")]
    Primitive(PrimitiveEvaluationError),
    #[error("reader error: {0}")]
    ReaderError(ReaderError),
    #[error("interpreter error: {0}")]
    Interpreter(InterpreterError),
}

impl From<ReaderError> for EvaluationError {
    fn from(error: ReaderError) -> Self {
        Self::ReaderError(error)
    }
}

type Scope = HashMap<String, Value>;
// map from identifier to Value::Var
type Namespace = HashMap<String, Value>;

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
        for (symbol, value) in prelude::BINDINGS.iter() {
            intern_value_in_namespace(symbol, value, &mut default_namespace, DEFAULT_NAMESPACE);
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

        interpreter
    }
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
    value: &Value,
    namespace: &mut Namespace,
    namespace_desc: &str,
) -> Value {
    match namespace.get(var_desc) {
        Some(var) => match var {
            Value::Var(v) => {
                *v.inner.borrow_mut() = value.clone();
                var.clone()
            }
            _ => unreachable!(),
        },
        None => {
            let var = var_with_value(value.clone(), namespace_desc, var_desc);
            namespace.insert(var_desc.to_string(), var.clone());
            var
        }
    }
}

fn eval_quasiquote(value: &Value) -> EvaluationResult<Value> {
    match value {
        Value::List(elems) => {
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
                    _ => {
                        let mut result = Value::List(PersistentList::new());
                        for form in elems.reverse().iter() {
                            match form {
                                Value::List(inner) => {
                                    if let Some(first_inner) = inner.first() {
                                        match first_inner {
                                            Value::Symbol(s, None) if s == "splice-unquote" => {
                                                if let Some(rest) = inner.drop_first() {
                                                    if let Some(second) = rest.first() {
                                                        result = list_with_values(
                                                            [
                                                                Value::Symbol(
                                                                    "concat".to_string(),
                                                                    None,
                                                                ),
                                                                second.clone(),
                                                                result,
                                                            ]
                                                            .iter()
                                                            .cloned(),
                                                        );
                                                    }
                                                }
                                            }
                                            _ => {
                                                result = list_with_values(
                                                    [
                                                        Value::Symbol(
                                                            "cons".to_string(),
                                                            Some("core".to_string()),
                                                        ),
                                                        eval_quasiquote(form)?,
                                                        result,
                                                    ]
                                                    .iter()
                                                    .cloned(),
                                                );
                                            }
                                        }
                                    } else {
                                        result = list_with_values(
                                            [
                                                Value::Symbol(
                                                    "cons".to_string(),
                                                    Some("core".to_string()),
                                                ),
                                                Value::List(PersistentList::new()),
                                                result,
                                            ]
                                            .iter()
                                            .cloned(),
                                        );
                                    }
                                }
                                form => {
                                    result = list_with_values(
                                        [
                                            Value::Symbol(
                                                "cons".to_string(),
                                                Some("core".to_string()),
                                            ),
                                            eval_quasiquote(form)?,
                                            result,
                                        ]
                                        .iter()
                                        .cloned(),
                                    );
                                }
                            }
                        }
                        return Ok(result);
                    }
                }
            }
            Ok(Value::List(elems.clone()))
        }
        elem @ Value::Map(_) | elem @ Value::Symbol(_, _) => {
            let args = vec![Value::Symbol("quote".to_string(), None), elem.clone()];
            Ok(list_with_values(args.into_iter()))
        }
        v => Ok(v.clone()),
    }
}

impl Interpreter {
    pub fn intern_args(&mut self, args: Args) {
        let form = args.map(Value::String).collect();
        self.intern_var(COMMAND_LINE_ARGS_SYMBOL, &Value::List(form));
    }

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
            _ => panic!("programmer error to not intern command line args as a list"),
        }
    }

    pub fn current_namespace(&self) -> &str {
        &self.current_namespace
    }

    fn intern_var(&mut self, identifier: &str, value: &Value) -> Value {
        let current_namespace = self.current_namespace().to_string();

        let ns = self
            .namespaces
            .get_mut(&current_namespace)
            .expect("current namespace always resolves");
        intern_value_in_namespace(identifier, value, ns, &current_namespace)
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
            v @ Value::Var(_) => Ok(var_into_inner(v)),
            other => Ok(other),
        }
    }

    fn enter_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    fn insert_value_in_current_scope(&mut self, identifier: &str, value: Value) {
        let scope = self.scopes.last_mut().unwrap();
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
    ) -> EvaluationResult<Value> {
        match form {
            Value::Symbol(identifier, ns_opt) => {
                if let Some(value) = resolve_symbol_in_scopes(scopes.iter().rev(), identifier) {
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
                        match iter.next() {
                            Some(Value::Vector(bindings)) => {
                                let mut scope = Scope::new();
                                if bindings.len() % 2 != 0 {
                                    return Err(EvaluationError::List(
                                        ListEvaluationError::Failure(
                                            "could not evaluate `let*`".to_string(),
                                        ),
                                    ));
                                }
                                let mut analyzed_bindings = PersistentVector::new();
                                for (name, value) in bindings.iter().tuples() {
                                    scope.insert(name.to_string(), name.clone());
                                    let analyzed_value =
                                        self.analyze_form_in_lambda(value, scopes)?;
                                    analyzed_bindings.push_back_mut(name.clone());
                                    analyzed_bindings.push_back_mut(analyzed_value);
                                }
                                analyzed_elems.push(Value::Vector(analyzed_bindings));
                                scopes.push(scope);
                            }
                            _ => {}
                        }
                    }
                    Some(Value::Symbol(s, None)) if s == "fn*" => match iter.next() {
                        Some(Value::Vector(bindings)) => {
                            let rest = iter.cloned().collect();
                            return self.analyze_symbols_in_lambda(rest, bindings, scopes);
                        }
                        _ => {}
                    },
                    Some(Value::Symbol(s, None)) if s == "catch*" || s == "quote" => {
                        match iter.next() {
                            Some(Value::Symbol(s, None)) => {
                                let mut scope = Scope::new();
                                scope.insert(s.to_string(), Value::Symbol(s.to_string(), None));
                                scopes.push(scope);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
                for elem in elems.iter().skip(analyzed_elems.len()) {
                    let analyzed_elem = self.analyze_form_in_lambda(elem, scopes)?;
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
                    let analyzed_elem = self.analyze_form_in_lambda(elem, scopes)?;
                    analyzed_elems.push_back_mut(analyzed_elem);
                }
                Ok(Value::Vector(analyzed_elems))
            }
            Value::Map(elems) => {
                let mut analyzed_elems = PersistentMap::new();
                for (k, v) in elems.iter() {
                    let analyzed_k = self.analyze_form_in_lambda(k, scopes)?;
                    let analyzed_v = self.analyze_form_in_lambda(v, scopes)?;
                    analyzed_elems.insert_mut(analyzed_k, analyzed_v);
                }
                Ok(Value::Map(analyzed_elems))
            }
            Value::Set(elems) => {
                let mut analyzed_elems = PersistentSet::new();
                for elem in elems.iter() {
                    let analyzed_elem = self.analyze_form_in_lambda(elem, scopes)?;
                    analyzed_elems.insert_mut(analyzed_elem);
                }
                Ok(Value::Set(analyzed_elems))
            }
            Value::Fn(_) => unreachable!(),
            Value::Primitive(_) => unreachable!(),
            Value::Recur(_) => unreachable!(),
            // Nil, Bool, Number, String, Keyword, Var, Atom, Macro, Exception
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
        forms: PersistentList<Value>,
        params: &PersistentVector<Value>,
        lambda_scopes: &mut Vec<Scope>,
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
        // walk the `forms`, resolving symbols where possible...
        lambda_scopes.push(parameters);
        let mut body = vec![Value::Symbol("do".to_string(), None)];
        for form in forms.iter() {
            let analyzed_form = self.analyze_form_in_lambda(form, lambda_scopes)?;
            body.push(analyzed_form);
        }
        lambda_scopes.pop();
        Ok(Value::Fn(FnImpl {
            body: body.into_iter().collect(),
            arity,
            level,
            variadic,
        }))
    }

    fn apply_macro(
        &mut self,
        body: PersistentList<Value>,
        arity: usize,
        level: usize,
        variadic: bool,
        forms: &PersistentList<Value>,
    ) -> EvaluationResult<Value> {
        if let Some(args) = forms.drop_first() {
            let correct_arity = if variadic {
                args.len() >= arity
            } else {
                args.len() == arity
            };
            if !correct_arity {
                return Err(EvaluationError::List(ListEvaluationError::Failure(
                    "could not apply macro: incorrect arity".to_string(),
                )));
            }
            self.enter_scope();
            let mut iter = args.iter().enumerate();
            if arity > 0 {
                while let Some((index, operand_form)) = iter.next() {
                    let parameter = lambda_parameter_key(index, level);
                    self.insert_value_in_current_scope(&parameter, operand_form.clone());
                    if index == arity - 1 {
                        break;
                    }
                }
            }
            if variadic {
                let mut variadic_args = vec![];
                for (_, elem) in iter {
                    variadic_args.push(elem.clone());
                }
                let operand = Value::List(variadic_args.into_iter().collect());
                let parameter = lambda_parameter_key(arity, level);
                self.insert_value_in_current_scope(&parameter, operand);
            }
            let result = self.evaluate(&Value::List(body));
            self.leave_scope();
            match result? {
                Value::List(forms) => {
                    return self.macroexpand(&forms);
                }
                result => return Ok(result),
            }
        }
        return Err(EvaluationError::List(ListEvaluationError::Failure(
            "could not apply macro".to_string(),
        )));
    }

    fn macroexpand(&mut self, forms: &PersistentList<Value>) -> EvaluationResult<Value> {
        match forms.first() {
            Some(Value::Symbol(identifier, ns_opt)) => {
                // bail early on special forms
                if let Ok(Value::Macro(FnImpl {
                    body,
                    arity,
                    level,
                    variadic,
                })) = self.resolve_symbol(identifier, ns_opt.as_ref())
                {
                    return self.apply_macro(body, arity, level, variadic, forms);
                }
            }
            Some(v @ Value::Var(_)) => {
                if let Value::Macro(FnImpl {
                    body,
                    arity,
                    level,
                    variadic,
                }) = var_into_inner(v.clone())
                {
                    return self.apply_macro(body, arity, level, variadic, forms);
                }
            }
            _ => {}
        }
        Ok(Value::List(forms.clone()))
    }

    fn apply_lambda(
        &mut self,
        body: &PersistentList<Value>,
        arity: usize,
        level: usize,
        variadic: bool,
        args: &PersistentList<Value>,
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
                match self.evaluate(operand_form) {
                    Ok(operand) => {
                        let parameter = lambda_parameter_key(index, level);
                        self.insert_value_in_current_scope(&parameter, operand);
                    }
                    Err(e) => {
                        self.leave_scope();
                        let mut error = String::from("could not apply `fn*`: ");
                        error += &e.to_string();
                        return Err(EvaluationError::List(ListEvaluationError::Failure(error)));
                    }
                }
                if index == arity - 1 {
                    break;
                }
            }
        }
        if variadic {
            let mut variadic_args = vec![];
            for (_, elem) in iter {
                let elem = self.evaluate(elem)?;
                variadic_args.push(elem);
            }
            let operand = Value::List(variadic_args.into_iter().collect());
            let parameter = lambda_parameter_key(arity, level);
            self.insert_value_in_current_scope(&parameter, operand);
        }
        let result = self.evaluate(&Value::List(body.clone()));
        self.leave_scope();
        result
    }

    fn eval_list(&mut self, forms: &PersistentList<Value>) -> EvaluationResult<Value> {
        match self.macroexpand(forms)? {
            Value::List(forms) => {
                if let Some(operator_form) = forms.first() {
                    match operator_form {
                        // (def! symbol value)
                        Value::Symbol(s, None) if s == "def!" => {
                            if let Some(rest) = forms.drop_first() {
                                if let Some(Value::Symbol(id, None)) = rest.first() {
                                    if let Some(rest) = rest.drop_first() {
                                        if let Some(value_form) = rest.first() {
                                            // allocate the var first, so e.g. `fn`s can
                                            // capture them allowing for recursive calls
                                            let _ = self.intern_var(id, &Value::Nil);
                                            let value = self.evaluate(value_form)?;
                                            let var = self.intern_var(id, &value);
                                            return Ok(var);
                                        }
                                    }
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `def!`".to_string(),
                            )));
                        }
                        Value::Symbol(s, None) if s == "var" => {
                            if let Some(rest) = forms.drop_first() {
                                if let Some(Value::Symbol(s, ns_opt)) = rest.first() {
                                    if let Some(ns_desc) = ns_opt {
                                        return self.resolve_var_in_namespace(s, ns_desc);
                                    } else {
                                        return self.resolve_var_in_current_namespace(s);
                                    }
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `var`".to_string(),
                            )));
                        }
                        // (let* [bindings*] body)
                        Value::Symbol(s, None) if s == "let*" => {
                            if let Some(rest) = forms.drop_first() {
                                if let Some(Value::Vector(elems)) = rest.first() {
                                    if elems.len() % 2 == 0 {
                                        if let Some(body) = rest.drop_first() {
                                            self.enter_scope();
                                            for (name, value_form) in elems.iter().tuples() {
                                                match name {
                                                    Value::Symbol(s, None) => {
                                                        let value = self.evaluate(value_form)?;
                                                        self.insert_value_in_current_scope(s, value)
                                                    }
                                                    _ => {
                                                        self.leave_scope();
                                                        return Err(EvaluationError::List(
                                                            ListEvaluationError::Failure(
                                                                "could not evaluate `let*`"
                                                                    .to_string(),
                                                            ),
                                                        ));
                                                    }
                                                }
                                            }
                                            let form = body
                                                .push_front(Value::Symbol("do".to_string(), None));
                                            let result = self.evaluate(&Value::List(form));
                                            self.leave_scope();
                                            return result;
                                        }
                                    }
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `let*`".to_string(),
                            )));
                        }
                        // (loop* [bindings*] body)
                        Value::Symbol(s, None) if s == "loop*" => {
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
                                                                "could not evaluate `loop*`"
                                                                    .to_string(),
                                                            ),
                                                        ));
                                                    }
                                                }
                                            }
                                            let form = body
                                                .push_front(Value::Symbol("do".to_string(), None));
                                            let form_to_eval = &Value::List(form);
                                            let mut result = self.evaluate(form_to_eval);
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
                                                for (key, value) in
                                                    bindings_keys.iter().zip(next_bindings.iter())
                                                {
                                                    self.insert_value_in_current_scope(
                                                        key,
                                                        value.clone(),
                                                    );
                                                }
                                                result = self.evaluate(form_to_eval);
                                            }
                                            self.leave_scope();
                                            return result;
                                        }
                                    }
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `loop*`".to_string(),
                            )));
                        }
                        // (recur forms*)
                        Value::Symbol(s, None) if s == "recur" => {
                            if let Some(rest) = forms.drop_first() {
                                let mut result = vec![];
                                for form in rest.into_iter() {
                                    let value = self.evaluate(form)?;
                                    result.push(value);
                                }
                                return Ok(Value::Recur(result.into_iter().collect()));
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `recur`".to_string(),
                            )));
                        }
                        // (if predicate consequent alternate?)
                        Value::Symbol(s, None) if s == "if" => {
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
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `if`".to_string(),
                            )));
                        }
                        // (do forms*)
                        Value::Symbol(s, None) if s == "do" => {
                            if let Some(rest) = forms.drop_first() {
                                return rest
                                    .iter()
                                    .fold_while(Ok(Value::Nil), |_, next| {
                                        match self.evaluate(next) {
                                            Ok(e @ Value::Exception(_)) => {
                                                if exception_is_thrown(&e) {
                                                    Done(Ok(e))
                                                } else {
                                                    Continue(Ok(e))
                                                }
                                            }
                                            e @ Err(_) => Done(e),
                                            value => Continue(value),
                                        }
                                    })
                                    .into_inner();
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `do`".to_string(),
                            )));
                        }
                        // (fn* [parameters*] body)
                        Value::Symbol(s, None) if s == "fn*" => {
                            if let Some(rest) = forms.drop_first() {
                                if let Some(Value::Vector(params)) = rest.first() {
                                    if let Some(body) = rest.drop_first() {
                                        let mut scopes = vec![];
                                        return self.analyze_symbols_in_lambda(
                                            body,
                                            &params,
                                            &mut scopes,
                                        );
                                    }
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `fn*`".to_string(),
                            )));
                        }
                        // (quote form)
                        Value::Symbol(s, None) if s == "quote" => {
                            if let Some(rest) = forms.drop_first() {
                                if rest.len() == 1 {
                                    if let Some(form) = rest.first() {
                                        return Ok(form.clone());
                                    }
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `quote`".to_string(),
                            )));
                        }
                        // (quasiquote form)
                        Value::Symbol(s, None) if s == "quasiquote" => {
                            if let Some(rest) = forms.drop_first() {
                                if let Some(second) = rest.first() {
                                    let expansion = eval_quasiquote(second)?;
                                    return self.evaluate(&expansion);
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `recur`".to_string(),
                            )));
                        }
                        // (defmacro! symbol fn*)
                        Value::Symbol(s, None) if s == "defmacro!" => {
                            if let Some(rest) = forms.drop_first() {
                                if let Some(Value::Symbol(id, None)) = rest.first() {
                                    if let Some(rest) = rest.drop_first() {
                                        if let Some(value_form) = rest.first() {
                                            // allocate the var first, so e.g. `fn`s can
                                            // capture them allowing for recursive calls
                                            let _ = self.intern_var(id, &Value::Nil);
                                            match self.evaluate(value_form)? {
                                                Value::Fn(lambda) => {
                                                    let var =
                                                        self.intern_var(id, &Value::Macro(lambda));
                                                    return Ok(var);
                                                }
                                                _ => {
                                                    return Err(EvaluationError::List(
                                                        ListEvaluationError::Failure(
                                                            "could not evaluate `defmacro!`"
                                                                .to_string(),
                                                        ),
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `defmacro!`".to_string(),
                            )));
                        }
                        // (macroexpand value)
                        Value::Symbol(s, None) if s == "macroexpand" => {
                            if let Some(rest) = forms.drop_first() {
                                match rest.first() {
                                    Some(Value::List(value)) => return self.macroexpand(value),
                                    _ => {}
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `macroexpand`".to_string(),
                            )));
                        }
                        // apply phase when operator is already evaluated:
                        Value::Fn(FnImpl {
                            body,
                            arity,
                            level,
                            variadic,
                        }) => {
                            if let Some(rest) = forms.drop_first() {
                                return self.apply_lambda(body, *arity, *level, *variadic, &rest);
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not apply `fn*`".to_string(),
                            )));
                        }
                        Value::Primitive(native_fn) => {
                            let mut operands = vec![];
                            if let Some(rest) = forms.drop_first() {
                                for operand_form in rest.iter() {
                                    let operand = self.evaluate(operand_form)?;
                                    operands.push(operand);
                                }
                            }
                            return native_fn(self, &operands);
                        }
                        // (try* body (catch* e exc-body))
                        Value::Symbol(s, None) if s == "try*" => {
                            if let Some(rest) = forms.drop_first() {
                                let catch_form = match rest.last() {
                                    Some(Value::List(last_form)) => match last_form.first() {
                                        Some(Value::Symbol(s, None)) if s == "catch*" => {
                                            if let Some(catch_form) = last_form.drop_first() {
                                                if let Some(exception_symbol) = catch_form.first() {
                                                    match exception_symbol {
                                                        s @ Value::Symbol(_, None) => {
                                                            if let Some(exception_body) =
                                                                catch_form.drop_first()
                                                            {
                                                                let mut exception_binding =
                                                                    PersistentVector::new();
                                                                exception_binding
                                                                    .push_back_mut(s.clone());
                                                                let mut scopes = vec![];
                                                                let body = self
                                                                    .analyze_symbols_in_lambda(
                                                                        exception_body,
                                                                        &exception_binding,
                                                                        &mut scopes,
                                                                    )?;
                                                                Some(body)
                                                            } else {
                                                                None
                                                            }
                                                        }
                                                        _ => {
                                                            return Err(EvaluationError::List(
                                                                ListEvaluationError::Failure(
                                                                    "could not evaluate `catch*`"
                                                                        .to_string(),
                                                                ),
                                                            ));
                                                        }
                                                    }
                                                } else {
                                                    None
                                                }
                                            } else {
                                                return Err(EvaluationError::List(
                                                    ListEvaluationError::Failure(
                                                        "could not evaluate `catch*`".to_string(),
                                                    ),
                                                ));
                                            }
                                        }
                                        _ => None,
                                    },
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
                                let body =
                                    forms_to_eval.push_front(Value::Symbol("do".to_string(), None));
                                match self.evaluate(&Value::List(body))? {
                                    e @ Value::Exception(_) if exception_is_thrown(&e) => {
                                        if let Some(Value::Fn(FnImpl { body, level, .. })) =
                                            catch_form
                                        {
                                            self.enter_scope();
                                            let parameter = lambda_parameter_key(0, level);
                                            self.insert_value_in_current_scope(
                                                &parameter,
                                                exception_from_thrown(&e),
                                            );
                                            let result = self.evaluate(&Value::List(body));
                                            self.leave_scope();
                                            return result;
                                        } else {
                                            return Ok(e);
                                        }
                                    }
                                    result => return Ok(result),
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `try*`".to_string(),
                            )));
                        }
                        _ => match self.evaluate(operator_form)? {
                            Value::Fn(FnImpl {
                                body,
                                arity,
                                level,
                                variadic,
                            }) => {
                                if let Some(rest) = forms.drop_first() {
                                    return self.apply_lambda(&body, arity, level, variadic, &rest);
                                }
                                return Err(EvaluationError::List(ListEvaluationError::Failure(
                                    "could not apply `fn*`".to_string(),
                                )));
                            }
                            Value::Primitive(native_fn) => {
                                let mut operands = vec![];
                                if let Some(rest) = forms.drop_first() {
                                    for operand_form in rest.iter() {
                                        let operand = self.evaluate(operand_form)?;
                                        operands.push(operand);
                                    }
                                }
                                return native_fn(self, &operands);
                            }
                            v => {
                                return Err(EvaluationError::List(
                                    ListEvaluationError::CannotInvoke(v),
                                ));
                            }
                        },
                    }
                }
                Ok(Value::List(PersistentList::new()))
            }
            expansion => return self.evaluate(&expansion),
        }
    }

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
            Value::Fn(_) => unreachable!(),
            Value::Primitive(_) => unreachable!(),
            Value::Recur(_) => unreachable!(),
            Value::Atom(_) => unreachable!(),
            Value::Macro(_) => unreachable!(),
            Value::Exception(_) => unreachable!(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::reader::read;
    use crate::value::{
        atom_with_value, exception, list_with_values, map_with_values, vector_with_values,
    };
    use Value::*;

    fn run_eval_test(test_cases: &[(&str, Value)]) {
        for (input, expected) in test_cases {
            let mut interpreter = Interpreter::default();
            // dbg!(&input);
            let forms = read(input).unwrap();
            let mut final_result: Option<Value> = None;
            for form in forms.iter() {
                // dbg!(&form);
                let result = interpreter.evaluate(form).unwrap();
                // dbg!(&result);
                final_result = Some(result);
            }
            assert_eq!(final_result.unwrap(), *expected)
        }
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
            (":hi", Keyword("hi".to_string(), None)),
            (
                ":foo/hi",
                Keyword("hi".to_string(), Some("foo".to_string())),
            ),
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
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_do() {
        let test_cases = vec![("(do )", Nil), ("(do 1 2 3)", Number(3))];
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
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_let() {
        let test_cases = vec![
            ("(let* [] )", Nil),
            ("(let* [a 1] )", Nil),
            ("(let* [a 3] a)", Number(3)),
            ("(let* [a 3] (+ a 5))", Number(8)),
            ("(let* [a 3] (+ a (let* [c 5] c)))", Number(8)),
            ("(let* [a 3] (+ a (let* [a 5] a)))", Number(8)),
            ("(let* [a 3 b a] (+ b 5))", Number(8)),
            (
                "(let* [a 3 b 33] (+ a (let* [c 4] (+ c 1)) b 5))",
                Number(46),
            ),
            ("(def! a 1) (let* [a 3] a)", Number(3)),
            ("(def! a 1) (let* [a 3] a) a", Number(1)),
            ("(def! b 1) (let* [a 3] (+ a b))", Number(4)),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_if() {
        let test_cases = vec![
            ("(if true 1 2)", Number(1)),
            ("(if true 1)", Number(1)),
            ("(if false 1 2)", Number(2)),
            ("(if false 2)", Nil),
            ("(if nil 1 2)", Number(2)),
            ("(if nil 2)", Nil),
            ("(let* [b nil] (if b 2 3))", Number(3)),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_prelude() {
        let test_cases = vec![
            (
                "(list 1 2)",
                list_with_values([Number(1), Number(2)].iter().map(|arg| arg.clone())),
            ),
            ("(list? (list 1))", Bool(true)),
            ("(list? [1 2])", Bool(false)),
            ("(empty? (list))", Bool(true)),
            ("(empty? (list 1))", Bool(false)),
            ("(count (list 44 42 41))", Number(3)),
            ("(if (< 2 3) 12 13)", Number(12)),
            ("(<= 12 12)", Bool(true)),
            ("(<= 13 12)", Bool(false)),
            ("(<= 12 13)", Bool(true)),
            ("(>= 13 12)", Bool(true)),
            ("(>= 13 13)", Bool(true)),
            ("(>= 13 14)", Bool(false)),
            ("(= 12 12)", Bool(true)),
            ("(= 12 13)", Bool(false)),
            (
                "(read-string \"(+ 1 2)\")",
                List(PersistentList::from_iter(vec![
                    Symbol("+".to_string(), None),
                    Number(1),
                    Number(2),
                ])),
            ),
            ("(eval (list + 1 2 3))", Number(6)),
            ("(str \"hi\" 3 :foo)", String("hi3:foo".to_string())),
            ("(str \"hi   \" 3 :foo)", String("hi   3:foo".to_string())),
            (
                "(cons 1 (list))",
                list_with_values([Number(1)].iter().cloned()),
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
                "(vec '(1 2 3))",
                vector_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            (
                "(vec [1 2 3])",
                vector_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            ),
            ("(vec nil)", vector_with_values([].iter().cloned())),
            ("(nth [1 2 3] 2)", Number(3)),
            ("(nth '(1 2 3) 1)", Number(2)),
            ("(first '(1 2 3))", Number(1)),
            ("(first '())", Nil),
            ("(first [1 2 3])", Number(1)),
            ("(first [])", Nil),
            ("(first nil)", Nil),
            (
                "(rest '(1 2 3))",
                list_with_values([Number(2), Number(3)].iter().cloned()),
            ),
            ("(rest '())", List(PersistentList::new())),
            (
                "(rest [1 2 3])",
                list_with_values([Number(2), Number(3)].iter().cloned()),
            ),
            ("(rest [])", List(PersistentList::new())),
            ("(rest nil)", List(PersistentList::new())),
            ("(apply str [1 2 3])", String("123".to_string())),
            ("(apply str '(1 2 3))", String("123".to_string())),
            ("(apply str 0 1 2 '(1 2 3))", String("012123".to_string())),
            (
                "(def! inc (fn* [a] (+ a 1))) (map inc [1 2 3])",
                list_with_values([Number(2), Number(3), Number(4)].iter().cloned()),
            ),
            (
                "(def! inc (fn* [a] (+ a 1))) (map inc '(1 2 3))",
                list_with_values([Number(2), Number(3), Number(4)].iter().cloned()),
            ),
            ("(nil? nil)", Bool(true)),
            ("(nil? true)", Bool(false)),
            ("(nil? false)", Bool(false)),
            ("(nil? [1 2 3])", Bool(false)),
            ("(true? true)", Bool(true)),
            ("(true? nil)", Bool(false)),
            ("(true? false)", Bool(false)),
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
            ("(symbol? [1 2 3])", Bool(false)),
            ("(symbol \"hi\")", Symbol("hi".to_string(), None)),
            ("(keyword \"hi\")", Keyword("hi".to_string(), None)),
            ("(keyword? :a)", Bool(true)),
            ("(keyword? false)", Bool(false)),
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
            ("(vector? :hi)", Bool(false)),
            ("(sequential? '(1 2))", Bool(true)),
            ("(sequential? [1 2])", Bool(true)),
            ("(sequential? :hi)", Bool(false)),
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
            ("(get {:a 1} :a)", Number(1)),
            ("(get {:a 1} :b)", Nil),
            ("(contains? {:a 1} :b)", Bool(false)),
            ("(contains? {:a 1} :a)", Bool(true)),
            ("(keys {})", list_with_values([].iter().cloned())),
            // (
            //     "(keys {:a 1 :b 2 :c 3})",
            //     list_with_values(
            //         [
            //             Keyword("a".to_string(), None),
            //             Keyword("b".to_string(), None),
            //             Keyword("c".to_string(), None),
            //         ]
            //         .iter()
            //         .cloned(),
            //     ),
            // ),
            ("(vals {})", list_with_values([].iter().cloned())),
            // (
            //     "(vals {:a 1 :b 2 :c 3})",
            //     list_with_values([Number(1), Number(2), Number(3)].iter().cloned()),
            // ),
            ("(last '(1 2 3))", Number(3)),
            ("(last [1 2 3])", Number(3)),
            ("(last '())", Nil),
            ("(last [])", Nil),
            ("(not [])", Bool(false)),
            ("(not nil)", Bool(true)),
            ("(not false)", Bool(true)),
            ("(not 1)", Bool(false)),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_fn() {
        let test_cases = vec![
            ("((fn* [a] (+ a 1)) 23)", Number(24)),
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
            ("(def! f (fn* [a] (+ a 1))) (f 23)", Number(24)),
            (
                "(def! b 12) (def! f (fn* [a] (+ a b))) (def! b 22) (f 1)",
                Number(23),
            ),
            (
                "(def! b 12) (def! f (fn* [a] ((fn* [] (+ a b))))) (def! b 22) (f 1)",
                Number(23),
            ),
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
            ("(def! a (atom 5)) (deref a)", Number(5)),
            ("(def! a (atom 5)) @a", Number(5)),
            ("(def! a (atom (fn* [a] (+ a 1)))) (@a 4)", Number(5)),
            ("(def! a (atom 5)) (reset! a 10)", Number(10)),
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
            (
                "(def! lst '(b c)) `(a lst d)",
                read("(a lst d)")
                    .expect("example is correct")
                    .into_iter()
                    .nth(0)
                    .expect("some"),
            ),
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
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_try_catch() {
        let exc = exception(
            "test",
            &map_with_values(
                [(
                    Keyword("cause".to_string(), None),
                    String("no memory".to_string()),
                )]
                .iter()
                .cloned(),
            ),
        );
        let test_cases = vec![
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
            )
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
            ("((fn* [& rest] (count rest)) 1 2 3)", Number(3)),
            ("((fn* [a & rest] (count rest)) 1 2 3)", Number(2)),
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
}
