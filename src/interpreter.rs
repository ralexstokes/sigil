use crate::prelude::{
    self, count, deref, divide, equal, eval, greater, greater_eq, is_atom, is_empty, is_list, less,
    less_eq, list, multiply, plus, pr, prn, read_string, reset_atom, slurp, spit, subtract,
    swap_atom, to_atom, to_str,
};
use crate::reader::{read, ReaderError};
use crate::value::{var_impl_into_inner, var_into_inner, var_with_value, Lambda, Value};
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

pub type EvaluationResult<T> = Result<T, EvaluationError>;

fn lambda_parameter_key(index: usize, level: usize) -> String {
    let mut key = String::new();
    let _ = write!(&mut key, ":system-lambda-%{}/{}", index, level);
    key
}

#[derive(Debug, Error)]
pub enum SymbolEvaluationError {
    #[error("namespace `{0}` not found for symbol `{1}`")]
    UndefinedNamespace(String, String),
    #[error("var `{0}` not found in namespace `{1}`")]
    MissingVar(String, String),
}

#[derive(Debug, Error)]
pub enum ListEvaluationError {
    #[error("cannot invoke the supplied value {0}")]
    CannotInvoke(Value),
    #[error("some failure: {0}")]
    Failure(String),
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
    // contains at least one scope, the "default" or "core" scope
    scopes: Vec<Scope>,
}

const DEFAULT_NAMESPACE: &str = "sigil";

impl Default for Interpreter {
    fn default() -> Self {
        let bindings = &[
            ("+", Value::Primitive(plus)),
            ("-", Value::Primitive(subtract)),
            ("*", Value::Primitive(multiply)),
            ("/", Value::Primitive(divide)),
            ("pr", Value::Primitive(pr)),
            ("prn", Value::Primitive(prn)),
            ("list", Value::Primitive(list)),
            ("list?", Value::Primitive(is_list)),
            ("empty?", Value::Primitive(is_empty)),
            ("count", Value::Primitive(count)),
            ("<", Value::Primitive(less)),
            ("<=", Value::Primitive(less_eq)),
            (">", Value::Primitive(greater)),
            (">=", Value::Primitive(greater_eq)),
            ("=", Value::Primitive(equal)),
            ("read-string", Value::Primitive(read_string)),
            ("spit", Value::Primitive(spit)),
            ("slurp", Value::Primitive(slurp)),
            ("eval", Value::Primitive(eval)),
            ("str", Value::Primitive(to_str)),
            ("atom", Value::Primitive(to_atom)),
            ("atom?", Value::Primitive(is_atom)),
            ("deref", Value::Primitive(deref)),
            ("reset!", Value::Primitive(reset_atom)),
            ("swap!", Value::Primitive(swap_atom)),
        ];
        let global_scope =
            HashMap::from_iter(bindings.iter().map(|(k, v)| (k.to_string(), v.clone())));

        let mut default_namespaces = HashMap::new();
        default_namespaces.insert(DEFAULT_NAMESPACE.to_string(), Namespace::default());

        let mut interpreter = Interpreter {
            current_namespace: DEFAULT_NAMESPACE.to_string(),
            namespaces: default_namespaces,
            scopes: vec![global_scope],
        };
        for line in prelude::SOURCE.lines() {
            let forms = read(line).expect("prelude source has no reader errors");
            for form in forms.iter() {
                let _ = interpreter
                    .evaluate(form)
                    .expect("prelude forms have no evaluation errors");
            }
        }

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

fn intern_value_in_namespace(var_desc: &str, value: &Value, namespace: &mut Namespace) -> Value {
    match namespace.get(var_desc) {
        Some(var) => match var {
            Value::Var(v) => {
                *v.borrow_mut() = value.clone();
                var.clone()
            }
            _ => unreachable!(),
        },
        None => {
            let var = var_with_value(value.clone());
            namespace.insert(var_desc.to_string(), var.clone());
            var
        }
    }
}

impl Interpreter {
    pub fn intern_args(&mut self, args: Args) {
        let form = PersistentList::from_iter(args.map(|arg| Value::String(arg.clone())));
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
            _ => unreachable!(),
        }
    }

    pub fn current_namespace(&self) -> String {
        self.current_namespace.clone()
    }

    fn find_namespace(&mut self, ns_description: &str) -> Option<&mut Namespace> {
        self.namespaces.get_mut(ns_description)
    }

    fn intern_var(&mut self, identifier: &str, value: &Value) -> Value {
        let ns = self
            .namespaces
            .get_mut(&self.current_namespace())
            .expect("current namespace always resolves");
        intern_value_in_namespace(identifier, value, ns)
    }

    // return a ref to some var in the current namespace
    fn get_var(&self, identifier: &str) -> EvaluationResult<Value> {
        let ns = self
            .namespaces
            .get(&self.current_namespace())
            .expect("current namespace always resolves");
        get_var_in_namespace(identifier, ns).ok_or_else(|| {
            EvaluationError::Symbol(SymbolEvaluationError::MissingVar(
                identifier.to_string(),
                self.current_namespace.to_string(),
            ))
        })
    }

    fn resolve_var_in_namespace(
        &mut self,
        identifier: &str,
        ns_desc: &String,
    ) -> EvaluationResult<Value> {
        self.find_namespace(ns_desc)
            .ok_or_else(|| {
                // ns is `Some` but missing in `self`...
                let mut sym = String::new();
                let _ = write!(&mut sym, "{}/{}", &ns_desc, identifier);
                EvaluationError::Symbol(SymbolEvaluationError::UndefinedNamespace(
                    ns_desc.to_string(),
                    sym,
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

    fn resolve_symbol(
        &mut self,
        identifier: &str,
        ns_opt: Option<&String>,
    ) -> EvaluationResult<Value> {
        // if namespaced, check there
        if let Some(ns_desc) = ns_opt {
            return self
                .resolve_var_in_namespace(identifier, ns_desc)
                .map(var_into_inner);
        }
        // else resolve in lexical scopes
        if let Some(value) = resolve_symbol_in_scopes(self.scopes.iter().rev(), identifier) {
            return Ok(value.clone());
        }
        // otherwise check current namespace
        self.get_var(identifier).map(var_into_inner)
    }

    fn enter_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    fn insert_value_in_current_scope(&mut self, identifier: &str, value: Value) {
        let scope = self.scopes.last_mut().unwrap();
        scope.insert(identifier.to_string(), value);
    }

    fn leave_scope(&mut self) {
        // NOTE: error if scopes underflow...
        let _ = self.scopes.pop().unwrap();
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
            // NOTE: if we do not resolve vars here then lambdas will fail
            // if any captures are unmapped later...
            Value::Symbol(identifier, ns_opt) => {
                // if namespaced, check there
                if let Some(ns_desc) = ns_opt {
                    return self.resolve_var_in_namespace(identifier, ns_desc);
                }

                // else resolve in lexical scopes
                // here we resolve parameters to internal representations
                if let Some(value) = resolve_symbol_in_scopes(scopes.iter().rev(), identifier) {
                    return Ok(value.clone());
                }
                // otherwise check current namespace
                self.get_var(identifier)
                    // unresolved symbols are preserved to be interpreted later
                    .or_else(|_| Ok(Value::Symbol(identifier.to_string(), None)))
            }
            Value::List(elems) => {
                let mut analyzed_elems = vec![];
                for elem in elems.iter() {
                    let analyzed_elem = self.analyze_form_in_lambda(elem, scopes)?;
                    analyzed_elems.push(analyzed_elem);
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
            // Nil, Bool, Number, String, Keyword, Var
            other @ _ => Ok(other.clone()),
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
        let mut parameters = Scope::new();
        for (index, param) in params.iter().enumerate() {
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
        // walk the `forms`, resolving symbols where possible...
        lambda_scopes.push(parameters);
        let mut body = PersistentList::new();
        for form in forms.iter() {
            let analyzed_form = self.analyze_form_in_lambda(form, lambda_scopes)?;
            body.push_front_mut(analyzed_form);
        }
        lambda_scopes.pop();
        return Ok(Value::Fn(Lambda {
            body: body.push_front(Value::Symbol("do".to_string(), None)),
            arity: params.len(),
            level,
        }));
    }

    pub fn evaluate(&mut self, form: &Value) -> EvaluationResult<Value> {
        let result = match form {
            Value::Nil => Value::Nil,
            Value::Bool(b) => Value::Bool(*b),
            Value::Number(n) => Value::Number(*n),
            Value::String(s) => Value::String(s.to_string()),
            Value::Keyword(id, ns_opt) => {
                Value::Keyword(id.to_string(), ns_opt.as_ref().map(String::from))
            }
            Value::Symbol(id, ns_opt) => {
                return self.resolve_symbol(id, ns_opt.as_ref());
            }
            Value::List(forms) => {
                if let Some(operator_form) = forms.first() {
                    match operator_form {
                        // (def! symbol value)
                        Value::Symbol(s, None) if s == "def!" => {
                            if let Some(rest) = forms.drop_first() {
                                if let Some(identifier) = rest.first() {
                                    match identifier {
                                        Value::Symbol(id, None) => {
                                            if let Some(rest) = rest.drop_first() {
                                                if let Some(value_form) = rest.first() {
                                                    let value = self.evaluate(value_form)?;
                                                    let var = self.intern_var(id, &value);
                                                    return Ok(var);
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `def!`".to_string(),
                            )));
                        }
                        // (let* [bindings*] body)
                        Value::Symbol(s, None) if s == "let*" => {
                            if let Some(rest) = forms.drop_first() {
                                if let Some(bindings) = rest.first() {
                                    match bindings {
                                        Value::Vector(elems) => {
                                            if elems.len() % 2 == 0 {
                                                if let Some(body) = rest.drop_first() {
                                                    self.enter_scope();
                                                    for (name, value_form) in elems.iter().tuples()
                                                    {
                                                        match name {
                                                            Value::Symbol(s, None) => {
                                                                let value =
                                                                    self.evaluate(value_form)?;
                                                                self.insert_value_in_current_scope(
                                                                    s, value,
                                                                )
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
                                                    let form = body.push_front(Value::Symbol(
                                                        "do".to_string(),
                                                        None,
                                                    ));
                                                    let result = self.evaluate(&Value::List(form));
                                                    self.leave_scope();
                                                    return result;
                                                }
                                            }
                                        }
                                        _ => {}
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
                                if let Some(bindings) = rest.first() {
                                    match bindings {
                                        Value::Vector(elems) => {
                                            if elems.len() % 2 == 0 {
                                                if let Some(body) = rest.drop_first() {
                                                    if body.len() == 0 {
                                                        return Ok(Value::Nil);
                                                    }
                                                    // TODO: analyze loop*
                                                    // if recur, must be in tail position
                                                    self.enter_scope();
                                                    let mut bindings_keys = vec![];
                                                    for (name, value_form) in elems.iter().tuples()
                                                    {
                                                        match name {
                                                            Value::Symbol(s, None) => {
                                                                let value =
                                                                    self.evaluate(value_form)?;
                                                                bindings_keys.push(s.clone());
                                                                self.insert_value_in_current_scope(
                                                                    s, value,
                                                                )
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
                                                    let form = body.push_front(Value::Symbol(
                                                        "do".to_string(),
                                                        None,
                                                    ));
                                                    let form_to_eval = &Value::List(form);
                                                    let mut result = self.evaluate(form_to_eval);
                                                    while let Ok(Value::Recur(next_bindings)) =
                                                        result
                                                    {
                                                        if next_bindings.len()
                                                            != bindings_keys.len()
                                                        {
                                                            self.leave_scope();
                                                            return Err(EvaluationError::List(
                                                                    ListEvaluationError::Failure(
                                                                        "could not evaluate `loop*`: recur with incorrect number of bindings"
                                                                            .to_string(),
                                                                    ),
                                                                ));
                                                        }
                                                        for (key, value) in bindings_keys
                                                            .iter()
                                                            .zip(next_bindings.iter())
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
                                        _ => {}
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
                                return Ok(Value::Recur(PersistentVector::from_iter(
                                    result.into_iter(),
                                )));
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
                                                    } else {
                                                        if let Some(rest) = rest.drop_first() {
                                                            if let Some(alternate) = rest.first() {
                                                                return self.evaluate(alternate);
                                                            } else {
                                                                return Ok(Value::Nil);
                                                            }
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
                                                _ => {}
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
                                    .try_fold(Value::Nil, |_, next| self.evaluate(next));
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `do`".to_string(),
                            )));
                        }
                        // (fn* [parameters*] body)
                        Value::Symbol(s, None) if s == "fn*" => {
                            if let Some(rest) = forms.drop_first() {
                                if let Some(params) = rest.first() {
                                    match params {
                                        Value::Vector(params) => {
                                            if let Some(body) = rest.drop_first() {
                                                let mut lambda_scopes = vec![];
                                                return self.analyze_symbols_in_lambda(
                                                    body,
                                                    &params,
                                                    &mut lambda_scopes,
                                                );
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            return Err(EvaluationError::List(ListEvaluationError::Failure(
                                "could not evaluate `fn*`".to_string(),
                            )));
                        }
                        Value::Fn(Lambda { body, arity, level }) => {
                            if let Some(rest) = forms.drop_first() {
                                if rest.len() == *arity {
                                    self.enter_scope();
                                    for (index, operand_form) in rest.iter().enumerate() {
                                        match self.evaluate(operand_form) {
                                            Ok(operand) => {
                                                let parameter = lambda_parameter_key(index, *level);
                                                self.insert_value_in_current_scope(
                                                    &parameter, operand,
                                                );
                                            }
                                            Err(e) => {
                                                self.leave_scope();
                                                let mut error =
                                                    String::from("could not apply `fn*`: ");
                                                error += &e.to_string();
                                                return Err(EvaluationError::List(
                                                    ListEvaluationError::Failure(error),
                                                ));
                                            }
                                        }
                                    }
                                    let result = self.evaluate(&Value::List(body.clone()));
                                    self.leave_scope();
                                    return result;
                                }
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
                        _ => match self.evaluate(operator_form)? {
                            Value::Fn(Lambda { body, arity, level }) => {
                                if let Some(rest) = forms.drop_first() {
                                    if rest.len() == arity {
                                        self.enter_scope();
                                        for (index, operand_form) in rest.iter().enumerate() {
                                            match self.evaluate(operand_form) {
                                                Ok(operand) => {
                                                    let parameter =
                                                        lambda_parameter_key(index, level);
                                                    self.insert_value_in_current_scope(
                                                        &parameter, operand,
                                                    );
                                                }
                                                Err(e) => {
                                                    self.leave_scope();
                                                    let mut error =
                                                        String::from("could not apply `fn*`: ");
                                                    error += &e.to_string();
                                                    return Err(EvaluationError::List(
                                                        ListEvaluationError::Failure(error),
                                                    ));
                                                }
                                            }
                                        }
                                        let result = self.evaluate(&Value::List(body));
                                        self.leave_scope();
                                        return result;
                                    }
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
                            v @ _ => {
                                return Err(EvaluationError::List(
                                    ListEvaluationError::CannotInvoke(v),
                                ));
                            }
                        },
                    }
                }
                Value::List(PersistentList::new())
            }
            Value::Vector(forms) => {
                let mut result = vec![];
                for form in forms.into_iter() {
                    let value = self.evaluate(form)?;
                    result.push(value);
                }
                Value::Vector(PersistentVector::from_iter(result.into_iter()))
            }
            Value::Map(forms) => {
                let mut result = vec![];
                for (k, v) in forms.into_iter() {
                    let key = self.evaluate(k)?;
                    let value = self.evaluate(v)?;
                    result.push((key, value));
                }
                Value::Map(PersistentMap::from_iter(result.into_iter()))
            }
            Value::Set(forms) => {
                let mut result = vec![];
                for form in forms.into_iter() {
                    let value = self.evaluate(form)?;
                    result.push(value);
                }
                Value::Set(PersistentSet::from_iter(result.into_iter()))
            }
            Value::Fn(_) => unreachable!(),
            Value::Primitive(_) => unreachable!(),
            Value::Var(v) => var_impl_into_inner(v),
            Value::Recur(_) => unreachable!(),
            Value::Atom(_) => unreachable!(),
        };
        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::reader::read;
    use crate::value::{atom_with_value, list_with_values};
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
            ("(def! a 3)", var_with_value(Number(3))),
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
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_fn() {
        let test_cases = vec![
            ("((fn* [a] (+ a 1)) 23)", Number(24)),
            ("((fn* [a] (let* [b 2] (+ a b))) 23)", Number(25)),
            ("((fn* [a] (let* [a 2] (+ a a))) 23)", Number(4)),
            (
                "(def! inc (fn* [a] (+ a 1))) ((fn* [a] (inc a)) 1)",
                Number(2),
            ),
            ("((fn* [a] ((fn* [b] (+ b 1)) a)) 1)", Number(2)),
            ("((fn* [] ((fn* [] ((fn* [] 13))))))", Number(13)),
            ("(def! factorial (fn* [n] (if (< n 2) 1 (* n (factorial (- n 1)))))) (factorial 20)", Number(2432902008176640000)),
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
}
