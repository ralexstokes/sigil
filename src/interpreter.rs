use crate::analyzer::{
    AnalysisError, AnalyzedForm, AnalyzedList, Analyzer, BodyForm, CatchError, CatchForm, DefForm,
    FnForm, IfForm, LetForm, LexicalBindings, LexicalForm, TryForm,
};
use crate::collections::{PersistentList, PersistentMap, PersistentSet, PersistentVector};
use crate::lang::core;
use crate::namespace::{Context as NamespaceContext, Namespace, NamespaceError, Var};
use crate::reader::{read, Form, Identifier, ReadError, Symbol};
use crate::value::{
    exception_from_system_err, list_with_values, var_impl_into_inner, ExceptionImpl, FnImpl,
    FnWithCapturesImpl, NativeFn, RuntimeValue, Value,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::default::Default;
use std::fmt::Write;
use std::iter::FromIterator;
use std::iter::IntoIterator;
use std::rc::Rc;
use std::time::SystemTimeError;
use std::{fmt, io};
use thiserror::Error;

const COMMAND_LINE_ARGS_SYMBOL: &str = "*command-line-args*";
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

#[derive(Debug, Error, Clone)]
pub enum InterpreterError {
    #[error("requested the {0}th command line arg but only {1} supplied")]
    MissingCommandLineArg(usize, usize),
    #[error("namespace {0} not found")]
    MissingNamespace(String),
    #[error("system time error: {0}")]
    SystemTimeError(#[from] SystemTimeError),
    #[error("io error: {0}")]
    IOError(IOErrorKindExt),
}

#[derive(Debug, Clone)]
// `IOErrorKindExt` exists to facilitate cloning of errors,
// which is necessary when wrapping system-level errors into
// user-level exceptions; `std::io::Error` does not implement
// `Clone` so we use the inner `std::io::ErrorKind` but hoist
// back into `std::io::Error` when implementing `std::fmt::Display`.
pub struct IOErrorKindExt(io::ErrorKind);

impl fmt::Display for IOErrorKindExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let err: io::Error = self.0.into();
        write!(f, "{}", err)
    }
}

impl From<io::Error> for InterpreterError {
    fn from(err: io::Error) -> Self {
        InterpreterError::IOError(IOErrorKindExt(err.kind()))
    }
}

#[derive(Debug, Error, Clone)]
pub enum SyntaxError {
    #[error("lexical bindings must be pairs of names and values but found unpaired set `{0}`")]
    LexicalBindingsMustBePaired(PersistentVector<Value>),
    #[error("expected vector of lexical bindings instead of `{0}`")]
    LexicalBindingsMustBeVector(Value),
    #[error("names in form must be non-namespaced symbols unlike `{0}`")]
    LexicalBindingsMustHaveSymbolNames(Value),
    #[error("missing argument for variadic binding")]
    VariadicArgMissing,
    #[error("found multiple variadic arguments in `{0}`; only one is allowed.")]
    VariadicArgMustBeUnique(Value),
}

#[derive(Debug, Error, Clone)]
pub enum EvaluationError {
    #[error("form invoked with an argument of the incorrect type: expected a value of type(s) `{expected}` but found value `{realized}`")]
    WrongType {
        expected: &'static str,
        realized: Value,
    },
    #[error("form invoked with incorrect arity: provided {realized} arguments but expected {expected} arguments")]
    WrongArity { expected: usize, realized: usize },
    #[error("var `{0}` not found in namespace `{1}`")]
    MissingVar(String, String),
    #[error("symbol `{0}` could not be resolved")]
    UnableToResolveSymbolToValue(String),
    #[error("cannot invoke the supplied value `{0}`")]
    CannotInvoke(Value),
    #[error("missing value for captured symbol `{0}`")]
    MissingCapturedValue(String),
    #[error("cannot deref an unbound var `{0}`")]
    CannotDerefUnboundVar(Value),
    #[error("overflow detected during arithmetic operation of {0} and {1}")]
    Overflow(i64, i64),
    #[error("could not negate {0}")]
    Negation(i64),
    #[error("underflow detected during arithmetic operation of {0} and {1}")]
    Underflow(i64, i64),
    #[error("requested index {0} in collection with length {1}")]
    IndexOutOfBounds(usize, usize),
    #[error("map cannot be constructed with an odd number of arguments: `{0}` with length `{1}`")]
    MapRequiresPairs(Value, usize),
    #[error("exception: {0}")]
    Exception(ExceptionImpl),
    #[error("syntax error: {0}")]
    Syntax(#[from] SyntaxError),
    #[error("interpreter error: {0}")]
    Interpreter(#[from] InterpreterError),
    // TODO: clean up errors above ~here
    #[error("namespace error: {0}")]
    Namespace(#[from] NamespaceError),
    #[error("reader error: {0}")]
    ReaderError(ReadError, String),
    #[error("analysis error: {0}")]
    AnalysisError(#[from] AnalysisError),
}

pub type EvaluationResult<T> = Result<T, EvaluationError>;
pub type SymbolIndex = HashSet<String>;
pub type Scope<'a> = HashMap<&'a Identifier, RuntimeValue>;

// fn lambda_parameter_key(index: usize, level: usize) -> String {
//     let mut key = String::new();
//     let _ = write!(&mut key, ":system-fn-%{}/{}", index, level);
//     key
// }

// `scopes` from most specific to least specific
fn resolve_symbol_in_scopes<'a>(
    scopes: impl Iterator<Item = &'a Scope<'a>>,
    identifier: &str,
) -> Option<&'a Value> {
    for scope in scopes {
        if let Some(value) = scope.get(identifier) {
            return Some(value);
        }
    }
    None
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
                                return Err(EvaluationError::WrongArity {
                                    expected: 1,
                                    realized: 0,
                                });
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
                return Err(EvaluationError::WrongArity {
                    realized: 0,
                    expected: 1,
                });
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

// TODO fix types
fn eval_quasiquote(form: AnalyzedForm) -> EvaluationResult<RuntimeValue> {
    match form {
        Value::List(elems) => eval_quasiquote_list(elems),
        Value::Vector(elems) => eval_quasiquote_vector(elems),
        elem @ Value::Map(_) | elem @ Value::Symbol(..) => {
            let args = vec![Value::Symbol("quote".to_string(), None), elem.clone()];
            Ok(list_with_values(args.into_iter()))
        }
        v => Ok(v.clone()),
    }
}

fn do_to_exactly_one_arg<A>(
    operand_forms: PersistentList<Value>,
    mut action: A,
) -> EvaluationResult<Value>
where
    A: FnMut(&Value) -> EvaluationResult<Value>,
{
    if operand_forms.len() != 1 {
        return Err(EvaluationError::WrongArity {
            expected: 1,
            realized: operand_forms.len(),
        });
    }
    let arg = operand_forms.first().unwrap();
    action(arg)
}

fn update_captures(
    captures: &mut HashMap<String, Option<Value>>,
    scopes: &[Scope],
) -> EvaluationResult<()> {
    for (capture, value) in captures {
        if value.is_none() {
            let captured_value = resolve_symbol_in_scopes(scopes.iter().rev(), capture)
                .ok_or_else(|| {
                    EvaluationError::UnableToResolveSymbolToValue(capture.to_string())
                })?;
            *value = Some(captured_value.clone());
        }
    }
    Ok(())
}

enum ControlFlow {
    Continue,
    Recur(Vec<RuntimeValue>),
}

#[derive(Debug)]
pub struct Interpreter {
    namespaces: NamespaceContext,
    symbol_index: Option<Rc<RefCell<SymbolIndex>>>,

    // stack of scopes
    // contains at least one scope, the "default" scope
    pub(crate) scopes: Vec<Scope>,
    control_stack: Vec<ControlFlow>,

    // low-res backtrace
    pub(crate) apply_stack: Vec<Value>,
    // index into `apply_stack` pointing at the first form to error
    failed_form: Option<usize>,
}

impl Default for Interpreter {
    fn default() -> Self {
        // build the default scope, which resolves special forms to themselves
        // so that they fall through to the interpreter's evaluation
        let mut default_scope = Scope::new();
        for form in SPECIAL_FORMS {
            default_scope.insert(form.to_string(), Value::Symbol(form.to_string(), None));
        }

        let mut interpreter = Interpreter {
            namespaces: NamespaceContext::default(),
            symbol_index: None,
            scopes: vec![default_scope],
            control_stack: vec![],
            apply_stack: vec![],
            failed_form: None,
        };

        // load the "core" namespace
        interpreter
            .activate_namespace(core::loader)
            .expect("is valid namespace");

        // add support for `*command-line-args*`
        let mut buffer = String::new();
        let _ = write!(&mut buffer, "(def! {} '())", COMMAND_LINE_ARGS_SYMBOL)
            .expect("can write to string");
        interpreter.interpret(&buffer).expect("valid source");

        interpreter
    }
}

pub type NamespaceLoader = fn(&mut Interpreter) -> EvaluationResult<()>;

impl Interpreter {
    pub fn activate_namespace(&mut self, loader: NamespaceLoader) -> EvaluationResult<()> {
        loader(self)
    }

    pub fn register_symbol_index(&mut self, symbol_index: Rc<RefCell<SymbolIndex>>) {
        let mut index = symbol_index.borrow_mut();
        // TODO: fixme
        // for namespace in self.namespaces.values() {
        //     for symbol in namespace.symbols() {
        //         index.insert(symbol.clone());
        //     }
        // }
        drop(index);

        self.symbol_index = Some(symbol_index);
    }

    // Returns the name of the loaded namespace
    pub fn load_namespace(&mut self, namespace: Namespace) -> EvaluationResult<()> {
        let key = &namespace.name;
        if let Some(existing) = self.namespaces.get_mut(key) {
            existing.merge(&namespace)?;
        } else {
            self.namespaces.insert(key.clone(), namespace);
        }
        Ok(())
    }

    /// Store `args` in the var referenced by `COMMAND_LINE_ARGS_SYMBOL`.
    pub fn intern_args(&mut self, args: impl Iterator<Item = String>) {
        let form = args.map(Value::String).collect();
        self.intern_var(COMMAND_LINE_ARGS_SYMBOL, Value::List(form))
            .expect("'*command-line-args* constructed correctly");
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

    fn intern(&mut self, symbol: &Symbol, value: Option<RuntimeValue>) -> EvaluationResult<Var> {
        let namespace = match symbol.namespace {
            Some(ns) => self.namespaces.get_namespace(&ns),
            None => self.namespaces.current_namespace(),
        };
        let var = namespace.intern(&symbol.identifier, value)?;

        // TODO update indexing...
        if let Some(index) = &self.symbol_index {
            let mut index = index.borrow_mut();
            index.insert(symbol.identifier.clone());
        }
        Ok(var)
    }

    fn unintern_var(&mut self, identifier: &str) {
        let current_namespace = self.current_namespace().to_string();

        let ns = self
            .namespaces
            .get_mut(&current_namespace)
            .expect("current namespace always resolves");
        ns.remove(identifier);
    }

    // return a ref to some var in the current namespace
    // fn resolve_var_in_current_namespace(&self, identifier: &str) -> EvaluationResult<Value> {
    //     let ns_desc = self.current_namespace();
    //     self.resolve_var_in_namespace(identifier, ns_desc)
    // }

    // namespace -> var
    // fn resolve_var_in_namespace(&self, identifier: &str, ns_desc: &str) -> EvaluationResult<Value> {
    //     self.namespaces
    //         .get(ns_desc)
    //         .ok_or_else(|| {
    //             EvaluationError::Interpreter(InterpreterError::MissingNamespace(
    //                 ns_desc.to_string(),
    //             ))
    //         })
    //         .and_then(|ns| {
    //             ns.get(identifier).cloned().ok_or_else(|| {
    //                 EvaluationError::MissingVar(identifier.to_string(), ns_desc.to_string())
    //             })
    //         })
    // }

    // symbol -> namespace -> var
    pub(crate) fn resolve_symbol_to_var(
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
            Value::Var(v) => match var_impl_into_inner(&v) {
                Some(value) => Ok(value),
                None => Ok(Value::Var(v)),
            },
            other => Ok(other),
        }
    }

    // fn enter_scope(&mut self) {
    //     self.scopes.push(Scope::default());
    // }

    // fn insert_value_in_current_scope(&mut self, identifier: &str, value: Value) {
    //     let scope = self.scopes.last_mut().expect("always one scope");
    //     scope.insert(identifier.to_string(), value);
    // }

    /// Exits the current lexical scope.
    /// NOTE: exposed for some prelude functionality.
    pub fn leave_scope(&mut self) {
        let _ = self.scopes.pop().expect("no underflow in scope stack");
    }

    fn apply_macro(
        &mut self,
        f: &FnImpl,
        operands: &PersistentList<Value>,
    ) -> EvaluationResult<Value> {
        let result = self.apply_fn_inner(f, operands, operands.len())?;
        if let Value::List(forms) = result {
            return self.expand_macro_if_present(&forms);
        }
        Ok(result)
    }

    fn expand_macro_if_present(
        &mut self,
        forms: &PersistentList<Value>,
    ) -> EvaluationResult<Value> {
        if let Some(first) = forms.first() {
            let rest = forms.drop_first().expect("list is not empty");
            if let Some(expansion) = self.get_macro_expansion(first, &rest) {
                expansion
            } else {
                Ok(Value::List(forms.clone()))
            }
        } else {
            Ok(Value::List(PersistentList::new()))
        }
    }

    /// Apply the given `Fn` to the supplied `args`.
    /// Exposed for various `prelude` functions.
    pub(crate) fn apply_fn_inner<'a>(
        &mut self,
        FnImpl {
            body,
            arity,
            level,
            variadic,
        }: &FnImpl,
        args: impl IntoIterator<Item = &'a Value>,
        args_count: usize,
    ) -> EvaluationResult<Value> {
        let arity = *arity;
        let level = *level;
        let variadic = *variadic;

        let correct_arity = if variadic {
            args_count >= arity
        } else {
            args_count == arity
        };
        if !correct_arity {
            return Err(EvaluationError::WrongArity {
                expected: arity,
                realized: args_count,
            });
        }
        self.enter_scope();
        let mut iter = args.into_iter().enumerate();
        if arity > 0 {
            for (index, arg) in &mut iter {
                let parameter = lambda_parameter_key(index, level);
                self.insert_value_in_current_scope(&parameter, arg.clone());

                if index == arity - 1 {
                    break;
                }
            }
        }
        if variadic {
            let operand = Value::List(iter.map(|(_, arg)| arg.clone()).collect());
            let parameter = lambda_parameter_key(arity, level);
            self.insert_value_in_current_scope(&parameter, operand);
        }
        let mut result = self.eval_do_inner(body);
        if let Ok(Value::FnWithCaptures(FnWithCapturesImpl { f, mut captures })) = result {
            update_captures(&mut captures, &self.scopes)?;
            result = Ok(Value::FnWithCaptures(FnWithCapturesImpl { f, captures }))
        }
        self.leave_scope();
        result
    }

    fn apply_fn(
        &mut self,
        f: &FnImpl,
        operand_forms: PersistentList<Value>,
    ) -> EvaluationResult<Value> {
        let mut args = Vec::with_capacity(operand_forms.len());
        for form in &operand_forms {
            let result = self.evaluate_form(form)?;
            args.push(result);
        }
        self.apply_fn_inner(f, &args, args.len())
    }

    fn apply_primitive(
        &mut self,
        native_fn: NativeFn,
        operand_forms: PersistentList<Value>,
    ) -> EvaluationResult<Value> {
        let mut operands = vec![];
        for operand_form in &operand_forms {
            let operand = self.evaluate_form(operand_form)?;
            operands.push(operand);
        }
        native_fn(self, &operands)
    }

    pub fn extend_from_captures(
        &mut self,
        captures: &HashMap<String, Option<Value>>,
    ) -> EvaluationResult<()> {
        self.enter_scope();
        for (capture, value) in captures {
            if let Some(value) = value {
                self.insert_value_in_current_scope(capture, value.clone());
            } else {
                self.leave_scope();
                return Err(EvaluationError::MissingCapturedValue(capture.to_string()));
            }
        }
        Ok(())
    }

    // fn eval_def_inner(&mut self, id: &str, value_form: &Value) -> EvaluationResult<Value> {
    //     // need to only adjust var if this `def!` is successful
    //     // also optimistically allocate in the interpreter so that
    //     // the def body can capture references to itself (e.g. for recursive fn)
    //     //
    //     // to address this:
    //     // get the existing var, or intern a sentinel value if it is missing
    //     let (var, var_already_exists) = match self.resolve_var_in_current_namespace(id) {
    //         Ok(v @ Value::Var(..)) => (v, true),
    //         Err(EvaluationError::MissingVar(..)) => (self.intern_unbound_var(id)?, false),
    //         e @ Err(_) => return e,
    //         _ => unreachable!(),
    //     };
    //     let value = self.evaluate_form(value_form).map_err(|err| {
    //         // and if the evaluation is not ok,
    //         if !var_already_exists {
    //             // and the var did not already exist, unintern the sentinel allocation
    //             self.unintern_var(id);
    //         }
    //         // (if the var did already exist, then simply leave alone)
    //         err
    //     })?;
    //     // and if the evaluation is ok, unconditionally update the var
    //     match &var {
    //         Value::Var(var) => var.update(value),
    //         _ => unreachable!(),
    //     }
    //     Ok(var)
    // }

    // fn eval_def_legacy(&mut self, operand_forms: PersistentList<Value>) -> EvaluationResult<Value> {
    //     // if !(operand_forms.len() == 1 || operand_forms.len() == 2) {
    //     //     return Err(EvaluationError::WrongArity {
    //     //         expected: 2,
    //     //         realized: operand_forms.len(),
    //     //     });
    //     // }
    //     // let name_form = operand_forms.first().unwrap();
    //     // let rest = operand_forms.drop_first().expect("list is not empty");
    //     // match name_form {
    //     //     Value::Symbol(id, None) => {
    //     //         if rest.is_empty() {
    //     //             return self.intern_unbound_var(id);
    //     //         }
    //     //         let value_form = rest.first().unwrap();
    //     //         self.eval_def_inner(id, value_form)
    //     //     }
    //     //     other => Err(EvaluationError::WrongType {
    //     //         expected: "SymbolWithoutNamespace",
    //     //         realized: other.clone(),
    //     //     }),
    //     // }
    // }

    // fn eval_var_legacy(&mut self, operand_forms: PersistentList<Value>) -> EvaluationResult<Value> {
    //     if operand_forms.len() != 1 {
    //         return Err(EvaluationError::WrongArity {
    //             expected: 1,
    //             realized: operand_forms.len(),
    //         });
    //     }
    //     let name_form = operand_forms.first().unwrap();
    //     match name_form {
    //         Value::Symbol(s, ns_opt) => {
    //             if let Some(ns_desc) = ns_opt {
    //                 self.resolve_var_in_namespace(s, ns_desc)
    //             } else {
    //                 self.resolve_var_in_current_namespace(s)
    //             }
    //         }
    //         other => Err(EvaluationError::WrongType {
    //             expected: "Symbol",
    //             realized: other.clone(),
    //         }),
    //     }
    // }

    // fn eval_let_legacy(&mut self, operand_forms: PersistentList<Value>) -> EvaluationResult<Value> {
    //     let LetForm { bindings, body } = analyze_let(&operand_forms)?;
    //     let forward_declarations = bindings.resolve_forward_declarations();
    //     if !forward_declarations.is_empty() {
    //         self.enter_scope();
    //         for identifier in &forward_declarations {
    //             let var = unbound_var("", identifier);
    //             self.insert_value_in_current_scope(identifier, var);
    //         }
    //     }
    //     self.enter_scope();
    //     for (identifier, value_form) in bindings {
    //         match self.evaluate_form(value_form) {
    //             Ok(value) => {
    //                 if let Some(Value::Var(var)) =
    //                     resolve_symbol_in_scopes(self.scopes.iter().rev(), identifier)
    //                 {
    //                     var.update(value);
    //                 } else {
    //                     self.insert_value_in_current_scope(identifier, value);
    //                 }
    //             }
    //             e @ Err(_) => {
    //                 self.leave_scope();
    //                 if !forward_declarations.is_empty() {
    //                     self.leave_scope();
    //                 }
    //                 return e;
    //             }
    //         }
    //     }
    //     let result = self.eval_do_inner(&body);
    //     self.leave_scope();
    //     if !forward_declarations.is_empty() {
    //         self.leave_scope();
    //     }
    //     result
    // }

    // fn eval_loop_legacy(
    //     &mut self,
    //     operand_forms: PersistentList<Value>,
    // ) -> EvaluationResult<Value> {
    //     let LetForm { bindings, body } = analyze_let(&operand_forms)?;
    //     self.enter_scope();
    //     let mut bindings_keys = vec![];
    //     for (name, value_form) in bindings.into_iter() {
    //         let value = self.evaluate_form(value_form)?;
    //         bindings_keys.push(name);
    //         self.insert_value_in_current_scope(name, value)
    //     }
    //     let mut result = self.eval_do_inner(&body);
    //     while let Ok(Value::Recur(next_bindings)) = result {
    //         if next_bindings.len() != bindings_keys.len() {
    //             self.leave_scope();
    //             return Err(EvaluationError::WrongArity {
    //                 expected: bindings_keys.len(),
    //                 realized: next_bindings.len(),
    //             });
    //         }
    //         for (key, value) in bindings_keys.iter().zip(next_bindings.iter()) {
    //             self.insert_value_in_current_scope(key, value.clone());
    //         }
    //         result = self.eval_do_inner(&body);
    //     }
    //     self.leave_scope();
    //     result
    // }

    // fn eval_recur_legacy(
    //     &mut self,
    //     operand_forms: PersistentList<Value>,
    // ) -> EvaluationResult<Value> {
    //     let mut result = PersistentVector::new();
    //     for form in operand_forms.into_iter() {
    //         let value = self.evaluate_form(form)?;
    //         result.push_back_mut(value);
    //     }
    //     Ok(Value::Recur(result))
    // }

    // fn eval_if_legacy(&mut self, operand_forms: PersistentList<Value>) -> EvaluationResult<Value> {
    //     if !(operand_forms.len() == 2 || operand_forms.len() == 3) {
    //         return Err(EvaluationError::WrongArity {
    //             expected: 2,
    //             realized: operand_forms.len(),
    //         });
    //     }
    //     let predicate_form = operand_forms.first().unwrap();
    //     let predicate = self.evaluate_form(predicate_form)?;
    //     let falsey = matches!(predicate, Value::Nil | Value::Bool(false));
    //     let rest = operand_forms.drop_first().expect("list is not empty");
    //     let consequent_form = rest.first().unwrap();
    //     match rest.len() {
    //         2 => {
    //             if !falsey {
    //                 self.evaluate_form(consequent_form)
    //             } else {
    //                 let rest = rest.drop_first().expect("list is not empty");
    //                 let alternate_form = rest.first().unwrap();
    //                 self.evaluate_form(alternate_form)
    //             }
    //         }
    //         1 => {
    //             if !falsey {
    //                 self.evaluate_form(consequent_form)
    //             } else {
    //                 Ok(Value::Nil)
    //             }
    //         }
    //         _ => unreachable!("validated len of `if` form"),
    //     }
    // }

    // fn eval_do_inner(&mut self, forms: &PersistentList<Value>) -> EvaluationResult<Value> {
    //     forms
    //         .iter()
    //         .try_fold(Value::Nil, |_, next| self.evaluate_form(next))
    // }

    // fn eval_do(&mut self, operand_forms: PersistentList<Value>) -> EvaluationResult<Value> {
    //     self.eval_do_inner(&operand_forms)
    // }

    // fn eval_fn_legacy(&mut self, operand_forms: PersistentList<Value>) -> EvaluationResult<Value> {
    //     if operand_forms.is_empty() {
    //         return Err(EvaluationError::WrongArity {
    //             expected: 1,
    //             realized: 0,
    //         });
    //     }
    //     let params_form = operand_forms.first().unwrap();
    //     let body = operand_forms.drop_first().expect("list is not empty");
    //     match params_form {
    //         Value::Vector(params) => analyze_fn(self, body, params),
    //         other => Err(SyntaxError::LexicalBindingsMustBeVector(other.clone()).into()),
    //     }
    // }

    // fn eval_quote_legacy(
    //     &mut self,
    //     operand_forms: PersistentList<Value>,
    // ) -> EvaluationResult<Value> {
    //     if operand_forms.len() != 1 {
    //         return Err(EvaluationError::WrongArity {
    //             expected: 1,
    //             realized: operand_forms.len(),
    //         });
    //     }
    //     Ok(operand_forms.first().cloned().unwrap())
    // }

    // fn eval_quasiquote(&mut self, operand_forms: PersistentList<Value>) -> EvaluationResult<Value> {
    //     if operand_forms.len() != 1 {
    //         return Err(EvaluationError::WrongArity {
    //             expected: 1,
    //             realized: operand_forms.len(),
    //         });
    //     }
    //     let operand_form = operand_forms.first().unwrap();
    //     let expansion = eval_quasiquote(operand_form)?;
    //     self.evaluate_form(&expansion)
    // }

    fn eval_defmacro_legacy(
        &mut self,
        operand_forms: PersistentList<Value>,
    ) -> EvaluationResult<Value> {
        match self.eval_def(operand_forms)? {
            Value::Var(var) => match var_impl_into_inner(&var) {
                Some(Value::Fn(f)) => {
                    var.update(Value::Macro(f));
                    Ok(Value::Var(var))
                }
                Some(other) => {
                    self.unintern_var(&var.identifier);
                    Err(EvaluationError::WrongType {
                        expected: "Fn",
                        realized: other,
                    })
                }
                None => {
                    self.unintern_var(&var.identifier);
                    Err(EvaluationError::WrongType {
                        expected: "Fn",
                        realized: Value::Var(var),
                    })
                }
            },
            _ => unreachable!("eval def only returns Value::Var"),
        }
    }

    fn eval_macroexpand(
        &mut self,
        operand_forms: PersistentList<Value>,
    ) -> EvaluationResult<Value> {
        do_to_exactly_one_arg(operand_forms, |arg| match self.evaluate_form(arg)? {
            Value::List(elems) => self.expand_macro_if_present(&elems),
            other => Ok(other),
        })
    }

    fn eval_try(&mut self, operand_forms: PersistentList<Value>) -> EvaluationResult<Value> {
        let catch_form = match operand_forms.last() {
            Some(Value::List(last_form)) => match last_form.first() {
                Some(Value::Symbol(s, None)) if s == "catch*" => {
                    // FIXME: deduplicate analysis of `catch*` here...
                    if let Some(catch_form) = last_form.drop_first() {
                        if let Some(exception_symbol) = catch_form.first() {
                            match exception_symbol {
                                s @ Value::Symbol(_, None) => {
                                    // if let Some(exception_body) = catch_form.drop_first() {
                                    //     let mut exception_binding = PersistentVector::new();
                                    //     exception_binding.push_back_mut(s.clone());
                                    //     let body =
                                    //         analyze_fn(self, exception_body, &exception_binding)?;
                                    //     Some(body)
                                    // } else {
                                    //     None
                                    // }
                                    None
                                }
                                other => {
                                    return Err(SyntaxError::LexicalBindingsMustHaveSymbolNames(
                                        other.clone(),
                                    )
                                    .into());
                                }
                            }
                        } else {
                            None
                        }
                    } else {
                        return Err(EvaluationError::WrongArity {
                            expected: 2,
                            realized: 0,
                        });
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
            operand_forms
        } else {
            let mut forms_to_eval = vec![];
            for (index, form) in operand_forms.iter().enumerate() {
                if index == operand_forms.len() - 1 {
                    break;
                }
                forms_to_eval.push(form.clone());
            }
            PersistentList::from_iter(forms_to_eval)
        };
        let apply_stack_pointer = self.apply_stack.len();
        match self.eval_do_inner(&forms_to_eval) {
            Ok(result) => Ok(result),
            Err(err) => match catch_form {
                Some(Value::Fn(FnImpl { body, level, .. })) => {
                    self.failed_form.take();
                    self.apply_stack.truncate(apply_stack_pointer);
                    self.enter_scope();
                    let parameter = lambda_parameter_key(0, level);
                    self.insert_value_in_current_scope(&parameter, exception_from_system_err(err));
                    let result = self.eval_do_inner(&body);
                    self.leave_scope();
                    result
                }
                Some(Value::FnWithCaptures(FnWithCapturesImpl {
                    f: FnImpl { body, level, .. },
                    mut captures,
                })) => {
                    self.failed_form.take();
                    self.apply_stack.truncate(apply_stack_pointer);
                    // FIXME: here we pull values from scopes just to turn around and put them back in a child scope.
                    // Can we skip this?
                    update_captures(&mut captures, &self.scopes)?;
                    self.extend_from_captures(&captures)?;
                    self.enter_scope();
                    let parameter = lambda_parameter_key(0, level);
                    self.insert_value_in_current_scope(&parameter, exception_from_system_err(err));
                    let result = self.eval_do_inner(&body);
                    self.leave_scope();
                    self.leave_scope();
                    result
                }
                None => Err(err),
                _ => unreachable!("`catch*` form yields callable or nothing via syntax analysis"),
            },
        }
    }

    pub(crate) fn get_macro_expansion(
        &mut self,
        operator: &Value,
        operands: &PersistentList<Value>,
    ) -> Option<EvaluationResult<Value>> {
        match operator {
            Value::Symbol(identifier, ns_opt) => {
                if let Ok(Value::Macro(f)) = self.resolve_symbol(identifier, ns_opt.as_ref()) {
                    Some(self.apply_macro(&f, operands))
                } else {
                    None
                }
            }
            Value::Var(v) => {
                if let Some(Value::Macro(f)) = var_impl_into_inner(v) {
                    Some(self.apply_macro(&f, operands))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn eval_list(&mut self, forms: &PersistentList<Value>) -> EvaluationResult<Value> {
        if forms.is_empty() {
            return Ok(Value::List(PersistentList::new()));
        }

        let operator_form = forms.first().unwrap();
        let operand_forms = forms.drop_first().unwrap_or_default();
        if let Some(expansion) = self.get_macro_expansion(operator_form, &operand_forms) {
            match expansion? {
                Value::List(forms) => return self.eval_list(&forms),
                other => return self.evaluate_form(&other),
            }
        }
        match operator_form {
            Value::Symbol(s, None) if s == "def!" => self.eval_def(operand_forms),
            Value::Symbol(s, None) if s == "var" => self.eval_var(operand_forms),
            Value::Symbol(s, None) if s == "let*" => self.eval_let(operand_forms),
            Value::Symbol(s, None) if s == "loop*" => self.eval_loop(operand_forms),
            Value::Symbol(s, None) if s == "recur" => self.eval_recur(operand_forms),
            Value::Symbol(s, None) if s == "if" => self.eval_if(operand_forms),
            Value::Symbol(s, None) if s == "do" => self.eval_do(operand_forms),
            Value::Symbol(s, None) if s == "fn*" => self.eval_fn(operand_forms),
            Value::Symbol(s, None) if s == "quote" => self.eval_quote(operand_forms),
            Value::Symbol(s, None) if s == "quasiquote" => self.eval_quasiquote(operand_forms),
            Value::Symbol(s, None) if s == "defmacro!" => self.eval_defmacro(operand_forms),
            Value::Symbol(s, None) if s == "macroexpand" => self.eval_macroexpand(operand_forms),
            Value::Symbol(s, None) if s == "try*" => self.eval_try(operand_forms),
            operator_form => match self.evaluate_form(operator_form)? {
                Value::Fn(f) => self.apply_fn(&f, operand_forms),
                Value::FnWithCaptures(FnWithCapturesImpl { f, captures }) => {
                    self.extend_from_captures(&captures)?;
                    let result = self.apply_fn(&f, operand_forms);
                    self.leave_scope();
                    result
                }
                Value::Primitive(native_fn) => {
                    self.apply_stack.push(operator_form.clone());
                    match self.apply_primitive(native_fn, operand_forms) {
                        result @ Ok(..) => {
                            self.apply_stack.pop().unwrap();
                            result
                        }
                        err @ Err(..) => {
                            if self.failed_form.is_none() {
                                self.failed_form = Some(self.apply_stack.len() - 1);
                            }
                            err
                        }
                    }
                }
                v => Err(EvaluationError::CannotInvoke(v)),
            },
        }
    }

    /// Evaluate the `form` according to the semantics of the language.
    pub fn evaluate(&mut self, form: &Value) -> EvaluationResult<Value> {
        let result = self.evaluate_form(form);
        self.failed_form.take();
        self.apply_stack.clear();
        result
    }

    fn evaluate_form(&mut self, form: &Value) -> EvaluationResult<Value> {
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
                    let value = self.evaluate_form(form)?;
                    result.push_back_mut(value);
                }
                Ok(Value::Vector(result))
            }
            Value::Map(forms) => {
                let mut result = PersistentMap::new();
                for (k, v) in forms {
                    let key = self.evaluate_form(k)?;
                    let value = self.evaluate_form(v)?;
                    result.insert_mut(key, value);
                }
                Ok(Value::Map(result))
            }
            Value::Set(forms) => {
                let mut result = PersistentSet::new();
                for form in forms {
                    let value = self.evaluate_form(form)?;
                    result.insert_mut(value);
                }
                Ok(Value::Set(result))
            }
            Value::Var(v) => match var_impl_into_inner(v) {
                Some(value) => Ok(value),
                None => Ok(Value::Var(v.clone())),
            },
            f @ Value::Fn(_) => Ok(f.clone()),
            Value::FnWithCaptures(FnWithCapturesImpl { f, captures }) => {
                let mut captures = captures.clone();
                update_captures(&mut captures, &self.scopes)?;
                Ok(Value::FnWithCaptures(FnWithCapturesImpl {
                    f: f.clone(),
                    captures,
                }))
            }
            f @ Value::Primitive(_) => Ok(f.clone()),
            Value::Recur(_) => unreachable!(),
            a @ Value::Atom(_) => Ok(a.clone()),
            Value::Macro(_) => unreachable!(),
            Value::Exception(_) => unreachable!(),
        }
    }

    /// Evaluate `form` in the global scope of the interpreter.
    /// This method is exposed for the `eval` primitive which
    /// has these semantics.
    pub(crate) fn evaluate_in_global_scope(&mut self, form: &Value) -> EvaluationResult<Value> {
        let mut child_scopes: Vec<_> = self.scopes.drain(1..).collect();
        let result = self.evaluate_form(form);
        self.scopes.append(&mut child_scopes);
        result
    }

    fn resolve_in_lexical_scopes(&self, identifier: &Identifier) -> RuntimeValue {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(identifier) {
                return value.clone();
            }
        }
        unreachable!("analysis guarantees identifier is in scope")
    }

    fn eval_def(&mut self, form: DefForm) -> EvaluationResult<RuntimeValue> {
        let var = match form {
            DefForm::Bound(symbol, form) => {
                let value = self.evaluate_analyzed_form(*form)?;
                self.intern(symbol, Some(value))?
            }
            DefForm::Unbound(symbol) => self.intern(symbol, None)?,
        };
        Ok(RuntimeValue::Var(var))
    }

    fn eval_var(&self, symbol: &Symbol) -> EvaluationResult<RuntimeValue> {
        self.namespaces
            .resolve_symbol(symbol)
            .map(RuntimeValue::Var)
            .map_err(EvaluationError::from)
    }

    fn extend_lexical_scope<'a>(
        &'a mut self,
        bindings: impl Iterator<Item = (&'a Identifier, RuntimeValue)>,
    ) {
        let scope = Scope::from(bindings);
        self.scopes.push(scope);
    }

    fn eval_body_form(&mut self, BodyForm { body }: &BodyForm) -> EvaluationResult<RuntimeValue> {
        let mut result = RuntimeValue::Nil;
        for &form in body {
            result = self.evaluate_analyzed_form(form)?
        }
        Ok(result)
    }

    fn eval_let(
        &mut self,
        LetForm {
            lexical_form,
            forward_declarations,
        }: LetForm,
    ) -> EvaluationResult<RuntimeValue> {
        let has_forward_declarations = !forward_declarations.is_empty();
        if has_forward_declarations {
            self.extend_lexical_scope(
                forward_declarations
                    .into_iter()
                    .map(|symbol| (&symbol.identifier, RuntimeValue::Var(Var::Unbound))),
            );
        }

        let bindings = match lexical_form.bindings {
            LexicalBindings::Bound(bindings) => bindings
                .into_iter()
                .map(|(symbol, form)| Ok((&symbol.identifier, self.evaluate_analyzed_form(*form)?)))
                .collect::<Result<Vec<(&Identifier, RuntimeValue)>, _>>()
                .map_err(|err| {
                    if has_forward_declarations {
                        self.leave_scope();
                    }
                    err
                })?,
            LexicalBindings::Unbound(bindings) => {
                unreachable!("let* cannot have unbound local symbols")
            }
        };
        // TODO: are forward decls being resolved correctly?
        self.extend_lexical_scope(bindings.iter());

        let result = self.eval_body_form(&lexical_form.body);

        self.leave_scope();
        if has_forward_declarations {
            self.leave_scope();
        }
        result
    }

    fn eval_loop<'a>(
        &'a mut self,
        lexical_form: LexicalForm<'a>,
    ) -> EvaluationResult<RuntimeValue> {
        let bindings = match lexical_form.bindings {
            LexicalBindings::Bound(bindings) => bindings
                .into_iter()
                .map(|(symbol, form)| Ok((&symbol.identifier, self.evaluate_analyzed_form(*form)?)))
                .collect::<Result<Vec<(&Identifier, RuntimeValue)>, _>>()?,
            LexicalBindings::Unbound(bindings) => {
                unreachable!("let* cannot have unbound local symbols")
            }
        };
        self.extend_lexical_scope(bindings.iter());

        let delimiter = self.control_stack.len();
        self.control_stack.push(ControlFlow::Continue);

        let body = &lexical_form.body;
        let mut result = self.eval_body_form(body).map_err(|err| {
            self.control_stack.truncate(delimiter);
            self.leave_scope();
            err
        })?;
        while let ControlFlow::Recur(values) = self.control_stack.pop().unwrap() {
            let scope = self.scopes.last_mut().unwrap();
            for ((name, _), value) in bindings.iter().zip(values.into_iter()) {
                scope.insert(*name, value);
            }
            result = self.eval_body_form(body).map_err(|err| {
                self.control_stack.truncate(delimiter);
                self.leave_scope();
                err
            })?;
        }

        // NOTE: any control information added in this fn was popped in loop
        self.leave_scope();

        Ok(result)
    }

    fn eval_recur(&mut self, BodyForm { body }: BodyForm) -> EvaluationResult<RuntimeValue> {
        let result = body
            .into_iter()
            .map(|form| self.evaluate_analyzed_form(form))
            .collect::<Result<Vec<RuntimeValue>, _>>()?;
        self.control_stack.push(ControlFlow::Recur(result));
        Ok(RuntimeValue::Nil)
    }

    fn eval_if(
        &mut self,
        IfForm {
            predicate,
            consequent,
            alternate,
        }: IfForm,
    ) -> EvaluationResult<RuntimeValue> {
        let predicate = self.evaluate_analyzed_form(*predicate)?;
        let falsey = matches!(predicate, RuntimeValue::Nil | RuntimeValue::Bool(false));
        if falsey {
            if let Some(alternate) = alternate {
                self.evaluate_analyzed_form(*alternate)
            } else {
                Ok(RuntimeValue::Nil)
            }
        } else {
            self.evaluate_analyzed_form(*consequent)
        }
    }

    fn eval_fn(&mut self, fn_form: FnForm) -> EvaluationResult<RuntimeValue> {
        // TODO: address lifetimes
        Ok(RuntimeValue::Fn(fn_form))
    }

    fn eval_quote(&mut self, form: Box<AnalyzedForm>) -> EvaluationResult<RuntimeValue> {
        Ok(form.into())
    }

    fn eval_quasiquote(&mut self, form: Box<AnalyzedForm>) -> EvaluationResult<RuntimeValue> {
        let expansion = eval_quasiquote(form)?;
        self.evaluate_analyzed_form(expansion)
    }

    fn eval_defmacro(&mut self, name: &Symbol, fn_form: FnForm) -> EvaluationResult<RuntimeValue> {
        // TODO
    }

    fn eval_macroexpand(&mut self, form: Box<AnalyzedForm>) -> EvaluationResult<RuntimeValue> {
        // TODO
    }

    fn eval_try(&mut self, TryForm { body, catch }: TryForm) -> EvaluationResult<RuntimeValue> {
        // TODO
    }

    fn eval_catch(
        &mut self,
        CatchForm {
            exception_binding,
            body,
        }: CatchError,
    ) -> EvaluationResult<RuntimeValue> {
        // TODO
    }

    fn evaluate_analyzed_form(&mut self, form: AnalyzedForm) -> EvaluationResult<RuntimeValue> {
        let value = match form {
            AnalyzedForm::LexicalSymbol(identifier) => self.resolve_in_lexical_scopes(identifier),
            AnalyzedForm::Var(var) => {
                match var {
                    Var::Bound(value) => {
                        // TODO fix
                        RuntimeValue::Nil
                    }
                    Var::Unbound => RuntimeValue::Var(var),
                }
            }
            AnalyzedForm::Atom(atom) => atom.into(),
            AnalyzedForm::List(inner) => match inner {
                AnalyzedList::Def(form) => self.eval_def(form),
                AnalyzedList::Var(symbol) => self.eval_var(symbol),
                AnalyzedList::Let(form) => self.eval_let(form),
                AnalyzedList::Loop(form) => self.eval_loop(form),
                AnalyzedList::Recur(form) => self.eval_recur(form),
                AnalyzedList::If(form) => self.eval_if(form),
                AnalyzedList::Do(form) => self.eval_body_form(&form),
                AnalyzedList::Fn(form) => self.eval_fn(form),
                AnalyzedList::Quote(form) => self.eval_quote(form),
                AnalyzedList::Quasiquote(form) => self.eval_quasiquote(form),
                // AnalyzedList::Unquote(form) => self.eval_unquote(form),
                // AnalyzedList::SpliceUnquote(form) => self.eval_splice_unquote(form),
                AnalyzedList::Defmacro(name, form) => self.eval_defmacro(name, form),
                AnalyzedList::Macroexpand(form) => self.eval_macroexpand(form),
                AnalyzedList::Try(form) => self.eval_try(form),
                AnalyzedList::Catch(form) => self.eval_catch(form),
                AnalyzedList::Form(coll) => {
                    let evaluated_coll = coll
                        .into_iter()
                        .map(|form| self.evaluate_analyzed_form(form))
                        .collect::<Result<PersistentList, _>>()?;
                    RuntimeValue::List(evaluated_coll)
                }
            },
            AnalyzedForm::Vector(coll) => {
                let evaluated_coll = coll
                    .into_iter()
                    .map(|form| self.evaluate_analyzed_form(form))
                    .collect::<Result<PersistentVector, _>>()?;
                RuntimeValue::Vector(evaluated_coll)
            }
            AnalyzedForm::Map(coll) => {
                let evaluated_coll = coll
                    .into_iter()
                    .map(|(key, value)| {
                        let evaluated_key = self.evaluate_analyzed_form(form);
                        let evaluated_value = self.evaluate_analyzed_form(form);
                        [evaluated_key, evaluated_value]
                    })
                    .collect::<Result<PersistentMap, _>>()?;
                RuntimeValue::Map(evaluated_coll)
            }
            AnalyzedForm::Set(coll) => {
                let evaluated_coll = coll
                    .into_iter()
                    .map(|form| self.evaluate_analyzed_form(form))
                    .collect::<Result<PersistentSet, _>>()?;
                RuntimeValue::Set(evaluated_coll)
            }
        };
        Ok(value)
    }

    fn analyze_and_evaluate(&mut self, form: &Form) -> EvaluationResult<RuntimeValue> {
        let analyzer = Analyzer::new(&self.namespaces);
        let analyzed_form = analyzer.analyze(form)?;
        self.evaluate_analyzed_form(analyzed_form)
    }

    pub fn interpret(&mut self, source: &str) -> EvaluationResult<Vec<Value>> {
        read(source)
            .map_err(|err| EvaluationError::ReaderError(err, source.to_string()))?
            .iter()
            .map(|form| self.analyze_and_evaluate(form))
            .collect::<Result<RuntimeValue, _>>()?;
        // TODO fix return
        Ok(vec![Value::Nil])
    }
}

#[cfg(test)]
mod test {
    use crate::collections::{PersistentList, PersistentMap, PersistentVector};
    use crate::reader;
    use crate::testing::run_eval_test;
    use crate::value::{
        atom_with_value, exception, list_with_values, map_with_values, var_with_value,
        vector_with_values,
        Value::{self, *},
    };

    const DEFAULT_NAMESPACE: &str = "core";

    fn read_one_value(input: &str) -> Value {
        let form = reader::read(input)
            .expect("is valid")
            .into_iter()
            .nth(0)
            .expect("some");

        (&form).into()
    }

    #[test]
    fn test_basic_self_evaluating() {
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
            ("\"abc\\ndef\\nghi\"", String("abc\ndef\nghi".to_string())),
            ("\"abc\\def\\ghi\"", String("abc\\def\\ghi".to_string())),
            ("\" \\\\n \"", String(" \\n ".to_string())),
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
        let test_cases =
            vec![
            ("(let* [] )", Nil),
            ("(let* [a 1] )", Nil),
            ("(let* [a 3] a)", Number(3)),
            ("(let* [_ 30 a _] a)", Number(30)),
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
            (
                "(let* [cst (fn* [n] (if (= n 0) :success (cst (- n 1))))] (cst 1))",
                Keyword("success".to_string(), None),
            ),
            (
                "(let* [f (fn* [n] (if (= n 0) :success (g (- n 1)))) g (fn* [n] (f n))] (f 2))",
                Keyword("success".to_string(), None),
            ),
            // test captures inside `let*`
            ("(let* [y (let* [x 12] (fn* [] x))] (y))", Number(12)),
            ("(let* [y (let* [x 12] (fn* [] (fn* [] x)))] ((y)))", Number(12)),
            ("(let* [y (let* [x 12] ((fn* [x] (fn* [] (inc x))) x))] (y))", Number(13)),
            ("(let* [y (let* [y 12] ((fn* [y] (fn* [] (inc y))) y))] (y))", Number(13)),
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
            ("((fn* []))", Nil),
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
            ("(def! f (fn* [a] (fn* [b] (+ a b)))) ((first (let* [x 12] (map (fn* [_] (f x)) '(10000000)))) 27)", Number(39)),
            // test `let*` bindings inside a `fn*`
            (
                "(defn f [] (let* [cst (fn* [n] (if (= n 0) :success (cst (- n 1))))] (cst 10))) (f)",
                Keyword("success".to_string(), None),
            ),
            (
                "(def! f (fn* [ast] (let* [ast ast] ast))) (f 22)",
                Number(22),
            ),
            ("(def! f (fn* [ast] (let* [ast (inc ast) bar (inc ast)] bar))) (f 22)", Number(24)),
            // test capturing let* bindings
            ("(def f (fn* [x] (let* [x x] (if (list? (first x)) (f (first x)) (fn* [] x))))) (first ((eval (f '(3)))))", Number(3)),
            ("(def f (fn* [x] (let* [x '((fn* [] 4))] (if (list? (first x)) (f (first x)) (fn* [] x))))) ((first ((eval (f '((fn* [] 3)))))))", Number(4)),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_loop_recur() {
        let test_cases = vec![
            ("(loop* [i 12] i)", Number(12)),
            ("(loop* [i 12])", Nil),
            ("(loop* [])", Nil),
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
                read_one_value("(a lst d)"),
            ),
            (
                "`(1 2 (3 4))",
                read_one_value("(1 2 (3 4))"),
            ),
            (
                "`(nil)",
                read_one_value("(nil)"),
            ),
            (
                "`(1 ())",
                read_one_value("(1 ())"),
            ),
            (
                "`(() 1)",
                read_one_value("(() 1)"),
            ),
            (
                "`(2 () 1)",
                read_one_value("(2 () 1)"),
            ),
            (
                "`(())",
                read_one_value("(())"),
            ),
            (
                "`(f () g (h) i (j k) l)",
                read_one_value("(f () g (h) i (j k) l)"),
            ),
            ("`~7", Number(7)),
            ("(def! a 8) `a", Symbol("a".to_string(), None)),
            ("(def! a 8) `~a", Number(8)),
            (
                "`(1 a 3)",
                read_one_value("(1 a 3)"),
            ),
            (
                "(def! a 8) `(1 ~a 3)",
                read_one_value("(1 8 3)"),
            ),
            (
                "(def! b '(1 :b :d)) `(1 b 3)",
                read_one_value("(1 b 3)"),
            ),
            (
                "(def! b '(1 :b :d)) `(1 ~b 3)",
                read_one_value("(1 (1 :b :d) 3)"),
            ),
            (
                "`(~1 ~2)",
                read_one_value("(1 2)"),
            ),
            ("(let* [x 0] `~x)", Number(0)),
            (
                "(def! lst '(b c)) `(a ~lst d)",
                read_one_value("(a (b c) d)"),
            ),
            (
                "(def! lst '(b c)) `(a ~@lst d)",
                read_one_value("(a b c d)"),
            ),
            (
                "(def! lst '(b c)) `(a ~@lst)",
                read_one_value("(a b c)"),
            ),
            (
                "(def! lst '(b c)) `(~@lst 2)",
                read_one_value("(b c 2)"),
            ),
            (
                "(def! lst '(b c)) `(~@lst ~@lst)",
                read_one_value("(b c b c)"),
            ),
            (
                "((fn* [q] (quasiquote ((unquote q) (quote (unquote q))))) (quote (fn* [q] (quasiquote ((unquote q) (quote (unquote q)))))))",
                read_one_value("((fn* [q] (quasiquote ((unquote q) (quote (unquote q))))) (quote (fn* [q] (quasiquote ((unquote q) (quote (unquote q)))))))"),
            ),
            (
                "`[]",
                read_one_value("[]"),
            ),
            (
                "`[[]]",
                read_one_value("[[]]"),
            ),
            (
                "`[()]",
                read_one_value("[()]"),
            ),
            (
                "`([])",
                read_one_value("([])"),
            ),
            (
                "(def! a 8) `[1 a 3]",
                read_one_value("[1 a 3]"),
            ),
            (
                "`[a [] b [c] d [e f] g]",
                read_one_value("[a [] b [c] d [e f] g]"),
            ),
            (
                "(def! a 8) `[~a]",
                read_one_value("[8]"),
            ),
            (
                "(def! a 8) `[(~a)]",
                read_one_value("[(8)]"),
            ),
            (
                "(def! a 8) `([~a])",
                read_one_value("([8])"),
            ),
            (
                "(def! a 8) `[a ~a a]",
                read_one_value("[a 8 a]"),
            ),
            (
                "(def! a 8) `([a ~a a])",
                read_one_value("([a 8 a])"),
            ),
            (
                "(def! a 8) `[(a ~a a)]",
                read_one_value("[(a 8 a)]"),
            ),
            (
                "(def! c '(1 :b :d)) `[~@c]",
                read_one_value("[1 :b :d]"),
            ),
            (
                "(def! c '(1 :b :d)) `[(~@c)]",
                read_one_value("[(1 :b :d)]"),
            ),
            (
                "(def! c '(1 :b :d)) `([~@c])",
                read_one_value("([1 :b :d])"),
            ),
            (
                "(def! c '(1 :b :d)) `[1 ~@c 3]",
                read_one_value("[1 1 :b :d 3]"),
            ),
            (
                "(def! c '(1 :b :d)) `([1 ~@c 3])",
                read_one_value("([1 1 :b :d 3])"),
            ),
            (
                "(def! c '(1 :b :d)) `[(1 ~@c 3)]",
                read_one_value("[(1 1 :b :d 3)]"),
            ),
            (
                "`(0 unquote)",
                read_one_value("(0 unquote)"),
            ),
            (
                "`(0 splice-unquote)",
                read_one_value("(0 splice-unquote)"),
            ),
            (
                "`[unquote 0]",
                read_one_value("[unquote 0]"),
            ),
            (
                "`[splice-unquote 0]",
                read_one_value("[splice-unquote 0]"),
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
            ("(defmacro! unless (fn* [pred a b] `(if ~pred ~b ~a))) (macroexpand '(unless PRED A B))",
                read_one_value("(if PRED B A)")
            ),
            ("(defmacro! unless (fn* [pred a b] (list 'if (list 'not pred) a b))) (macroexpand '(unless PRED A B))",
                read_one_value("(if (not PRED) A B)")
            ),
            ("(defmacro! unless (fn* [pred a b] (list 'if (list 'not pred) a b))) (macroexpand '(unless 2 3 4))",
                read_one_value("(if (not 2) 3 4)")
            ),
            ("(defmacro! identity (fn* [x] x)) (let* [a 123] (macroexpand (identity a)))",
                Number(123),
            ),
            ("(defmacro! identity (fn* [x] x)) (let* [a 123] (identity a))",
                Number(123),
            ),
            ("(macroexpand (cond))", Nil),
            ("(cond)", Nil),
            ("(macroexpand '(cond X Y))",
                read_one_value("(if X Y (cond))")
            ),
            ("(cond true 7)", Number(7)),
            ("(cond true 7 true 8)", Number(7)),
            ("(cond false 7)", Nil),
            ("(cond false 7 true 8)", Number(8)),
            ("(cond false 7 false 8 :else 9)", Number(9)),
            ("(cond false 7 (= 2 2) 8 :else 9)", Number(8)),
            ("(cond false 7 false 8 false 9)", Nil),
            ("(let* [x (cond false :no true :yes)] x)", Keyword("yes".to_string(), None)),
            ("(macroexpand '(cond X Y Z T))",
                read_one_value("(if X Y (cond Z T))")
            ),
            ("(def! x 2) (defmacro! a (fn* [] x)) (a)", Number(2)),
            ("(def! x 2) (defmacro! a (fn* [] x)) (let* [x 3] (a))", Number(2)),
            ("(def! f (fn* [x] (number? x))) (defmacro! m f) [(f (+ 1 1)) (m (+ 1 1))]", vector_with_values(vec![Bool(true), Bool(false)])),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_try_catch() {
        fn exception_value(msg: &str, data: &Value) -> Value {
            Value::Exception(exception(msg, data))
        }

        let exc = exception_value(
            "test",
            &map_with_values(vec![(
                Keyword("cause".to_string(), None),
                String("no memory".to_string()),
            )]),
        );
        let test_cases = vec![
            // NOTE: these are errors from uncaught exceptions now...
            // TODO: map to evaluation error test cases
            // let basic_exc = exception_value("", &String("test".to_string()));
            // ( "(throw \"test\")", basic_exc),
            // ( "(throw {:msg :foo})", exception_value("", &map_with_values(vec![(Keyword("msg".to_string(), None), Keyword("foo".to_string(), None))]))),
            (
                "(try* (throw '(1 2 3)) (catch* e e))",
                exception_value("", &list_with_values(vec![Number(1), Number(2), Number(3)])),
            ),
            ("(try* 22)", Number(22)),
            ("(try* (prn 222) 22)", Number(22)),
            (
                "(try* (ex-info \"test\" {:cause \"no memory\"}))",
                exc.clone(),
            ),
            ("(try* 123 (catch* e 0))", Number(123)),
            (
                "(try* (ex-info \"test\" {:cause \"no memory\"}) (catch* e 0))",
                exc,
            ),
            (
                "(try* (throw (ex-info \"test\" {:cause \"no memory\"})) (catch* e (str e)))",
                String("test, {:cause \"no memory\"}".to_string()),
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
                exception_value(
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
                exception_value(
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
            (
                "(try* abc (catch* exc (prn exc) 2222))",
                Number(2222),
            ),
            (
                "(try* (abc 1 2) (catch* exc (prn exc)))",
                Nil,
            ),
            (
                "(try* (nth () 1) (catch* exc (prn exc)))",
                Nil,
            ),
            (
                "(try* (try* (nth () 1) (catch* exc (prn exc) (throw exc))) (catch* exc (prn exc) 33))",
                Number(33),
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
}
