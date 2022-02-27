use crate::analyzer::{AnalysisError, Analyzer};
use crate::collections::{PersistentList, PersistentMap, PersistentSet, PersistentVector};
use crate::lang::core;
use crate::namespace::{Context as NamespaceContext, NamespaceDesc, NamespaceError, Var};
use crate::reader::{read, Form, Identifier, ReadError, Symbol};
use crate::value::{
    exception_from_system_err, BodyForm, CatchForm, DefForm, ExceptionImpl, FnForm, IfForm,
    LetForm, RuntimeValue, SpecialForm, TryForm,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::default::Default;
use std::fmt::Write;
use std::rc::Rc;
use std::time::SystemTimeError;
use std::{fmt, io};
use thiserror::Error;

const COMMAND_LINE_ARGS_IDENTIFIER: &str = "*command-line-args*";

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
    LexicalBindingsMustBePaired(PersistentVector<RuntimeValue>),
    #[error("expected vector of lexical bindings instead of `{0}`")]
    LexicalBindingsMustBeVector(RuntimeValue),
    #[error("names in form must be non-namespaced symbols unlike `{0}`")]
    LexicalBindingsMustHaveSymbolNames(RuntimeValue),
    #[error("missing argument for variadic binding")]
    VariadicArgMissing,
    #[error("found multiple variadic arguments in `{0}`; only one is allowed.")]
    VariadicArgMustBeUnique(RuntimeValue),
}

#[derive(Debug, Error, Clone)]
pub enum EvaluationError {
    #[error("form invoked with an argument of the incorrect type: expected a value of type(s) `{expected}` but found value `{realized}`")]
    WrongType {
        expected: &'static str,
        realized: RuntimeValue,
    },
    #[error("form invoked with incorrect arity: provided {realized} arguments but expected {expected} arguments")]
    WrongArity { expected: usize, realized: usize },
    #[error("var `{0}` not found in namespace `{1}`")]
    MissingVar(String, String),
    #[error("symbol `{0}` could not be resolved")]
    UnableToResolveSymbolToValue(String),
    #[error("cannot invoke the supplied value `{0}`")]
    CannotInvoke(RuntimeValue),
    #[error("missing value for captured symbol `{0}`")]
    MissingCapturedValue(String),
    #[error("cannot deref an unbound var `{0}`")]
    CannotDerefUnboundVar(RuntimeValue),
    #[error("overflow detected during arithmetic operation of {0} and {1}")]
    Overflow(i64, i64),
    #[error("could not negate {0}")]
    Negation(i64),
    #[error("underflow detected during arithmetic operation of {0} and {1}")]
    Underflow(i64, i64),
    #[error("requested index {0} in collection with length {1}")]
    IndexOutOfBounds(usize, usize),
    #[error("map cannot be constructed with an odd number of arguments: `{0}` with length `{1}`")]
    MapRequiresPairs(RuntimeValue, usize),
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
pub type Scope = HashMap<Identifier, RuntimeValue>;

// fn eval_quasiquote_list_inner(
//     elems: impl Iterator<Item = & Value>,
// ) -> EvaluationResult<Value> {
//     let mut result = Value::List(PersistentList::new());
//     for form in elems {
//         match form {
//             Value::List(inner) => {
//                 if let Some(first_inner) = inner.first() {
//                     match first_inner {
//                         Value::Symbol(s, None) if s == "splice-unquote" => {
//                             if let Some(rest) = inner.drop_first() {
//                                 if let Some(second) = rest.first() {
//                                     result = list_with_values(vec![
//                                         Value::Symbol(
//                                             "concat".to_string(),
//                                             Some("core".to_string()),
//                                         ),
//                                         second.clone(),
//                                         result,
//                                     ]);
//                                 }
//                             } else {
//                                 return Err(EvaluationError::WrongArity {
//                                     expected: 1,
//                                     realized: 0,
//                                 });
//                             }
//                         }
//                         _ => {
//                             result = list_with_values(vec![
//                                 Value::Symbol("cons".to_string(), Some("core".to_string())),
//                                 eval_quasiquote(form)?,
//                                 result,
//                             ]);
//                         }
//                     }
//                 } else {
//                     result = list_with_values(vec![
//                         Value::Symbol("cons".to_string(), Some("core".to_string())),
//                         Value::List(PersistentList::new()),
//                         result,
//                     ]);
//                 }
//             }
//             form => {
//                 result = list_with_values(vec![
//                     Value::Symbol("cons".to_string(), Some("core".to_string())),
//                     eval_quasiquote(form)?,
//                     result,
//                 ]);
//             }
//         }
//     }
//     Ok(result)
// }

// fn eval_quasiquote_list(elems: &PersistentList<Value>) -> EvaluationResult<Value> {
//     if let Some(first) = elems.first() {
//         match first {
//             Value::Symbol(s, None) if s == "unquote" => {
//                 if let Some(rest) = elems.drop_first() {
//                     if let Some(argument) = rest.first() {
//                         return Ok(argument.clone());
//                     }
//                 }
//                 return Err(EvaluationError::WrongArity {
//                     realized: 0,
//                     expected: 1,
//                 });
//             }
//             _ => return eval_quasiquote_list_inner(elems.reverse().iter()),
//         }
//     }
//     Ok(Value::List(PersistentList::new()))
// }

// fn eval_quasiquote_vector(elems: &PersistentVector<Value>) -> EvaluationResult<Value> {
//     Ok(list_with_values(vec![
//         Value::Symbol("vec".to_string(), Some("core".to_string())),
//         eval_quasiquote_list_inner(elems.iter().rev())?,
//     ]))
// }

// fn eval_quasiquote(form: RuntimeValue) -> EvaluationResult<RuntimeValue> {
//     match form {
//         Value::List(elems) => eval_quasiquote_list(&elems),
//         Value::Vector(elems) => eval_quasiquote_vector(&elems),
//         elem @ Value::Map(_) | elem @ Value::Symbol(..) => {
//             let args = vec![Value::Symbol("quote".to_string(), None), elem.clone()];
//             Ok(list_with_values(args.into_iter()))
//         }
//         v => Ok(v.clone()),
//     }
// }

// fn update_captures(
//     captures: &mut HashMap<String, Option<Value>>,
//     scopes: &[Scope],
// ) -> EvaluationResult<()> {
//     for (capture, value) in captures {
//         if value.is_none() {
//             let captured_value = resolve_symbol_in_scopes(scopes.iter().rev(), capture)
//                 .ok_or_else(|| {
//                     EvaluationError::UnableToResolveSymbolToValue(capture.to_string())
//                 })?;
//             *value = Some(captured_value.clone());
//         }
//     }
//     Ok(())
// }

#[derive(Debug)]
enum ControlFlow {
    Continue,
    Recur(Vec<RuntimeValue>),
}

#[derive(Debug)]
pub struct Interpreter {
    namespaces: Rc<RefCell<NamespaceContext>>,
    analyzer: Analyzer,
    symbol_index: Option<Rc<RefCell<SymbolIndex>>>,

    pub(crate) scopes: Vec<Scope>,
    control_stack: Vec<ControlFlow>,
    // low-res backtrace
    // pub(crate) apply_stack: Vec<RuntimeValue>,
    // index into `apply_stack` pointing at the first form to error
    // failed_form: Option<usize>,
}

impl Default for Interpreter {
    fn default() -> Self {
        let namespaces = Rc::new(RefCell::new(NamespaceContext::default()));
        let analyzer = Analyzer::new(namespaces.clone());

        let mut interpreter = Interpreter {
            namespaces,
            analyzer,
            symbol_index: None,
            scopes: vec![],
            control_stack: vec![],
            // apply_stack: vec![],
            // failed_form: None,
        };

        interpreter
            .load_namespace(core::namespace())
            .expect("is valid");

        // add support for `*command-line-args*`
        let mut buffer = String::new();
        let _ = write!(&mut buffer, "(def! {} '())", COMMAND_LINE_ARGS_IDENTIFIER)
            .expect("can write to string");
        interpreter.interpret(&buffer).expect("valid source");

        interpreter
    }
}

impl Interpreter {
    pub fn current_namespace(&self) -> Identifier {
        self.namespaces.borrow().current_namespace_name().clone()
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

    pub fn load_namespace(
        &mut self,
        NamespaceDesc {
            name,
            namespace,
            source,
        }: NamespaceDesc,
    ) -> EvaluationResult<()> {
        self.namespaces
            .borrow_mut()
            .intern_namespace(&name, namespace);

        if let Some(source) = source {
            // self.interpret(source)?;
        }

        Ok(())
    }

    /// Store `args` in the var referenced by `COMMAND_LINE_ARGS_SYMBOL`.
    pub fn intern_args(&mut self, args: impl Iterator<Item = String>) {
        let form = args.map(RuntimeValue::String).collect();
        self.intern(
            &Symbol {
                identifier: COMMAND_LINE_ARGS_IDENTIFIER.to_string(),
                namespace: None,
            },
            Some(RuntimeValue::List(form)),
        )
        .expect("'*command-line-args* constructed correctly");
    }

    /// Read the interned command line argument at position `n` in the collection.
    pub fn command_line_arg(&mut self, n: usize) -> EvaluationResult<String> {
        let symbol = Symbol {
            identifier: COMMAND_LINE_ARGS_IDENTIFIER.to_string(),
            namespace: None,
        };
        match self.resolve_symbol_to_value(&symbol)? {
            RuntimeValue::List(args) => match args.iter().nth(n) {
                Some(value) => match value {
                    RuntimeValue::String(arg) => Ok(arg.clone()),
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
        let mut namespaces = self.namespaces.borrow_mut();
        let namespace = match &symbol.namespace {
            Some(ns) => namespaces.get_namespace_mut(ns),
            None => namespaces.current_namespace_mut(),
        };
        let var = namespace.intern(&symbol.identifier, value)?;

        // TODO update indexing...
        if let Some(index) = &self.symbol_index {
            let mut index = index.borrow_mut();
            index.insert(symbol.identifier.clone());
        }
        Ok(var)
    }

    fn resolve_symbol_to_value(&self, symbol: &Symbol) -> EvaluationResult<RuntimeValue> {
        // TODO Rc for cheap clone
        Ok(self.namespaces.borrow().resolve_symbol(symbol)?.value())
    }

    fn enter_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    /// Exits the current lexical scope.
    /// NOTE: exposed for some prelude functionality.
    pub(crate) fn exit_scope(&mut self) {
        let _ = self.scopes.pop().expect("no underflow in scope stack");
    }

    // fn apply_macro(
    //     &mut self,
    //     f: &FnImpl,
    //     operands: &PersistentList<Value>,
    // ) -> EvaluationResult<Value> {
    //     let result = self.apply_fn_inner(f, operands, operands.len())?;
    //     if let Value::List(forms) = result {
    //         return self.expand_macro_if_present(&forms);
    //     }
    //     Ok(result)
    // }

    // fn expand_macro_if_present(
    //     &mut self,
    //     forms: &PersistentList<Value>,
    // ) -> EvaluationResult<Value> {
    //     if let Some(first) = forms.first() {
    //         let rest = forms.drop_first().expect("list is not empty");
    //         if let Some(expansion) = self.get_macro_expansion(first, &rest) {
    //             expansion
    //         } else {
    //             Ok(Value::List(forms.clone()))
    //         }
    //     } else {
    //         Ok(Value::List(PersistentList::new()))
    //     }
    // }

    // pub fn extend_from_captures(
    //     &mut self,
    //     captures: &HashMap<String, Option<Value>>,
    // ) -> EvaluationResult<()> {
    //     self.enter_scope();
    //     for (capture, value) in captures {
    //         if let Some(value) = value {
    //             self.insert_value_in_current_scope(capture, value.clone());
    //         } else {
    //             self.exit_scope();
    //             return Err(EvaluationError::MissingCapturedValue(capture.to_string()));
    //         }
    //     }
    //     Ok(())
    // }

    // pub(crate) fn get_macro_expansion(
    //     &mut self,
    //     operator: &Value,
    //     operands: &PersistentList<Value>,
    // ) -> Option<EvaluationResult<Value>> {
    //     match operator {
    //         Value::Symbol(identifier, ns_opt) => {
    //             if let Ok(Value::Macro(f)) = self.resolve_symbol(identifier, ns_opt.as_ref()) {
    //                 Some(self.apply_macro(&f, operands))
    //             } else {
    //                 None
    //             }
    //         }
    //         Value::Var(v) => {
    //             if let Some(Value::Macro(f)) = var_impl_into_inner(v) {
    //                 Some(self.apply_macro(&f, operands))
    //             } else {
    //                 None
    //             }
    //         }
    //         _ => None,
    //     }
    // }

    fn resolve_in_lexical_scopes(&self, identifier: &Identifier) -> RuntimeValue {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(identifier) {
                // TODO: wrap in Rc for cheap clone?
                return value.clone();
            }
        }
        unreachable!("analysis guarantees identifier is in scope")
    }

    fn eval_def(&mut self, form: &DefForm) -> EvaluationResult<RuntimeValue> {
        let var = match form {
            DefForm::Bound(symbol, form) => {
                let value = self.evaluate_analyzed_form(form)?;
                self.intern(symbol, Some(value))?
            }
            DefForm::Unbound(symbol) => self.intern(symbol, None)?,
        };
        Ok(RuntimeValue::Var(var))
    }

    fn eval_var(&self, symbol: &Symbol) -> EvaluationResult<RuntimeValue> {
        self.namespaces
            .borrow()
            .resolve_symbol(&symbol)
            .map(RuntimeValue::Var)
            .map_err(EvaluationError::from)
    }

    fn eval_body_form(&mut self, BodyForm { body }: &BodyForm) -> EvaluationResult<RuntimeValue> {
        let mut result = RuntimeValue::Nil;
        for form in body {
            result = self.evaluate_analyzed_form(form)?
        }
        Ok(result)
    }

    fn eval_let(&mut self, let_form: &LetForm) -> EvaluationResult<RuntimeValue> {
        let bindings = &let_form.bindings;
        let body = &let_form.body;
        let forward_declarations = &let_form.forward_declarations;

        self.enter_scope();
        for &index in forward_declarations {
            let name = let_form.identifier_for_binding(index).unwrap();
            let value = RuntimeValue::Var(Var::Unbound);
            let lexical_scope = self.scopes.last_mut().unwrap();
            lexical_scope.insert(name.clone(), value);
        }

        for (name, value) in bindings {
            match self.evaluate_analyzed_form(value) {
                Ok(value) => {
                    let lexical_scope = self.scopes.last_mut().unwrap();
                    lexical_scope.insert(name.clone(), value);
                }
                err @ Err(..) => {
                    self.exit_scope();
                    return err;
                }
            }
        }

        // TODO: are forward decls being resolved correctly?
        let result = self.eval_body_form(body);
        self.exit_scope();
        result
    }

    fn eval_loop(&mut self, loop_form: &LetForm) -> EvaluationResult<RuntimeValue> {
        let bindings = &loop_form.bindings;
        let body = &loop_form.body;
        let forward_declarations = &loop_form.forward_declarations;

        self.enter_scope();
        for &index in forward_declarations {
            let name = loop_form.identifier_for_binding(index).unwrap();
            let value = RuntimeValue::Var(Var::Unbound);
            let lexical_scope = self.scopes.last_mut().unwrap();
            lexical_scope.insert(name.clone(), value);
        }

        for (name, value) in bindings {
            match self.evaluate_analyzed_form(value) {
                Ok(value) => {
                    let lexical_scope = self.scopes.last_mut().unwrap();
                    lexical_scope.insert(name.clone(), value);
                }
                err @ Err(..) => {
                    self.exit_scope();
                    return err;
                }
            }
        }

        let delimiter = self.control_stack.len();
        self.control_stack.push(ControlFlow::Continue);

        let mut result = self.eval_body_form(body).map_err(|err| {
            self.control_stack.truncate(delimiter);
            self.exit_scope();
            err
        })?;
        while let ControlFlow::Recur(values) = self.control_stack.pop().unwrap() {
            let scope = self.scopes.last_mut().unwrap();
            for ((name, _), value) in bindings.iter().zip(values.into_iter()) {
                scope.insert(name.to_string(), value);
            }
            result = self.eval_body_form(body).map_err(|err| {
                self.control_stack.truncate(delimiter);
                self.exit_scope();
                err
            })?;
        }

        // NOTE: any control information added in this fn was popped in loop
        self.exit_scope();

        Ok(result)
    }

    fn eval_recur(&mut self, BodyForm { body }: &BodyForm) -> EvaluationResult<RuntimeValue> {
        let result = body
            .iter()
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
        }: &IfForm,
    ) -> EvaluationResult<RuntimeValue> {
        let predicate = self.evaluate_analyzed_form(predicate)?;
        let falsey = matches!(predicate, RuntimeValue::Nil | RuntimeValue::Bool(false));
        if falsey {
            if let Some(alternate) = alternate {
                self.evaluate_analyzed_form(alternate)
            } else {
                Ok(RuntimeValue::Nil)
            }
        } else {
            self.evaluate_analyzed_form(consequent)
        }
    }

    fn eval_fn(&mut self, fn_form: &FnForm) -> EvaluationResult<RuntimeValue> {
        // walk the form again to grab any more refs
        // TODO captures here as well?
        Ok(RuntimeValue::Fn(fn_form.clone()))
    }

    fn eval_quote(&mut self, form: &RuntimeValue) -> EvaluationResult<RuntimeValue> {
        Ok(form.clone())
    }

    fn eval_quasiquote(&mut self, form: &RuntimeValue) -> EvaluationResult<RuntimeValue> {
        // TODO
        // let expansion = eval_quasiquote(form)?;
        // self.evaluate_analyzed_form(expansion)
        Ok(RuntimeValue::Nil)
    }

    fn eval_defmacro(&mut self, name: &Symbol, fn_form: &FnForm) -> EvaluationResult<RuntimeValue> {
        // TODO
        Ok(RuntimeValue::Nil)
    }

    fn eval_macroexpand(&mut self, form: &RuntimeValue) -> EvaluationResult<RuntimeValue> {
        // TODO
        Ok(RuntimeValue::Nil)
    }

    fn eval_try(&mut self, TryForm { body, catch }: &TryForm) -> EvaluationResult<RuntimeValue> {
        match self.eval_body_form(&body) {
            Ok(result) => Ok(result),
            Err(err) => {
                if let Some(catch_form) = catch {
                    self.eval_catch(catch_form, err)
                } else {
                    Err(err)
                }
            }
        }
    }

    fn eval_catch(
        &mut self,
        CatchForm {
            exception_binding,
            body,
        }: &CatchForm,
        err: EvaluationError,
    ) -> EvaluationResult<RuntimeValue> {
        let exception = exception_from_system_err(err);

        self.enter_scope();
        let scope = self.scopes.last_mut().unwrap();
        scope.insert(exception_binding.clone(), exception);

        let result = self.eval_body_form(body);

        self.exit_scope();
        result
    }

    pub(crate) fn apply_fn(
        &mut self,
        f: &FnForm,
        mut args: Vec<RuntimeValue>,
    ) -> EvaluationResult<RuntimeValue> {
        let correct_arity = if f.variadic() {
            args.len() >= f.arity()
        } else {
            args.len() == f.arity()
        };
        if !correct_arity {
            // TODO better wording on error?
            return Err(EvaluationError::WrongArity {
                expected: f.arity(),
                realized: args.len(),
            });
        }

        let variable = args.drain(f.arity()..).collect::<PersistentList<_>>();

        self.enter_scope();
        for (parameter, value) in f.parameters.iter().zip(args.into_iter()) {
            let scope = self.scopes.last_mut().unwrap();
            scope.insert(parameter.clone(), value);
        }

        if f.variadic() {
            let variadic_binding = f.variadic.as_ref().unwrap();
            let variadic_arg = RuntimeValue::List(variable);

            let scope = self.scopes.last_mut().unwrap();
            scope.insert(variadic_binding.clone(), variadic_arg);
        }

        let result = self.eval_body_form(&f.body);

        self.exit_scope();
        result
    }

    fn evaluate_analyzed_list(
        &mut self,
        coll: &PersistentList<RuntimeValue>,
    ) -> EvaluationResult<RuntimeValue> {
        if let Some(operator) = coll.first() {
            let operands = coll.drop_first().unwrap();
            // TODO macroexpand
            match self.evaluate_analyzed_form(operator)? {
                RuntimeValue::Fn(f) => {
                    let operands = operands
                        .iter()
                        .map(|form| self.evaluate_analyzed_form(form))
                        .collect::<Result<Vec<RuntimeValue>, _>>()?;
                    self.apply_fn(&f, operands)
                }
                RuntimeValue::Primitive(f) => {
                    let operands = operands
                        .iter()
                        .map(|form| self.evaluate_analyzed_form(form))
                        .collect::<Result<Vec<RuntimeValue>, _>>()?;
                    f.apply(self, &operands)
                }
                v => Err(EvaluationError::CannotInvoke(v)),
            }
        } else {
            Ok(RuntimeValue::List(PersistentList::new()))
        }
    }

    fn evaluate_special_form(&mut self, form: &SpecialForm) -> EvaluationResult<RuntimeValue> {
        match form {
            SpecialForm::Def(form) => self.eval_def(form),
            SpecialForm::Var(symbol) => self.eval_var(symbol),
            SpecialForm::Let(form) => self.eval_let(form),
            SpecialForm::Loop(form) => self.eval_loop(form),
            SpecialForm::Recur(form) => self.eval_recur(form),
            SpecialForm::If(form) => self.eval_if(form),
            SpecialForm::Do(form) => self.eval_body_form(&form),
            SpecialForm::Fn(form) => self.eval_fn(form),
            SpecialForm::Quote(form) => self.eval_quote(form),
            SpecialForm::Quasiquote(form) => self.eval_quasiquote(form),
            SpecialForm::Defmacro(name, form) => self.eval_defmacro(name, form),
            SpecialForm::Macroexpand(form) => self.eval_macroexpand(form),
            SpecialForm::Try(form) => self.eval_try(form),
        }
    }

    fn evaluate_analyzed_form(&mut self, form: &RuntimeValue) -> EvaluationResult<RuntimeValue> {
        let value = match form {
            RuntimeValue::Nil => form.clone(),
            RuntimeValue::Bool(..) => form.clone(),
            RuntimeValue::Number(..) => form.clone(),
            RuntimeValue::String(..) => form.clone(),
            RuntimeValue::Keyword(..) => form.clone(),
            RuntimeValue::Symbol(symbol) => self.resolve_symbol_to_value(symbol)?,
            RuntimeValue::LexicalSymbol(identifier) => self.resolve_in_lexical_scopes(identifier),
            RuntimeValue::Var(var) => var.value(),
            RuntimeValue::List(coll) => self.evaluate_analyzed_list(coll)?,
            RuntimeValue::Vector(coll) => {
                let evaluated_coll = coll
                    .iter()
                    .map(|form| self.evaluate_analyzed_form(form))
                    .collect::<Result<PersistentVector<RuntimeValue>, _>>()?;
                RuntimeValue::Vector(evaluated_coll)
            }
            RuntimeValue::Map(coll) => {
                let mut evaluated_coll = PersistentMap::new();
                for (key, value) in coll {
                    let evaluated_key = self.evaluate_analyzed_form(key)?;
                    let evaluated_value = self.evaluate_analyzed_form(value)?;
                    evaluated_coll.insert_mut(evaluated_key, evaluated_value);
                }
                RuntimeValue::Map(evaluated_coll)
            }
            RuntimeValue::Set(coll) => {
                let mut evaluated_coll = PersistentSet::new();
                for elem in coll {
                    let analyzed_elem = self.evaluate_analyzed_form(elem)?;
                    evaluated_coll.insert_mut(analyzed_elem);
                }
                RuntimeValue::Set(evaluated_coll)
            }
            RuntimeValue::SpecialForm(form) => self.evaluate_special_form(form)?,
            RuntimeValue::Fn(..) => {
                // TODO?
                form.clone()
            }
            RuntimeValue::Primitive(f) => {
                // TODO?
                form.clone()
            }
            RuntimeValue::Exception(exc) => {
                // TODO?
                form.clone()
            }
        };
        Ok(value)
    }

    // Evaluate `form` in the global scope of the interpreter.
    // This method is exposed for the `eval` primitive which
    // has these semantics.
    pub(crate) fn evaluate_in_global_scope(
        &mut self,
        form: &RuntimeValue,
    ) -> EvaluationResult<RuntimeValue> {
        let mut child_scopes: Vec<_> = self.scopes.drain(0..).collect();
        let result = self.evaluate_analyzed_form(form);
        self.scopes.append(&mut child_scopes);
        result
    }

    pub(crate) fn analyze_and_evaluate(&mut self, form: &Form) -> EvaluationResult<RuntimeValue> {
        let analyzed_form = self.analyzer.analyze(form)?;
        self.evaluate_analyzed_form(&analyzed_form)
    }

    pub fn interpret(&mut self, source: &str) -> EvaluationResult<Vec<RuntimeValue>> {
        let result = read(source)
            .map_err(|err| EvaluationError::ReaderError(err, source.to_string()))?
            .iter()
            .map(|form| self.analyze_and_evaluate(form))
            .collect::<Result<Vec<RuntimeValue>, _>>();
        self.analyzer.reset();
        result
    }
}

#[cfg(test)]
mod test {
    use crate::collections::{PersistentList, PersistentMap, PersistentVector};
    use crate::namespace::new_var;
    use crate::reader::{self, Symbol};
    use crate::testing::run_eval_test;
    use crate::value::{
        exception,
        RuntimeValue::{self, *},
    };

    fn read_one_value(input: &str) -> RuntimeValue {
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
            (
                ":hi",
                Keyword(Symbol {
                    identifier: "hi".to_string(),
                    namespace: None,
                }),
            ),
            (
                ":foo/hi",
                Keyword(Symbol {
                    identifier: "hi".to_string(),
                    namespace: Some("foo".to_string()),
                }),
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
                RuntimeValue::Vector(PersistentVector::from_iter(vec![
                    Number(1),
                    Number(2),
                    Number(3),
                ])),
            ),
            (
                "{\"a\" (+ 7 8)}",
                RuntimeValue::Map(PersistentMap::from_iter(vec![(
                    String("a".to_string()),
                    Number(15),
                )])),
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
            ("(def! a 3)", RuntimeValue::Var(new_var(Number(3)))),
            ("(def! a 3) (+ a 1)", Number(4)),
            ("(def! a (+ 1 7)) (+ a 1)", Number(9)),
            ("(def! some-num 3)", RuntimeValue::Var(new_var(Number(3)))),
            ("(def! SOME-NUM 4)", RuntimeValue::Var(new_var(Number(4)))),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_let() {
        let test_cases = vec![
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
                RuntimeValue::Vector(PersistentVector::from_iter(vec![
                    Number(3),
                    Number(4),
                    Number(5),
                    RuntimeValue::Vector(PersistentVector::from_iter(vec![Number(6), Number(7)])),
                    Number(8),
                ])),
            ),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_advanced_let() {
        let test_cases =
            vec![
            (
                "(let* [cst (fn* [n] (if (= n 0) :success (cst (- n 1))))] (cst 1))",
                Keyword(Symbol {
                    identifier: "success".to_string(),
                    namespace: None,
                }),
            ),
            (
                "(let* [f (fn* [n] (if (= n 0) :success (g (- n 1)))) g (fn* [n] (f n))] (f 2))",
                Keyword(Symbol {
                    identifier: "success".to_string(),
                    namespace: None,
                }),
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
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_advanced_fn() {
        let test_cases = vec![
            ("(((fn* [a] (fn* [b] (+ a b))) 5) 7)", Number(12)),
            ("(def! gen-plus-x (fn* [x] (fn* [b] (+ x b)))) (def! plus7 (gen-plus-x 7)) (plus7 8)", Number(15)),
            ("((((fn* [a] (fn* [b] (fn* [c] (+ a b c)))) 1) 2) 3)", Number(6)),
            ("(((fn* [a] (fn* [b] (* b ((fn* [c] (+ a c)) 32)))) 1) 2)", Number(66)),
            ("(def! f (fn* [a] (fn* [b] (+ a b)))) ((first (let* [x 12] (map (fn* [_] (f x)) '(10000000)))) 27)", Number(39)),
            // test `let*` bindings inside a `fn*`
            (
                "(defn f [] (let* [cst (fn* [n] (if (= n 0) :success (cst (- n 1))))] (cst 10))) (f)",
                Keyword(Symbol {
                    identifier: "success".to_string(),
                    namespace: None,
                }),
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
        unimplemented!();
        // let test_cases = vec![
        //     ("(atom 5)", atom_with_value(Number(5))),
        //     ("(atom? (atom 5))", Bool(true)),
        //     ("(atom? nil)", Bool(false)),
        //     ("(atom? 1)", Bool(false)),
        //     ("(def! a (atom 5)) (deref a)", Number(5)),
        //     ("(def! a (atom 5)) @a", Number(5)),
        //     ("(def! a (atom (fn* [a] (+ a 1)))) (@a 4)", Number(5)),
        //     ("(def! a (atom 5)) (reset! a 10)", Number(10)),
        //     ("(def! a (atom 5)) (reset! a 10) @a", Number(10)),
        //     (
        //         "(def! a (atom 5)) (def! inc (fn* [x] (+ x 1))) (swap! a inc)",
        //         Number(6),
        //     ),
        //     ("(def! a (atom 5)) (swap! a + 1 2 3 4 5)", Number(20)),
        //     (
        //         "(def! a (atom 5)) (swap! a + 1 2 3 4 5) (deref a)",
        //         Number(20),
        //     ),
        //     (
        //         "(def! a (atom 5)) (swap! a + 1 2 3 4 5) (reset! a 10)",
        //         Number(10),
        //     ),
        //     (
        //         "(def! a (atom 5)) (swap! a + 1 2 3 4 5) (reset! a 10) (deref a)",
        //         Number(10),
        //     ),
        //     (
        //         "(def! a (atom 7)) (def! f (fn* [] (swap! a inc))) (f) (f)",
        //         Number(9),
        //     ),
        //     (
        //         "(def! g (let* [a (atom 0)] (fn* [] (swap! a inc)))) (def! a (atom 1)) (g) (g) (g)",
        //         Number(3),
        //     ),
        //     (
        //         "(def! e (atom {:+ +})) (swap! e assoc :- -) ((get @e :+) 7 8)",
        //         Number(15),
        //     ),
        //     (
        //         "(def! e (atom {:+ +})) (swap! e assoc :- -) ((get @e :-) 11 8)",
        //         Number(3),
        //     ),
        //     (
        //         "(def! e (atom {:+ +})) (swap! e assoc :- -) (swap! e assoc :foo ()) (get @e :foo)",
        //         list_with_values(vec![]),
        //     ),
        //     (
        //         "(def! e (atom {:+ +})) (swap! e assoc :- -) (swap! e assoc :bar '(1 2 3)) (get @e :bar)",
        //         list_with_values(vec![Number(1), Number(2), Number(3)]),
        //     ),
        // ];
        // run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_quote() {
        let test_cases = vec![
            ("(quote 5)", Number(5)),
            (
                "(quote (1 2 3))",
                RuntimeValue::List(PersistentList::from_iter(
                    [Number(1), Number(2), Number(3)].iter().cloned(),
                )),
            ),
            (
                "(quote (1 2 (into+ [] foo :baz/bar)))",
                RuntimeValue::List(PersistentList::from_iter(
                    [
                        Number(1),
                        Number(2),
                        RuntimeValue::List(PersistentList::from_iter(
                            [
                                Symbol(Symbol {
                                    identifier: "into+".to_string(),
                                    namespace: None,
                                }),
                                Vector(PersistentVector::new()),
                                Symbol(Symbol {
                                    identifier: "foo".to_string(),
                                    namespace: None,
                                }),
                                Keyword(Symbol {
                                    identifier: "bar".to_string(),
                                    namespace: Some("baz".to_string()),
                                }),
                            ]
                            .iter()
                            .cloned(),
                        )),
                    ]
                    .iter()
                    .cloned(),
                )),
            ),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_quasiquote() {
        let test_cases = vec![
            ("(quasiquote nil)", Nil),
            ("(quasiquote ())", RuntimeValue::List(PersistentList::new())),
            ("(quasiquote 7)", Number(7)),
            ("(quasiquote a)", Symbol(Symbol {
                identifier: "a".to_string(),
                namespace: None,
            })),
            (
                "(quasiquote {:a b})",
                RuntimeValue::Map(PersistentMap::from_iter(vec![(
                    Keyword(Symbol {
                        identifier: "a".to_string(),
                        namespace: None,
                    }),
                    Symbol(Symbol {
                        identifier: "b".to_string(),
                        namespace: None,
                    }),
                )])),
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
            ("(def! a 8) `a", Symbol(Symbol {
                identifier: "a".to_string(),
                namespace: None,
            })),
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
            ("(let* [x (cond false :no true :yes)] x)", Keyword(Symbol {
                identifier: "yes".to_string(),
                namespace: None,
            })),
            ("(macroexpand '(cond X Y Z T))",
                read_one_value("(if X Y (cond Z T))")
            ),
            ("(def! x 2) (defmacro! a (fn* [] x)) (a)", Number(2)),
            ("(def! x 2) (defmacro! a (fn* [] x)) (let* [x 3] (a))", Number(2)),
            ("(def! f (fn* [x] (number? x))) (defmacro! m f) [(f (+ 1 1)) (m (+ 1 1))]", RuntimeValue::Vector(PersistentVector::from_iter(vec![Bool(true), Bool(false)]))),
        ];
        run_eval_test(&test_cases);
    }

    #[test]
    fn test_basic_try_catch() {
        fn exception_value(msg: &str, data: &RuntimeValue) -> RuntimeValue {
            RuntimeValue::Exception(exception(msg, data.clone()))
        }

        let exc = exception_value(
            "test",
            &RuntimeValue::Map(PersistentMap::from_iter(vec![(
                Keyword(Symbol {
                    identifier: "cause".to_string(),
                    namespace: None,
                }),
                String("no memory".to_string()),
            )])),
        );
        let test_cases = vec![
            // NOTE: these are errors from uncaught exceptions now...
            // TODO: map to evaluation error test cases
            // let basic_exc = exception_value("", &String("test".to_string()));
            // ( "(throw \"test\")", basic_exc),
            // ( "(throw {:msg :foo})", exception_value("", &map_with_values(vec![(Keyword("msg".to_string(), None), Keyword("foo".to_string(), None))]))),
            (
                "(try* (throw '(1 2 3)) (catch* e e))",
                exception_value("", &RuntimeValue::List(PersistentList::from_iter(vec![Number(1), Number(2), Number(3)]))),
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
                    &RuntimeValue::Map(PersistentMap::from_iter(
                        [(
                            Keyword(Symbol {
                                identifier: "cause".to_string(),
                                namespace: None,
                            }),
                            String("no memory".to_string()),
                        )]
                        .iter()
                        .cloned(),
                    )),
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
                Keyword(Symbol {
                    identifier: "outer".to_string(),
                    namespace: None,
                }),
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
            (
                "*command-line-args*",
                RuntimeValue::List(PersistentList::new()),
            ),
        ];
        run_eval_test(&test_cases);
    }
}
