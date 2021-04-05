use crate::prelude::{count, divide, is_empty, is_list, list, multiply, plus, pr, prn, subtract};
use crate::value::{var_impl_into_inner, var_into_inner, var_with_value, Lambda, Value};
use itertools::Itertools;
use rpds::{
    HashTrieMap as PersistentMap, HashTrieSet as PersistentSet, List as PersistentList,
    Vector as PersistentVector,
};
use std::collections::HashMap;
use std::default::Default;
use std::fmt::Write;
use std::iter::FromIterator;
use thiserror::Error;

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
pub enum EvaluationError {
    #[error("symbol error: {0}")]
    Symbol(SymbolEvaluationError),
    #[error("list error: {0}")]
    List(ListEvaluationError),
    #[error("primitive error: {0}")]
    Primitve(PrimitiveEvaluationError),
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
        ];
        let global_scope =
            HashMap::from_iter(bindings.iter().map(|(k, v)| (k.to_string(), v.clone())));

        let mut default_namespaces = HashMap::new();
        default_namespaces.insert(DEFAULT_NAMESPACE.to_string(), Namespace::default());

        Interpreter {
            current_namespace: DEFAULT_NAMESPACE.to_string(),
            namespaces: default_namespaces,
            scopes: vec![global_scope],
        }
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

    fn enter_scope(&mut self) -> usize {
        self.scopes.push(Scope::default());
        self.scopes.len()
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
        // local scopes to the lambda context
        scopes: &mut Vec<Scope>,
    ) -> EvaluationResult<Value> {
        match form {
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
                let mut elems = elems.iter();
                if let Some(elem) = elems.next() {
                    match elem {
                        // handle `let*` forms with special care...
                        Value::Symbol(s, None) if s == "let*" => {
                            analyzed_elems.push(Value::Symbol("let*".to_string(), None));
                            if let Some(bindings) = elems.next() {
                                match bindings {
                                    Value::Vector(bindings) => {
                                        let mut let_scope = Scope::new();
                                        let mut analyzed_bindings = PersistentVector::new();
                                        let mut pairs = bindings.iter().tuples();
                                        while let Some((symbol, binding)) = pairs.next() {
                                            match symbol {
                                                Value::Symbol(identifier, None) => {
                                                    // let bindings resolve to themselves to
                                                    // preserve meaning during further syntactic analysis
                                                    let analyzed_symbol =
                                                        Value::Symbol(identifier.to_string(), None);
                                                    let_scope.insert(
                                                        identifier.to_string(),
                                                        analyzed_symbol.clone(),
                                                    );
                                                    scopes.push(let_scope.clone());
                                                    let analyzed_binding = self
                                                        .analyze_form_in_lambda(binding, scopes)?;
                                                    scopes.pop();
                                                    analyzed_bindings
                                                        .push_back_mut(analyzed_symbol);
                                                    analyzed_bindings
                                                        .push_back_mut(analyzed_binding);
                                                }
                                                symbol @ _ => {
                                                    // invalid let binding
                                                    // drain rest of form and bail...
                                                    analyzed_bindings.push_back_mut(symbol.clone());
                                                    analyzed_bindings
                                                        .push_back_mut(binding.clone());
                                                    while let Some((symbol, binding)) = pairs.next()
                                                    {
                                                        analyzed_bindings
                                                            .push_back_mut(symbol.clone());
                                                        analyzed_bindings
                                                            .push_back_mut(binding.clone());
                                                    }
                                                    analyzed_elems
                                                        .push(Value::Vector(analyzed_bindings));
                                                    while let Some(elem) = elems.next() {
                                                        analyzed_elems.push(elem.clone());
                                                    }
                                                    // NOTE: not "Ok" but currently no way to signal errors at
                                                    // the analysis stage...
                                                    return Ok(Value::List(
                                                        PersistentList::from_iter(analyzed_elems),
                                                    ));
                                                }
                                            }
                                        }
                                        analyzed_elems.push(Value::Vector(analyzed_bindings));
                                        // analyze rest of let* form with extended scope
                                        scopes.push(let_scope);
                                        while let Some(elem) = elems.next() {
                                            let analyzed_elem =
                                                self.analyze_form_in_lambda(elem, scopes)?;
                                            analyzed_elems.push(analyzed_elem);
                                        }
                                        scopes.pop();
                                    }
                                    // bindings are not in a vector...
                                    // drain rest of list and bail...
                                    bindings @ _ => {
                                        analyzed_elems.push(bindings.clone());
                                        while let Some(elem) = elems.next() {
                                            analyzed_elems.push(elem.clone());
                                        }
                                    }
                                }
                            }
                        }
                        // and handle `fn*` forms with special care...
                        Value::Symbol(s, None) if s == "fn*" => {
                            analyzed_elems.push(Value::Symbol("fn*".to_string(), None));
                            if let Some(bindings) = elems.next() {
                                match bindings {
                                    Value::Vector(params) => {
                                        let mut forms = vec![];
                                        while let Some(form) = elems.next() {
                                            forms.push(form.clone());
                                        }
                                        return self.analyze_lambda(
                                            PersistentList::from_iter(forms.into_iter()),
                                            params,
                                            scopes,
                                        );
                                    }
                                    // bindings are not in a vector...
                                    // drain rest of list and bail...
                                    bindings @ _ => {
                                        analyzed_elems.push(bindings.clone());
                                        while let Some(elem) = elems.next() {
                                            analyzed_elems.push(elem.clone());
                                        }
                                    }
                                }
                            }
                        }
                        // otherwise, just map analysis over the list...
                        elem @ _ => {
                            // analyze first form
                            let analyzed_elem = self.analyze_form_in_lambda(elem, scopes)?;
                            analyzed_elems.push(analyzed_elem);
                            // rest of body is not special wrt scope, analyze and proceed
                            while let Some(elem) = elems.next() {
                                let analyzed_elem = self.analyze_form_in_lambda(elem, scopes)?;
                                analyzed_elems.push(analyzed_elem);
                            }
                        }
                    }
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
            other @ _ => Ok(other.clone()),
        }
    }

    // Non-local symbols should:
    // 1. resolve to a parameter
    // 2. resolve to a value in the enclosing environment, which is captured
    // otherwise, the lambda is an error
    //
    // Note: parameters are resolved to (ordinal) reserved symbols
    fn analyze_lambda(
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
                        // (if predicate consequent alternate?)
                        Value::Symbol(s, None) if s == "if" => {
                            if let Some(rest) = forms.drop_first() {
                                let mut forms = vec![];
                                for form in rest.iter() {
                                    let value = self.evaluate(form)?;
                                    forms.push(value);
                                }
                                match forms.len() {
                                    n @ 2 | n @ 3 => match &forms[0] {
                                        &Value::Bool(predicate) => {
                                            if predicate {
                                                // consequent
                                                return Ok(forms[1].clone());
                                            } else {
                                                if n == 2 {
                                                    // false predicate with no alternate
                                                    return Ok(Value::Nil);
                                                } else {
                                                    // alternate
                                                    return Ok(forms[2].clone());
                                                }
                                            }
                                        }
                                        &Value::Nil => {
                                            if n == 2 {
                                                // false predicate with no alternate
                                                return Ok(Value::Nil);
                                            } else {
                                                // alternate
                                                return Ok(forms[2].clone());
                                            }
                                        }
                                        _ => {}
                                    },
                                    _ => {}
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
                                                return self.analyze_lambda(
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
                                return native_fn(&operands);
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
            Value::Fn(lambda) => Value::Fn(lambda.clone()),
            Value::Primitive(_) => unreachable!(),
            Value::Var(v) => var_impl_into_inner(v),
        };
        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::reader::read;
    use crate::value::list_with_values;

    #[test]
    fn test_basic_eval() {
        use Value::*;

        let test_cases = vec![
            ("nil", Nil),
            ("1337", Number(1337)),
            ("-1337", Number(-1337)),
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
            ("(do 1 2 3)", Number(3)),
            ("(def! a 3)", var_with_value(Number(3))),
            ("(do (def! a 3) (+ a 1))", Number(4)),
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
            ("(do (def! a 1) (let* [a 3] a))", Number(3)),
            ("(do (def! a 1) (let* [a 3] a) a)", Number(1)),
            ("(do (def! b 1) (let* [a 3] (+ a b)))", Number(4)),
            ("(if true 1 2)", Number(1)),
            ("(if true 1)", Number(1)),
            ("(if false 1 2)", Number(2)),
            ("(if false 2)", Nil),
            ("(if nil 1 2)", Number(2)),
            ("(if nil 2)", Nil),
            ("(let* [b nil] (if b 2 3))", Number(3)),
            (
                "(list 1 2)",
                list_with_values([Number(1), Number(2)].iter().map(|arg| arg.clone())),
            ),
            ("(list? (list 1))", Bool(true)),
            ("(list? [1 2])", Bool(false)),
            ("(empty? (list))", Bool(true)),
            ("(empty? (list 1))", Bool(false)),
            ("(count (list 44 42 41))", Number(3)),
            ("((fn* [a] (+ a 1)) 23)", Number(24)),
            ("((fn* [a] (let* [b 2] (+ a b))) 23)", Number(25)),
            ("((fn* [a] (let* [a 2] (+ a a))) 23)", Number(4)),
            (
                "(do (def! inc (fn* [a] (+ a 1))) ((fn* [a] (inc a)) 1))",
                Number(2),
            ),
            ("((fn* [a] ((fn* [b] (+ b 1)) a)) 1)", Number(2)),
            ("((fn* [] ((fn* [] ((fn* [] 13))))))", Number(13)),
            ("(do (def! f (fn* [a] (+ a 1))) (f 23))", Number(24)),
            (
                "(do (def! b 12) (def! f (fn* [a] (+ a b))) (def! b 22) (f 1))",
                Number(23),
            ),
            (
                "(do (def! b 12) (def! f (fn* [a] ((fn* [] (+ a b))))) (def! b 22) (f 1))",
                Number(23),
            ),
        ];

        for (input, expected) in &test_cases {
            let mut interpreter = Interpreter::default();
            let read_result = read(input).unwrap();
            // dbg!(&input);
            let form = &read_result[0];
            // dbg!(&form);
            let result = interpreter.evaluate(form).unwrap();
            // dbg!(&result);
            assert_eq!(result, *expected)
        }

        // try again while retaining state in the interpreter
        let mut interpreter = Interpreter::default();
        for (input, expected) in test_cases {
            let read_result = read(input).unwrap();
            // dbg!(&input);
            let form = &read_result[0];
            // dbg!(&form);
            let result = interpreter.evaluate(form).unwrap();
            // dbg!(&result);
            assert_eq!(result, expected)
        }
    }
}
