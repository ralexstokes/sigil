use crate::interpreter::{EvaluationError, EvaluationResult, Interpreter, Scope, SyntaxError};
use crate::value::{
    FnImpl, FnWithCapturesImpl, PersistentList, PersistentMap, PersistentSet, PersistentVector,
    Value,
};
use itertools::Itertools;
use std::collections::HashSet;
use std::fmt::Write;
use std::iter::FromIterator;

const MIN_VARIADIC_PARAM_COUNT: usize = 2;

type BindingRef<'a> = (&'a String, &'a Value);
// each new `fn*` introduces a new "frame"
// forms within a `fn*` can introduce a new "scope"
#[derive(Default, Debug)]
struct Frame {
    scopes: Vec<Scope>,
    forward_declarations: Vec<Scope>,
}
// ref to a Frame in set of Frames and an identifier within that Frame
type CaptureSet = HashSet<(usize, String)>;

pub struct LetBindings<'a> {
    bindings: Vec<BindingRef<'a>>,
}

fn binding_declares_fn((name, value): &BindingRef) -> Option<String> {
    match value {
        Value::List(elems) => match elems.first() {
            Some(Value::Symbol(s, None)) if s == "fn*" => Some(name.to_string()),
            _ => None,
        },
        _ => None,
    }
}

impl<'a> LetBindings<'a> {
    // allow let bindings that declare `fn*`s to capture other
    // let bindings that declare `fn*`s
    pub fn resolve_forward_declarations(&self) -> HashSet<String> {
        self.bindings
            .iter()
            .filter_map(binding_declares_fn)
            .collect()
    }
}

impl<'a> IntoIterator for LetBindings<'a> {
    type Item = BindingRef<'a>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.bindings.into_iter()
    }
}

pub struct LetForm<'a> {
    pub bindings: LetBindings<'a>,
    pub body: PersistentList<Value>,
}

fn parse_let_bindings(bindings_form: &Value) -> EvaluationResult<LetBindings> {
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
                        other => {
                            return Err(SyntaxError::LexicalBindingsMustHaveSymbolNames(
                                other.clone(),
                            )
                            .into());
                        }
                    }
                }
                Ok(LetBindings {
                    bindings: validated_bindings,
                })
            } else {
                Err(SyntaxError::LexicalBindingsMustBePaired(bindings.clone()).into())
            }
        }
        other => Err(SyntaxError::LexicalBindingsMustBeVector(other.clone()).into()),
    }
}

fn parse_let(forms: &PersistentList<Value>) -> EvaluationResult<LetForm> {
    let bindings_form = forms.first().ok_or(EvaluationError::WrongArity {
        expected: 1,
        realized: 0,
    })?;
    let body = forms.drop_first().ok_or(EvaluationError::WrongArity {
        expected: 2,
        realized: 1,
    })?;
    let bindings = parse_let_bindings(bindings_form)?;
    Ok(LetForm { bindings, body })
}

pub(crate) fn analyze_let(let_forms: &PersistentList<Value>) -> EvaluationResult<LetForm> {
    let let_form = parse_let(let_forms)?;
    Ok(let_form)
}

pub(crate) fn lambda_parameter_key(index: usize, level: usize) -> String {
    let mut key = String::new();
    let _ = write!(&mut key, ":system-fn-%{}/{}", index, level);
    key
}

pub struct Analyzer<'a> {
    interpreter: &'a mut Interpreter,
}

impl<'a> Analyzer<'a> {
    fn new(interpreter: &'a mut Interpreter) -> Self {
        Self { interpreter }
    }

    fn analyze_lexical_bindings_in_fn(
        &mut self,
        bindings: &PersistentVector<Value>,
        frames: &mut Vec<Frame>,
        captures: &mut Vec<CaptureSet>,
    ) -> EvaluationResult<Value> {
        if bindings.len() % 2 != 0 {
            return Err(SyntaxError::LexicalBindingsMustBePaired(bindings.clone()).into());
        }
        let mut analyzed_bindings = PersistentVector::new();
        // NOTE: this is duplicated w/ `let*` analysis elsewhere...
        // TODO: consolidate to one analysis phase
        let mut forward_declarations = Scope::new();
        for (name, value) in bindings.iter().tuples() {
            match name {
                Value::Symbol(s, None) => {
                    if binding_declares_fn(&(s, value)).is_some() {
                        forward_declarations.insert(s.clone(), Value::Symbol(s.clone(), None));
                    }
                }
                other => {
                    return Err(
                        SyntaxError::LexicalBindingsMustHaveSymbolNames(other.clone()).into(),
                    );
                }
            }
        }
        let bindings_scope_index = {
            let frame = frames.last_mut().expect("did push");
            frame.scopes.push(Scope::new());
            frame.forward_declarations.push(forward_declarations);
            frame.scopes.len() - 1
        };
        for (name, value) in bindings.iter().tuples() {
            let analyzed_value = self.analyze_form_in_fn(value, frames, captures)?;
            analyzed_bindings.push_back_mut(name.clone());
            analyzed_bindings.push_back_mut(analyzed_value);
            // lexical bindings serially extend scope per binding:
            match name {
                Value::Symbol(s, None) => {
                    let local_scopes = &mut frames.last_mut().expect("already pushed").scopes;
                    let scope = local_scopes
                        .get_mut(bindings_scope_index)
                        .expect("did push bindings scope");
                    scope.insert(s.clone(), Value::Symbol(s.clone(), None));
                }
                _ => unreachable!("already verified symbol names"),
            }
        }
        let frame = frames.last_mut().expect("did push");
        frame.forward_declarations.pop();
        Ok(Value::Vector(analyzed_bindings))
    }

    // Note: can only have captures over enclosing fns if we have recursive nesting of fns
    fn analyze_fn_in_fn_with_possible_captures(
        &mut self,
        body: PersistentList<Value>,
        bindings: &PersistentVector<Value>,
        frames: &mut Vec<Frame>,
        captures: &mut Vec<CaptureSet>,
    ) -> EvaluationResult<Value> {
        captures.push(CaptureSet::new());
        let analyzed_fn = self.analyze_symbols_in_fn(body, bindings, frames, captures)?;
        let captures_at_this_level = captures.pop().expect("did push");
        if captures_at_this_level.is_empty() {
            return Ok(analyzed_fn);
        }
        let current_frame_index = frames.len() - 1;
        match analyzed_fn {
            Value::Fn(f) => {
                // Note: need to hoist captures if there are intervening functions along the way...
                for (captured_frame_index, capture) in &captures_at_this_level {
                    if *captured_frame_index < current_frame_index {
                        let target_captures = captures
                            .get_mut(*captured_frame_index)
                            .expect("already pushed captures");
                        target_captures.insert((*captured_frame_index, capture.to_string()));
                    }
                }
                let captures = captures_at_this_level
                    .iter()
                    .map(|(_, capture)| (capture.to_string(), None))
                    .collect();
                Ok(Value::FnWithCaptures(FnWithCapturesImpl { f, captures }))
            }
            _ => unreachable!("only returns Fn variant"),
        }
    }

    fn analyze_list_in_fn(
        &mut self,
        elems: &PersistentList<Value>,
        frames: &mut Vec<Frame>,
        captures: &mut Vec<CaptureSet>,
    ) -> EvaluationResult<Value> {
        let existing_scopes_count = {
            let local_scopes = &frames
                .last_mut()
                .expect("did push on analysis entry")
                .scopes;
            local_scopes.len()
        };

        // if first elem introduces a new lexical scope...
        let mut iter = elems.iter();
        let mut analyzed_elems = vec![];
        match iter.next() {
            Some(Value::Symbol(s, None)) if s == "let*" => {
                analyzed_elems.push(Value::Symbol(s.to_string(), None));
                if let Some(Value::Vector(bindings)) = iter.next() {
                    let analyzed_bindings =
                        self.analyze_lexical_bindings_in_fn(bindings, frames, captures)?;
                    analyzed_elems.push(analyzed_bindings);
                }
            }
            Some(Value::Symbol(s, None)) if s == "loop*" => {
                analyzed_elems.push(Value::Symbol(s.to_string(), None));
                if let Some(Value::Vector(bindings)) = iter.next() {
                    let analyzed_bindings =
                        self.analyze_lexical_bindings_in_fn(bindings, frames, captures)?;
                    analyzed_elems.push(analyzed_bindings);
                }
            }
            Some(Value::Symbol(s, None)) if s == "fn*" => {
                if let Some(Value::Vector(bindings)) = iter.next() {
                    let body = iter.cloned().collect();
                    return self
                        .analyze_fn_in_fn_with_possible_captures(body, bindings, frames, captures);
                }
            }
            Some(Value::Symbol(s, None)) if s == "catch*" => {
                if let Some(Value::Symbol(s, None)) = iter.next() {
                    let mut bindings = PersistentVector::new();
                    bindings.push_back_mut(Value::Symbol(s.clone(), None));
                    let body = iter.cloned().collect();
                    return self.analyze_fn_in_fn_with_possible_captures(
                        body, &bindings, frames, captures,
                    );
                }
            }
            Some(Value::Symbol(s, None)) if s == "quote" => {
                if let Some(Value::Symbol(s, None)) = iter.next() {
                    let mut scope = Scope::new();
                    scope.insert(s.to_string(), Value::Symbol(s.to_string(), None));
                    let local_scopes = &mut frames.last_mut().expect("did push").scopes;
                    local_scopes.push(scope);
                }
            }
            _ => {}
        }
        for elem in elems.iter().skip(analyzed_elems.len()) {
            let analyzed_elem = self.analyze_form_in_fn(elem, frames, captures)?;
            analyzed_elems.push(analyzed_elem);
        }
        let local_scopes = &mut frames.last_mut().expect("did push").scopes;
        local_scopes.truncate(existing_scopes_count);
        Ok(Value::List(PersistentList::from_iter(analyzed_elems)))
    }

    // Analyze symbols (recursively) in `form`:
    // 1. Rewrite lambda parameters
    // 2. Capture references to external vars
    fn analyze_form_in_fn(
        &mut self,
        form: &Value,
        frames: &mut Vec<Frame>,
        captures: &mut Vec<CaptureSet>,
    ) -> EvaluationResult<Value> {
        match form {
            Value::Symbol(identifier, ns_opt) => {
                let current_frame_index = frames.len() - 1;
                for (frame_index, frame) in frames.iter().enumerate().rev() {
                    // NOTE: for now, need to side step rest of symbol resolution if a symbol
                    // is part of a forward declaration...
                    for scope in frame.forward_declarations.iter().rev() {
                        if let Some(Value::Symbol(resolved_identifier, None)) =
                            scope.get(identifier)
                        {
                            return Ok(Value::Symbol(resolved_identifier.clone(), None));
                        }
                    }
                    for scope in frame.scopes.iter().rev() {
                        match scope.get(identifier) {
                            Some(Value::Symbol(resolved_identifier, None)) => {
                                let reference_outlives_source = frame_index < current_frame_index;
                                // NOTE: current particularity of the implementation is to _not_
                                // capture forward declarations from `let*` bindings...
                                if reference_outlives_source {
                                    let captures_at_level = captures
                                        .last_mut()
                                        .expect("did push captures to grab earlier frame");
                                    // TODO: work through lifetimes here to avoid cloning...
                                    captures_at_level
                                        .insert((frame_index, resolved_identifier.clone()));
                                }
                                return Ok(Value::Symbol(resolved_identifier.clone(), None));
                            }
                            Some(other) => {
                                unreachable!("encountered unexpected value in `Scope`: {}", other)
                            }
                            None => {}
                        }
                    }
                }
                self.interpreter
                    .resolve_symbol_to_var(identifier, ns_opt.as_ref())
            }
            Value::List(elems) => {
                if elems.is_empty() {
                    return Ok(Value::List(PersistentList::new()));
                }

                let first = elems.first().unwrap();
                let rest = elems.drop_first().expect("list is not empty");
                if let Some(expansion) = self.interpreter.get_macro_expansion(first, &rest) {
                    match expansion? {
                        Value::List(elems) => self.analyze_list_in_fn(&elems, frames, captures),
                        other => self.analyze_form_in_fn(&other, frames, captures),
                    }
                } else {
                    self.analyze_list_in_fn(elems, frames, captures)
                }
            }
            Value::Vector(elems) => {
                let mut analyzed_elems = PersistentVector::new();
                for elem in elems.iter() {
                    let analyzed_elem = self.analyze_form_in_fn(elem, frames, captures)?;
                    analyzed_elems.push_back_mut(analyzed_elem);
                }
                Ok(Value::Vector(analyzed_elems))
            }
            Value::Map(elems) => {
                let mut analyzed_elems = PersistentMap::new();
                for (k, v) in elems.iter() {
                    let analyzed_k = self.analyze_form_in_fn(k, frames, captures)?;
                    let analyzed_v = self.analyze_form_in_fn(v, frames, captures)?;
                    analyzed_elems.insert_mut(analyzed_k, analyzed_v);
                }
                Ok(Value::Map(analyzed_elems))
            }
            Value::Set(elems) => {
                let mut analyzed_elems = PersistentSet::new();
                for elem in elems.iter() {
                    let analyzed_elem = self.analyze_form_in_fn(elem, frames, captures)?;
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

    fn extract_scope_from_fn_bindings(
        &self,
        params: &PersistentVector<Value>,
        level: usize,
    ) -> EvaluationResult<(Scope, bool)> {
        let mut parameters = Scope::new();
        let mut variadic = false;
        let params_count = params.len();
        for (index, param) in params.iter().enumerate() {
            match param {
                Value::Symbol(s, None) if s == "&" => {
                    if index + MIN_VARIADIC_PARAM_COUNT > params_count {
                        return Err(SyntaxError::VariadicArgMissing.into());
                    }
                    variadic = true;
                }
                Value::Symbol(s, None) => {
                    if variadic {
                        if index + 1 != params_count {
                            return Err(SyntaxError::VariadicArgMustBeUnique(Value::Vector(
                                params.clone(),
                            ))
                            .into());
                        }

                        let parameter = lambda_parameter_key(index - 1, level);
                        parameters.insert(s.to_string(), Value::Symbol(parameter, None));
                    } else {
                        let parameter = lambda_parameter_key(index, level);
                        parameters.insert(s.to_string(), Value::Symbol(parameter, None));
                    }
                }
                other => {
                    return Err(
                        SyntaxError::LexicalBindingsMustHaveSymbolNames(other.clone()).into(),
                    );
                }
            }
        }
        Ok((parameters, variadic))
    }

    // Non-local symbols should:
    // 1. resolve to a parameter
    // 2. resolve to a value in the enclosing environment, which is captured
    // otherwise, the lambda is an error
    //
    // Note: parameters are resolved to (ordinal) reserved symbols
    fn analyze_symbols_in_fn(
        &mut self,
        body: PersistentList<Value>,
        params: &PersistentVector<Value>,
        frames: &mut Vec<Frame>,
        // record any values captured from the environment that would outlive the lifetime of this particular lambda
        captures: &mut Vec<CaptureSet>,
    ) -> EvaluationResult<Value> {
        let level = frames.len();
        let (parameters, variadic) = self.extract_scope_from_fn_bindings(params, level)?;
        let arity = if variadic {
            parameters.len() - 1
        } else {
            parameters.len()
        };
        let mut frame = Frame::default();
        frame.scopes.push(parameters);

        frames.push(frame);
        // walk the `body`, resolving symbols where possible...
        let mut analyzed_body = Vec::with_capacity(body.len());
        for form in body.iter() {
            let analyzed_form = self.analyze_form_in_fn(form, frames, captures)?;
            analyzed_body.push(analyzed_form);
        }
        frames.pop();
        Ok(Value::Fn(FnImpl {
            body: analyzed_body.into_iter().collect(),
            arity,
            level,
            variadic,
        }))
    }
}

pub fn analyze_fn(
    interpreter: &mut Interpreter,
    body: PersistentList<Value>,
    params: &PersistentVector<Value>,
) -> EvaluationResult<Value> {
    let mut analyzer = Analyzer::new(interpreter);
    let mut frames = vec![];
    let mut captures = vec![];
    analyzer.analyze_symbols_in_fn(body, params, &mut frames, &mut captures)
}
