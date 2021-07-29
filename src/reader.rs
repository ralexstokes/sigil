use crate::value::{list_with_values, map_with_values, set_with_values, vector_with_values, Value};
use itertools::Itertools;
use std::num::ParseIntError;
use std::{iter::Peekable, str::CharIndices};
use thiserror::Error;

fn is_newline(input: char) -> bool {
    input == '\n'
}

fn is_whitespace(input: char) -> bool {
    char::is_whitespace(input) || input == ','
}

fn is_comment(input: char) -> bool {
    input == ';'
}

pub fn is_token(input: char) -> bool {
    !is_whitespace(input) && !is_comment(input)
}

fn is_numeric(input: char) -> bool {
    char::is_numeric(input)
}

pub fn is_symbolic(input: char) -> bool {
    match input {
        '*' | '+' | '!' | '-' | '_' | '\'' | '?' | '<' | '>' | '=' | '/' | '&' | ':' | '$'
        | '#' => true,
        _ => char::is_alphanumeric(input),
    }
}

pub fn is_structural(input: char) -> bool {
    matches!(
        input,
        '(' | ')' | '[' | ']' | '{' | '}' | '#' | '@' | '\'' | '`' | '~' | '"'
    )
}

fn parse_identifier_and_optional_namespace(
    symbolic: &str,
) -> Result<(String, Option<String>), ReaderError> {
    if symbolic.is_empty() {
        return Err(ReaderError::InvalidIdentifier);
    }
    if let Some((ns, identifier)) = symbolic.split_once('/') {
        match (ns, identifier) {
            ("", "") => Ok(("/".to_string(), None)),
            ("", _) => Err(ReaderError::MissingNamespace),
            (_, "") => Err(ReaderError::InvalidIdentifier),
            (namespace, identifier) => {
                if namespace.contains(':') {
                    return Err(ReaderError::InvalidNamespace);
                }
                if identifier.contains(':') {
                    return Err(ReaderError::InvalidIdentifier);
                }
                Ok((identifier.to_string(), Some(ns.to_string())))
            }
        }
    } else {
        if let Some((ns, identifier)) = symbolic.split_once(':') {
            if ns.is_empty() {
                if identifier.is_empty() {
                    return Err(ReaderError::InvalidIdentifier);
                }
                // `::identifier` form
                return Ok((identifier.to_string(), Some(":".to_string())));
            }
        }
        Ok((symbolic.to_string(), None))
    }
}

fn parse_symbolic_with_namespace(symbolic: &str) -> Result<Value, ReaderError> {
    if let Some(symbolic) = symbolic.strip_prefix(':') {
        if symbolic == "/" {
            return Err(ReaderError::InvalidIdentifier);
        }
        let (identifier, ns_opt) = parse_identifier_and_optional_namespace(symbolic)?;
        Ok(Value::Keyword(identifier, ns_opt))
    } else {
        let (identifier, ns_opt) = parse_identifier_and_optional_namespace(symbolic)?;
        Ok(Value::Symbol(identifier, ns_opt))
    }
}

fn parse_symbolic(symbolic: &str) -> Result<Value, ReaderError> {
    match symbolic {
        "nil" => Ok(Value::Nil),
        "true" => Ok(Value::Bool(true)),
        "false" => Ok(Value::Bool(false)),
        symbolic => parse_symbolic_with_namespace(symbolic),
    }
}

fn find_string_close(stream: &mut Stream) -> Result<usize, ReaderError> {
    while let Some((index, ch)) = stream.next() {
        match ch {
            '"' => {
                return Ok(index);
            }
            '\\' => {
                let (_, next_ch) = stream.next().ok_or(ReaderError::ExpectedMoreInput)?;
                if next_ch == '"' {
                    continue;
                }
            }
            _ => {}
        }
    }
    Err(ReaderError::UnbalancedString)
}

fn apply_string_escapes(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut iter = input.chars().peekable();
    while let Some(ch) = iter.next() {
        match ch {
            '\\' => {
                if let Some(next) = iter.peek() {
                    let next = *next;
                    match next {
                        '\\' => {
                            result.push('\\');
                            iter.next().expect("from peek");
                        }
                        'n' => {
                            result.push('\n');
                            iter.next().expect("from peek");
                        }
                        '"' => {
                            result.push('"');
                            iter.next().expect("from peek");
                        }
                        _ => {
                            result.push(ch);
                        }
                    };
                } else {
                    result.push(ch);
                }
            }
            ch => result.push(ch),
        }
    }
    result
}

type Stream<'a> = Peekable<CharIndices<'a>>;

#[derive(Debug)]
enum Range {
    ToEnd(usize),
    Slice(usize, usize),
}

#[derive(Debug, Error, Clone)]
pub enum ReaderError {
    #[error("error parsing number: {0}")]
    CouldNotParseNumber(#[from] ParseIntError),
    #[error("error negating number: {0}")]
    CouldNotNegateNumber(i64),
    #[error("unexpected input {0}")]
    UnexpectedInput(char),
    #[error("expected further input but found EOF")]
    ExpectedMoreInput,
    #[error("expected a namespace but none was given")]
    MissingNamespace,
    #[error("namespace for symbolic form was invalid in the given context")]
    InvalidNamespace,
    #[error("expected an identifier for symbol in namespace but none was given")]
    MissingIdentifier,
    #[error("identifier for symbolic form was invalid in the given context")]
    InvalidIdentifier,
    #[error("started reading a string but did not find the terminating `\"`")]
    UnbalancedString,
    #[error("unbalanced collection: missing closing {0}")]
    UnbalancedCollection(char),
    #[error("map literal given with unpaired entries")]
    MapLiteralWithUnpairedElements,
    #[error("could not parse dispatch with following char: #{0}")]
    CouldNotParseDispatch(char),
    #[error("reader macro `#'` requires a symbol suffix but found {0} instead")]
    VarDispatchRequiresSymbol(Value),
    #[error("internal error: {0}")]
    Internal(&'static str),
}

#[derive(Debug, Clone)]
/// A `ReadError` wraps a `ReaderError` with information
/// contextualizing the source of the error in the input data.
pub struct ReadError(ReaderError, usize);

impl ReadError {
    pub fn context<'a>(&self, input: &'a str) -> &'a str {
        &input[self.1..]
    }
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} starting at index {} in the input", self.0, self.1)
    }
}

impl std::error::Error for ReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

#[derive(Copy, Clone, Debug)]
enum ParseState {
    Reading,
    Exiting,
}

impl Default for ParseState {
    fn default() -> Self {
        Self::Reading
    }
}

#[derive(Debug)]
enum Span {
    // captures an atomic value
    Simple(Range),
    // captures a compound value with an enclosing span
    // and some number of enclosed spans
    Compound(Range, Vec<Span>),
    // a span of whitespace `   ,,,,  ,, `
    Whitespace(Range),
    // span of a comment, e.g. `;; some comment`
    Comment(Range),
}

#[derive(Default, Debug)]
struct Reader<'a> {
    input: &'a str,
    spans: Vec<Span>,
    values: Vec<Value>,
    line_count: usize,
    // beginning of the current focus in `input`
    cursor: usize,
    parse_state: ParseState,
}

impl<'a> Reader<'a> {
    fn new() -> Self {
        Self::default()
    }

    fn read_whitespace(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        let (start, ch) = stream.next().expect("from peek");
        let mut end = None;
        self.cursor = start;

        if is_newline(ch) {
            self.line_count += 1;
        }

        while let Some((index, ch)) = stream.peek() {
            if is_whitespace(*ch) {
                let (_, ch) = stream.next().expect("from peek");
                if is_newline(ch) {
                    self.line_count += 1;
                }
            } else {
                end = Some(*index);
                break;
            }
        }

        let span = if let Some(end) = end {
            Range::Slice(start, end)
        } else {
            Range::ToEnd(start)
        };
        self.spans.push(Span::Whitespace(span));
        Ok(())
    }

    fn read_comment(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        let (start, _) = stream.next().expect("from peek");
        let mut end = None;
        self.cursor = start;

        while let Some((_, ch)) = stream.next() {
            if is_newline(ch) {
                self.line_count += 1;
                break;
            }
        }
        if let Some((index, _)) = stream.peek() {
            end = Some(*index);
        }
        let span = if let Some(end) = end {
            Range::Slice(start, end)
        } else {
            Range::ToEnd(start)
        };
        self.spans.push(Span::Comment(span));
        Ok(())
    }

    fn read_number(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        let (start, _) = stream.next().expect("from peek");
        let mut end = None;
        self.cursor = start;

        while let Some((index, ch)) = stream.peek() {
            end = Some(*index);
            let ch = *ch;
            // NOTE: want to scan until whitespace, comment or structural ch
            if is_token(ch) && !is_structural(ch) {
                stream.next();
                continue;
            }
            break;
        }
        if stream.peek().is_none() {
            end = Some(self.input.len());
        }
        if let Some(end) = end {
            let source = &self.input[start..end];
            let n = source.parse()?;
            let span = Range::Slice(start, end);
            self.spans.push(Span::Simple(span));
            self.values.push(Value::Number(n));
            Ok(())
        } else {
            Err(ReaderError::ExpectedMoreInput)
        }
    }

    fn read_symbolic(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        let (start, _) = stream.next().expect("from peek");
        let mut end = None;
        self.cursor = start;

        while let Some((index, ch)) = stream.peek() {
            end = Some(*index);
            if is_symbolic(*ch) {
                stream.next();
                continue;
            }
            break;
        }
        if stream.peek().is_none() {
            end = Some(self.input.len());
        }
        if let Some(end) = end {
            let source = &self.input[start..end];
            let value = parse_symbolic(source)?;
            self.values.push(value);
            let span = Range::Slice(start, end);
            self.spans.push(Span::Simple(span));
            Ok(())
        } else {
            Err(ReaderError::ExpectedMoreInput)
        }
    }

    fn read_string(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        let (start, _) = stream.next().expect("from peek");
        self.cursor = start;

        let end = find_string_close(stream)?;
        // start at character after first '"'
        let source = &self.input[start + 1..end];
        let escaped_string = apply_string_escapes(source);
        let span = Range::Slice(start, end);
        self.spans.push(Span::Simple(span));
        let value = Value::String(escaped_string);
        self.values.push(value);
        Ok(())
    }

    fn read_number_and_negate(
        &mut self,
        start: usize,
        stream: &mut Stream,
    ) -> Result<(), ReaderError> {
        self.cursor = start;
        self.read_number(stream).map_err(|err| {
            self.cursor = start;
            err
        })?;
        let number = self.values.last_mut().expect("did read number");
        let span = self.spans.last_mut().expect("did range number");
        match (number, span) {
            (Value::Number(n), Span::Simple(range)) => {
                match range {
                    Range::Slice(number_start, _) => {
                        *number_start = start;
                    }
                    Range::ToEnd(number_start) => {
                        *number_start = start;
                    }
                }

                let neg_n = n
                    .checked_neg()
                    .ok_or_else(|| ReaderError::CouldNotNegateNumber(*n))?;
                *n = neg_n;
            }
            _ => unreachable!("should have read number with simple span"),
        }
        Ok(())
    }

    fn read_symbolic_and_prepend_dash(
        &mut self,
        start: usize,
        stream: &mut Stream,
    ) -> Result<(), ReaderError> {
        self.cursor = start;
        self.read_symbolic(stream).map_err(|err| {
            self.cursor = start;
            err
        })?;
        let symbol = self.values.last_mut().expect("did read symbol");
        let span = self.spans.last_mut().expect("did range symbol");
        match (symbol, span) {
            (Value::Symbol(identifier, None), Span::Simple(range)) if identifier == "/" => {
                match range {
                    Range::Slice(symbol_start, _) => {
                        *symbol_start = start;
                    }
                    Range::ToEnd(symbol_start) => {
                        *symbol_start = start;
                    }
                }
                self.cursor = start;
                return Err(ReaderError::MissingIdentifier);
            }
            (Value::Symbol(identifier, ns_opt), Span::Simple(range)) => {
                match range {
                    Range::Slice(symbol_start, _) => {
                        *symbol_start = start;
                    }
                    Range::ToEnd(symbol_start) => {
                        *symbol_start = start;
                    }
                }

                if let Some(ns) = ns_opt {
                    ns.insert(0, '-');
                } else {
                    identifier.insert(0, '-');
                }
            }
            _ => unreachable!("should have read symbol with simple span"),
        }
        Ok(())
    }

    fn disambiguate_dash(&mut self, start: usize, stream: &mut Stream) -> Result<(), ReaderError> {
        stream.next().expect("from peek");

        if let Some((end, next)) = stream.peek() {
            match *next {
                ch if is_numeric(ch) => self.read_number_and_negate(start, stream)?,
                ch if is_symbolic(ch) => self.read_symbolic_and_prepend_dash(start, stream)?,
                _ => {
                    self.cursor = start;
                    let value = Value::Symbol('-'.to_string(), None);
                    self.values.push(value);
                    let span = Range::Slice(start, *end);
                    self.spans.push(Span::Simple(span));
                }
            }
        } else {
            self.cursor = start;
            let value = Value::Symbol('-'.to_string(), None);
            self.values.push(value);
            let span = Range::ToEnd(start);
            self.spans.push(Span::Simple(span));
        }
        Ok(())
    }

    fn read_atom(
        &mut self,
        first_char: char,
        start: usize,
        stream: &mut Stream,
    ) -> Result<(), ReaderError> {
        match first_char {
            ch if ch == '-' => self.disambiguate_dash(start, stream),
            ch if is_numeric(ch) => self.read_number(stream),
            ch if is_symbolic(ch) => self.read_symbolic(stream),
            ch => {
                self.cursor = start;
                Err(ReaderError::UnexpectedInput(ch))
            }
        }
    }

    fn read_collection<C>(
        &mut self,
        terminal: char,
        stream: &mut Stream,
        collector: C,
    ) -> Result<(), ReaderError>
    where
        C: Fn(Vec<Value>) -> Result<Value, ReaderError>,
    {
        let (start, _) = stream.next().expect("from peek");
        self.cursor = start;
        let values_index = self.values.len();
        let spans_index = self.spans.len();
        let previous_state = self.parse_state;
        self.parse_state = ParseState::Reading;
        self.read_from_stream(stream)?;
        self.parse_state = previous_state;

        let collection = collector(self.values.drain(values_index..).collect())?;
        self.values.push(collection);

        let (end, ch) = stream.next().ok_or_else(|| {
            self.cursor = start;
            ReaderError::UnbalancedCollection(terminal)
        })?;
        if ch != terminal {
            self.cursor = start;
            return Err(ReaderError::UnbalancedCollection(terminal));
        }
        let range = Range::Slice(start, end);
        let intervening_spans = self.spans.drain(spans_index..).collect();
        self.spans.push(Span::Compound(range, intervening_spans));
        Ok(())
    }

    fn read_dispatch(&mut self, start: usize, stream: &mut Stream) -> Result<(), ReaderError> {
        self.cursor = start;
        let (_, next_ch) = stream.peek().ok_or(ReaderError::ExpectedMoreInput)?;
        match *next_ch {
            '{' => {
                self.read_collection('}', stream, |elems| Ok(set_with_values(elems)))
                    .map_err(|err| {
                        self.cursor = start;
                        err
                    })?;
                let span = self.spans.last_mut().expect("just read set");
                match span {
                    Span::Compound(enclosing, _) => match enclosing {
                        Range::Slice(set_start, _) => {
                            *set_start = start;
                        }
                        _ => unreachable!("reading collection yields slice range"),
                    },
                    _ => unreachable!("reading collection yields compound span"),
                }
                Ok(())
            }
            '\'' => {
                stream.next().expect("from peek");
                self.read_exactly_one_form(start, stream).map_err(|err| {
                    self.cursor = start;
                    err
                })?;
                let symbol = self.values.pop().expect("just read symbol");
                let span = self.spans.pop().expect("just ranged symbol");
                match symbol {
                    symbol @ Value::Symbol(..) => {
                        let expansion = list_with_values(
                            [Value::Symbol("var".to_string(), None), symbol]
                                .iter()
                                .cloned(),
                        );
                        self.values.push(expansion);

                        let dispatch_span = match span {
                            Span::Simple(range) => match range {
                                Range::Slice(_, end) => Range::Slice(start, end),
                                Range::ToEnd(_) => Range::ToEnd(start),
                            },
                            _ => unreachable!("reading symbol yields simple span"),
                        };
                        self.spans.push(Span::Simple(dispatch_span));
                        Ok(())
                    }
                    other => {
                        self.cursor = start;
                        Err(ReaderError::VarDispatchRequiresSymbol(other))
                    }
                }
            }
            '_' => {
                stream.next().expect("from peek");
                self.read_exactly_one_form(start, stream).map_err(|err| {
                    self.cursor = start;
                    err
                })?;

                self.values.pop().expect("just read one form");
                self.spans.pop().expect("just ranged one form");
                Ok(())
            }
            ch => Err(ReaderError::CouldNotParseDispatch(ch)),
        }
    }

    fn read_exactly_one_form(
        &mut self,
        start: usize,
        stream: &mut Stream,
    ) -> Result<(), ReaderError> {
        self.cursor = start;
        let values_count = self.values.len();
        let previous_state = self.parse_state;
        self.parse_state = ParseState::Exiting;
        self.read_from_stream(stream).map_err(|err| {
            self.cursor = start;
            err
        })?;
        self.parse_state = previous_state;

        match self.values.len() {
            len if len == values_count => Err(ReaderError::ExpectedMoreInput),
            len if len > values_count + 1 => Err(ReaderError::Internal(
                "read too many forms during reader macro",
            )),
            len if len < values_count => Err(ReaderError::Internal(
                "unexpectedly dropped forms during reader macro",
            )),
            _ => Ok(()),
        }
    }

    fn read_macro(
        &mut self,
        identifier: &str,
        start: usize,
        stream: &mut Stream,
    ) -> Result<(), ReaderError> {
        self.read_exactly_one_form(start, stream).map_err(|err| {
            self.cursor = start;
            err
        })?;
        let form = self.values.pop().expect("just read form");
        let expansion = list_with_values(
            [Value::Symbol(identifier.to_string(), None), form]
                .iter()
                .cloned(),
        );
        self.values.push(expansion);

        let span = self.spans.pop().expect("just ranged form");
        let span = match span {
            Span::Simple(range) => {
                let range = match range {
                    Range::Slice(_, end) => Range::Slice(start, end),
                    Range::ToEnd(_) => Range::ToEnd(start),
                };
                Span::Simple(range)
            }
            Span::Compound(range, enclosed) => {
                let range = match range {
                    Range::Slice(_, end) => Range::Slice(start, end),
                    Range::ToEnd(_) => Range::ToEnd(start),
                };
                Span::Compound(range, enclosed)
            }
            _ => unreachable!("read some form"),
        };
        self.spans.push(span);
        Ok(())
    }

    fn read_form(
        &mut self,
        next_char: char,
        next_index: usize,
        stream: &mut Stream,
    ) -> Result<(), ReaderError> {
        match next_char {
            '(' => {
                self.read_collection(')', stream, |elems| Ok(list_with_values(elems)))?;
            }
            ')' => {
                self.parse_state = ParseState::Exiting;
            }
            '[' => {
                self.read_collection(']', stream, |elems| Ok(vector_with_values(elems)))?;
            }
            ']' => {
                self.parse_state = ParseState::Exiting;
            }
            '{' => {
                self.read_collection('}', stream, |elems| {
                    if elems.len() % 2 != 0 {
                        Err(ReaderError::MapLiteralWithUnpairedElements)
                    } else {
                        Ok(map_with_values(elems.into_iter().tuples()))
                    }
                })?;
            }
            '}' => {
                self.parse_state = ParseState::Exiting;
            }
            '#' => {
                stream.next().expect("from peek");
                self.read_dispatch(next_index, stream)?;
            }
            '@' => {
                stream.next().expect("from peek");
                self.read_macro("deref", next_index, stream)?;
            }
            '\'' => {
                stream.next().expect("from peek");
                self.read_macro("quote", next_index, stream)?;
            }
            '`' => {
                stream.next().expect("from peek");
                self.read_macro("quasiquote", next_index, stream)?;
            }
            '~' => {
                stream.next().expect("from peek");
                let (_, next) = stream.peek().ok_or(ReaderError::ExpectedMoreInput)?;
                let identifier = if *next == '@' {
                    stream.next().expect("from peek");
                    "splice-unquote"
                } else {
                    "unquote"
                };
                self.read_macro(identifier, next_index, stream)?;
            }
            '"' => self.read_string(stream)?,
            ch if is_token(ch) => self.read_atom(ch, next_index, stream)?,
            _ => unreachable!(),
        }
        Ok(())
    }

    fn read_from_stream(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        while let Some((index, ch)) = stream.peek() {
            let ch = *ch;
            if is_whitespace(ch) {
                self.read_whitespace(stream)?;
                continue;
            }
            if is_comment(ch) {
                self.read_comment(stream)?;
                continue;
            }
            self.read_form(ch, *index, stream)?;
            if matches!(self.parse_state, ParseState::Exiting) {
                break;
            }
        }
        Ok(())
    }

    fn read(&mut self, input: &'a str) -> Result<(), ReaderError> {
        self.input = input;
        let mut stream = input.char_indices().peekable();
        self.read_from_stream(&mut stream)?;
        if let Some((_, ch)) = stream.next() {
            return Err(ReaderError::UnexpectedInput(ch));
        }
        Ok(())
    }
}

pub fn read(input: &str) -> Result<Vec<Value>, ReadError> {
    let mut reader = Reader::new();
    match reader.read(input) {
        Ok(_) => Ok(reader.values),
        Err(err) => Err(ReadError(err, reader.cursor)),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        list_with_values, map_with_values, read, set_with_values, vector_with_values, ReadError,
        ReaderError, Value::*,
    };
    use itertools::Itertools;

    #[test]
    fn test_read_error() {
        use std::ops::Fn;

        let cases: Vec<(&str, Box<dyn Fn(&ReaderError) -> bool>, usize)> = vec![
            (
                "234897abc",
                Box::new(|err| matches!(err, ReaderError::CouldNotParseNumber(_))),
                0,
            ),
            (
                "( 234897abc",
                Box::new(|err| matches!(err, ReaderError::CouldNotParseNumber(_))),
                2,
            ),
            (
                "-234897abc",
                Box::new(|err| matches!(err, ReaderError::CouldNotParseNumber(_))),
                0,
            ),
            (
                "-/",
                Box::new(|err| matches!(err, ReaderError::MissingIdentifier)),
                0,
            ),
            (
                "123 abc -/",
                Box::new(|err| matches!(err, ReaderError::MissingIdentifier)),
                8,
            ),
            (
                "/foo",
                Box::new(|err| matches!(err, ReaderError::MissingNamespace)),
                0,
            ),
            (
                "/-",
                Box::new(|err| matches!(err, ReaderError::MissingNamespace)),
                0,
            ),
            (
                "//",
                Box::new(|err| matches!(err, ReaderError::MissingNamespace)),
                0,
            ),
            (
                "///",
                Box::new(|err| matches!(err, ReaderError::MissingNamespace)),
                0,
            ),
            (
                "\"some string",
                Box::new(|err| matches!(err, ReaderError::UnbalancedString)),
                0,
            ),
            (
                "(  \"some string)",
                Box::new(|err| matches!(err, ReaderError::UnbalancedString)),
                3,
            ),
            (
                "\"some string \\\"nested string\\\"",
                Box::new(|err| matches!(err, ReaderError::UnbalancedString)),
                0,
            ),
            (
                "\"some string \\\"nested string does not end",
                Box::new(|err| matches!(err, ReaderError::UnbalancedString)),
                0,
            ),
            (
                ",,,,,\"some string",
                Box::new(|err| matches!(err, ReaderError::UnbalancedString)),
                5,
            ),
            (
                ",,,,,\n\n\"some string",
                Box::new(|err| matches!(err, ReaderError::UnbalancedString)),
                7,
            ),
            (
                "foo/:",
                Box::new(|err| matches!(err, ReaderError::InvalidIdentifier)),
                0,
            ),
            (
                "foo/:bar",
                Box::new(|err| matches!(err, ReaderError::InvalidIdentifier)),
                0,
            ),
            (
                ":",
                Box::new(|err| matches!(err, ReaderError::InvalidIdentifier)),
                0,
            ),
            (
                ":/",
                Box::new(|err| matches!(err, ReaderError::InvalidIdentifier)),
                0,
            ),
            (
                ":/:",
                Box::new(|err| matches!(err, ReaderError::MissingNamespace)),
                0,
            ),
            (
                ":/foo",
                Box::new(|err| matches!(err, ReaderError::MissingNamespace)),
                0,
            ),
            (
                "::",
                Box::new(|err| matches!(err, ReaderError::InvalidIdentifier)),
                0,
            ),
            (
                "::/",
                Box::new(|err| matches!(err, ReaderError::InvalidIdentifier)),
                0,
            ),
            (
                "::foo/bar",
                Box::new(|err| matches!(err, ReaderError::InvalidNamespace)),
                0,
            ),
            (
                ";; some comment \n:",
                Box::new(|err| matches!(err, ReaderError::InvalidIdentifier)),
                17,
            ),
            (
                "(",
                Box::new(|err| matches!(err, ReaderError::UnbalancedCollection(')'))),
                0,
            ),
            (
                "(1 2",
                Box::new(|err| matches!(err, ReaderError::UnbalancedCollection(')'))),
                0,
            ),
            (
                "[1 2",
                Box::new(|err| matches!(err, ReaderError::UnbalancedCollection(']'))),
                0,
            ),
            (
                "{1 2",
                Box::new(|err| matches!(err, ReaderError::UnbalancedCollection('}'))),
                0,
            ),
            (
                "#{1 2",
                Box::new(|err| matches!(err, ReaderError::UnbalancedCollection('}'))),
                0,
            ),
            (
                "1 2 (1 2",
                Box::new(|err| matches!(err, ReaderError::UnbalancedCollection(')'))),
                4,
            ),
            (
                "[1 2 (1 2])",
                Box::new(|err| matches!(err, ReaderError::UnbalancedCollection(')'))),
                5,
            ),
            (
                "{1 3 4}",
                Box::new(|err| matches!(err, ReaderError::MapLiteralWithUnpairedElements)),
                5,
            ),
            (
                "{1 3 [1 2}",
                Box::new(|err| matches!(err, ReaderError::UnbalancedCollection(']'))),
                5,
            ),
            (
                "(((((((([))))))))",
                Box::new(|err| matches!(err, ReaderError::UnbalancedCollection(']'))),
                8,
            ),
            (
                "(a b c)\u{200B}",
                Box::new(|err| matches!(err, ReaderError::UnexpectedInput('\u{200B}'))),
                7,
            ),
            (
                "(a b c)  \u{200B}",
                Box::new(|err| matches!(err, ReaderError::UnexpectedInput('\u{200B}'))),
                9,
            ),
            (
                "#",
                Box::new(|err| matches!(err, ReaderError::ExpectedMoreInput)),
                0,
            ),
            (
                "#!some-form",
                Box::new(|err| matches!(err, ReaderError::CouldNotParseDispatch('!'))),
                0,
            ),
            (
                "#'(not-a-symbol)",
                Box::new(|err| matches!(err, ReaderError::VarDispatchRequiresSymbol(_))),
                0,
            ),
            (
                "@",
                Box::new(|err| matches!(err, ReaderError::ExpectedMoreInput)),
                0,
            ),
            (
                "1 2 @,,,,,,,",
                Box::new(|err| matches!(err, ReaderError::ExpectedMoreInput)),
                4,
            ),
            (
                "1 2 [3 4 @] 5",
                Box::new(|err| matches!(err, ReaderError::ExpectedMoreInput)),
                9,
            ),
            (
                "'",
                Box::new(|err| matches!(err, ReaderError::ExpectedMoreInput)),
                0,
            ),
            (
                "`",
                Box::new(|err| matches!(err, ReaderError::ExpectedMoreInput)),
                0,
            ),
            (
                "~",
                Box::new(|err| matches!(err, ReaderError::ExpectedMoreInput)),
                0,
            ),
        ];
        for (case, err_pattern, expected_index) in cases {
            match read(case) {
                Ok(value) => {
                    println!(
                        "read value(s) {:?} successfully when expected error on this input `{}`",
                        value, case
                    );
                    assert!(false);
                }
                Err(ReadError(err, index)) => {
                    if !err_pattern(&err) {
                        println!("did not get back the expected error type when reading `{}`, instead got {}", case, err);
                        assert!(false);
                    }
                    if index != expected_index {
                        println!("did not locate the correct error position when reading `{}`: expected {} but got {}", case, expected_index, index);
                        assert!(false);
                    }
                }
            }
        }
    }

    #[test]
    fn test_basic_read() {
        let cases = vec![
            ("nil", vec![Nil], "nil"),
            ("true", vec![Bool(true)], "true"),
            ("false", vec![Bool(false)], "false"),
            (" false", vec![Bool(false)], "false"),
            ("false ", vec![Bool(false)], "false"),
            ("1337", vec![Number(1337)], "1337"),
            ("-1337", vec![Number(-1337)], "-1337"),
            ("-1337  ", vec![Number(-1337)], "-1337"),
            ("  -1337", vec![Number(-1337)], "-1337"),
            ("  -1337  ", vec![Number(-1337)], "-1337"),
            ("1337  ", vec![Number(1337)], "1337"),
            ("    1337  ", vec![Number(1337)], "1337"),
            (" ,  1337, ", vec![Number(1337)], "1337"),
            (" ", vec![], ""),
            (",", vec![], ""),
            ("  ", vec![], ""),
            (",,,", vec![], ""),
            ("  ", vec![], ""),
            (" , ", vec![], ""),
            (
                "true, , , false",
                vec![Bool(true), Bool(false)],
                "true false",
            ),
            ("", vec![], ""),
            ("baz", vec![Symbol("baz".into(), None)], "baz"),
            ("baz#", vec![Symbol("baz#".into(), None)], "baz#"),
            ("baz$$", vec![Symbol("baz$$".into(), None)], "baz$$"),
            ("$baz", vec![Symbol("$baz".into(), None)], "$baz"),
            ("$baz$$", vec![Symbol("$baz$$".into(), None)], "$baz$$"),
            (
                "foo/$baz$$",
                vec![Symbol("$baz$$".into(), Some("foo".into()))],
                "foo/$baz$$",
            ),
            (
                "$foo/$baz$$",
                vec![Symbol("$baz$$".into(), Some("$foo".into()))],
                "$foo/$baz$$",
            ),
            (
                "$foo$/$baz$$",
                vec![Symbol("$baz$$".into(), Some("$foo$".into()))],
                "$foo$/$baz$$",
            ),
            ("-", vec![Symbol("-".into(), None)], "-"),
            ("-=", vec![Symbol("-=".into(), None)], "-="),
            ("--", vec![Symbol("--".into(), None)], "--"),
            ("-baz", vec![Symbol("-baz".into(), None)], "-baz"),
            ("--baz", vec![Symbol("--baz".into(), None)], "--baz"),
            ("-$baz", vec![Symbol("-$baz".into(), None)], "-$baz"),
            (
                "--/baz",
                vec![Symbol("baz".into(), Some("--".to_string()))],
                "--/baz",
            ),
            (
                "-=/baz",
                vec![Symbol("baz".into(), Some("-=".to_string()))],
                "-=/baz",
            ),
            (
                "- baz",
                vec![Symbol("-".into(), None), Symbol("baz".into(), None)],
                "- baz",
            ),
            ("->>", vec![Symbol("->>".into(), None)], "->>"),
            ("/", vec![Symbol("/".into(), None)], "/"),
            (
                "foo//",
                vec![Symbol("/".into(), Some("foo".into()))],
                "foo//",
            ),
            (
                "bar/-",
                vec![Symbol("-".into(), Some("bar".into()))],
                "bar/-",
            ),
            (
                "bar/- 1",
                vec![Symbol("-".into(), Some("bar".into())), Number(1)],
                "bar/- 1",
            ),
            (
                "bar/- -1",
                vec![Symbol("-".into(), Some("bar".into())), Number(-1)],
                "bar/- -1",
            ),
            (
                "foo/baz",
                vec![Symbol("baz".into(), Some("foo".into()))],
                "foo/baz",
            ),
            (
                "foo/baz bar true",
                vec![
                    Symbol("baz".into(), Some("foo".into())),
                    Symbol("bar".into(), None),
                    Bool(true),
                ],
                "foo/baz bar true",
            ),
            ("\"\"", vec![String("".into())], "\"\""),
            (r#""\"""#, vec![String("\"".into())], "\"\\\"\""),
            ("\"hi\"", vec![String("hi".into())], "\"hi\""),
            ("\" \\\\ \"", vec![String(" \\ ".into())], "\" \\\\ \""),
            ("\"&\"", vec![String("&".into())], "\"&\""),
            ("\"'\"", vec![String("'".into())], "\"'\""),
            ("\"(\"", vec![String("(".into())], "\"(\""),
            ("\")\"", vec![String(")".into())], "\")\""),
            ("\"*\"", vec![String("*".into())], "\"*\""),
            ("\"+\"", vec![String("+".into())], "\"+\""),
            ("\",\"", vec![String(",".into())], "\",\""),
            ("\"-\"", vec![String("-".into())], "\"-\""),
            ("\"/\"", vec![String("/".into())], "\"/\""),
            ("\":\"", vec![String(":".into())], "\":\""),
            ("\";\"", vec![String(";".into())], "\";\""),
            ("\"<\"", vec![String("<".into())], "\"<\""),
            ("\"=\"", vec![String("=".into())], "\"=\""),
            ("\">\"", vec![String(">".into())], "\">\""),
            ("\"?\"", vec![String("?".into())], "\"?\""),
            ("\"@\"", vec![String("@".into())], "\"@\""),
            ("\"[\"", vec![String("[".into())], "\"[\""),
            ("\"]\"", vec![String("]".into())], "\"]\""),
            ("\"^\"", vec![String("^".into())], "\"^\""),
            ("\"_\"", vec![String("_".into())], "\"_\""),
            ("\"`\"", vec![String("`".into())], "\"`\""),
            ("\"{\"", vec![String("{".into())], "\"{\""),
            ("\"}\"", vec![String("}".into())], "\"}\""),
            ("\"~\"", vec![String("~".into())], "\"~\""),
            ("\"!\"", vec![String("!".into())], "\"!\""),
            ("\"\\n\"", vec![String("\n".into())], "\"\\n\""),
            ("\"#\"", vec![String("#".into())], "\"#\""),
            ("\"$\"", vec![String("$".into())], "\"$\""),
            ("\"%\"", vec![String("%".into())], "\"%\""),
            ("\".\"", vec![String(".".into())], "\".\""),
            ("\"|\"", vec![String("|".into())], "\"|\""),
            (
                "\"123foo\" true",
                vec![String("123foo".into()), Bool(true)],
                "\"123foo\" true",
            ),
            (
                "\"hi (test with parens)\"",
                vec![String("hi (test with parens)".into())],
                "\"hi (test with parens)\"",
            ),
            (r#""abc""#, vec![String("abc".into())], "\"abc\""),
            (":foobar", vec![Keyword("foobar".into(), None)], ":foobar"),
            (
                ":net/hi",
                vec![Keyword("hi".into(), Some("net".into()))],
                ":net/hi",
            ),
            (
                "::bar",
                vec![Keyword("bar".into(), Some(":".into()))],
                // NOTE: this is to match `Value` specific behavior
                // the `::` form is interpreted as an auto-resolving
                // namespace inside the interpreter
                "::/bar",
            ),
            (
                ":a0987234",
                vec![Keyword("a0987234".into(), None)],
                ":a0987234",
            ),
            ("; sdlkfjsldfjsldjflsdjf", vec![], ""),
            (
                "foo/bar true ;; some comment",
                vec![Symbol("bar".into(), Some("foo".into())), Bool(true)],
                "foo/bar true",
            ),
            ("baz ;; comment \n", vec![Symbol("baz".into(), None)], "baz"),
            (
                "baz ;; comment \n12",
                vec![Symbol("baz".into(), None), Number(12)],
                "baz 12",
            ),
            (
                "baz ;; comment \n   12",
                vec![Symbol("baz".into(), None), Number(12)],
                "baz 12",
            ),
            (
                "baz ;; comment \n   12 ;;; another comment!",
                vec![Symbol("baz".into(), None), Number(12)],
                "baz 12",
            ),
            ("1;!", vec![Number(1)], "1"),
            ("1;\"", vec![Number(1)], "1"),
            ("1;#", vec![Number(1)], "1"),
            ("1;$", vec![Number(1)], "1"),
            ("1;%", vec![Number(1)], "1"),
            ("1;'", vec![Number(1)], "1"),
            ("1;\\", vec![Number(1)], "1"),
            ("1;\\\\", vec![Number(1)], "1"),
            ("1;\\\\\\", vec![Number(1)], "1"),
            ("1; &()*+,-./:;<=>?@[]^_{|}~", vec![Number(1)], "1"),
            ("()", vec![list_with_values(vec![])], "()"),
            (
                "(a b c)",
                vec![list_with_values(vec![
                    Symbol("a".into(), None),
                    Symbol("b".into(), None),
                    Symbol("c".into(), None),
                ])],
                "(a b c)",
            ),
            (
                "(a b, c,,,,),,",
                vec![list_with_values(vec![
                    Symbol("a".into(), None),
                    Symbol("b".into(), None),
                    Symbol("c".into(), None),
                ])],
                "(a b c)",
            ),
            (
                "(12 :foo/bar \"extra\")",
                vec![list_with_values(vec![
                    Number(12),
                    Keyword("bar".into(), Some("foo".into())),
                    String("extra".into()),
                ])],
                "(12 :foo/bar \"extra\")",
            ),
            ("[]", vec![vector_with_values(vec![])], "[]"),
            (
                "[a b c]",
                vec![vector_with_values(vec![
                    Symbol("a".into(), None),
                    Symbol("b".into(), None),
                    Symbol("c".into(), None),
                ])],
                "[a b c]",
            ),
            (
                "[12 :foo/bar \"extra\"]",
                vec![vector_with_values(vec![
                    Number(12),
                    Keyword("bar".into(), Some("foo".into())),
                    String("extra".into()),
                ])],
                "[12 :foo/bar \"extra\"]",
            ),
            ("{}", vec![map_with_values(vec![])], "{}"),
            (
                "{a b c d}",
                vec![map_with_values(vec![
                    (Symbol("a".into(), None), Symbol("b".into(), None)),
                    (Symbol("c".into(), None), Symbol("d".into(), None)),
                ])],
                "",
            ),
            (
                "{12 13 :foo/bar \"extra\"}",
                vec![map_with_values(vec![
                    (Number(12), Number(13)),
                    (
                        Keyword("bar".into(), Some("foo".into())),
                        String("extra".into()),
                    ),
                ])],
                "",
            ),
            (
                "() []",
                vec![list_with_values(vec![]), vector_with_values(vec![])],
                "() []",
            ),
            (
                "()\n[]",
                vec![list_with_values(vec![]), vector_with_values(vec![])],
                "() []",
            ),
            (
                "(())",
                vec![list_with_values(vec![list_with_values(vec![])])],
                "(())",
            ),
            (
                "(([]))",
                vec![list_with_values(vec![list_with_values(vec![
                    vector_with_values(vec![]),
                ])])],
                "(([]))",
            ),
            (
                "(([]))()",
                vec![
                    list_with_values(vec![list_with_values(vec![vector_with_values(vec![])])]),
                    list_with_values(vec![]),
                ],
                "(([])) ()",
            ),
            (
                "(12 (true [34 false]))(), (7)",
                vec![
                    list_with_values(vec![
                        Number(12),
                        list_with_values(vec![
                            Bool(true),
                            vector_with_values(vec![Number(34), Bool(false)]),
                        ]),
                    ]),
                    list_with_values(vec![]),
                    list_with_values(vec![Number(7)]),
                ],
                "(12 (true [34 false])) () (7)",
            ),
            (
                "  [ +   1   [+   2 3   ]   ]  ",
                vec![vector_with_values(vec![
                    Symbol("+".to_string(), None),
                    Number(1),
                    vector_with_values(vec![Symbol("+".to_string(), None), Number(2), Number(3)]),
                ])],
                "[+ 1 [+ 2 3]]",
            ),
            ("#{}", vec![set_with_values(vec![])], "#{}"),
            ("#{1}", vec![set_with_values(vec![Number(1)])], "#{1}"),
            ("#{   1}", vec![set_with_values(vec![Number(1)])], "#{1}"),
            ("#{   1  }", vec![set_with_values(vec![Number(1)])], "#{1}"),
            (
                "#{   \"hi\"  }",
                vec![set_with_values(vec![String("hi".to_string())])],
                "#{\"hi\"}",
            ),
            (
                "#{1 2 3}",
                vec![set_with_values(vec![Number(1), Number(2), Number(3)])],
                "",
            ),
            (
                "#{(1 3) :foo}",
                vec![set_with_values(vec![
                    list_with_values(vec![Number(1), Number(3)]),
                    Keyword("foo".into(), None),
                ])],
                "",
            ),
            (
                "+ ! =",
                vec![
                    Symbol("+".into(), None),
                    Symbol("!".into(), None),
                    Symbol("=".into(), None),
                ],
                "+ ! =",
            ),
            (
                "(defn foo [a] (+ a 1))",
                vec![list_with_values(vec![
                    Symbol("defn".into(), None),
                    Symbol("foo".into(), None),
                    vector_with_values(vec![Symbol("a".into(), None)]),
                    list_with_values(vec![
                        Symbol("+".into(), None),
                        Symbol("a".into(), None),
                        Number(1),
                    ]),
                ])],
                "(defn foo [a] (+ a 1))",
            ),
            (
                "(defn foo\n [a]\n (+ a 1))",
                vec![list_with_values(vec![
                    Symbol("defn".into(), None),
                    Symbol("foo".into(), None),
                    vector_with_values(vec![Symbol("a".into(), None)]),
                    list_with_values(vec![
                        Symbol("+".into(), None),
                        Symbol("a".into(), None),
                        Number(1),
                    ]),
                ])],
                "(defn foo [a] (+ a 1))",
            ),
            (
                r#"
                (defn foo  ; some comment
                  [a]      ; another comment
                  (+ a 1)) ; one final comment
                "#,
                vec![list_with_values(vec![
                    Symbol("defn".into(), None),
                    Symbol("foo".into(), None),
                    vector_with_values(vec![Symbol("a".into(), None)]),
                    list_with_values(vec![
                        Symbol("+".into(), None),
                        Symbol("a".into(), None),
                        Number(1),
                    ]),
                ])],
                "(defn foo [a] (+ a 1))",
            ),
            (
                "@a",
                vec![list_with_values(vec![
                    Symbol("deref".into(), None),
                    Symbol("a".into(), None),
                ])],
                "(deref a)",
            ),
            (
                "@          a",
                vec![list_with_values(vec![
                    Symbol("deref".into(), None),
                    Symbol("a".into(), None),
                ])],
                "(deref a)",
            ),
            (
                "@,,,,,a",
                vec![list_with_values(vec![
                    Symbol("deref".into(), None),
                    Symbol("a".into(), None),
                ])],
                "(deref a)",
            ),
            (
                "@ ,, ,   a",
                vec![list_with_values(vec![
                    Symbol("deref".into(), None),
                    Symbol("a".into(), None),
                ])],
                "(deref a)",
            ),
            (
                "'1",
                vec![list_with_values(vec![
                    Symbol("quote".into(), None),
                    Number(1),
                ])],
                "(quote 1)",
            ),
            (
                "'(1 2 3)",
                vec![list_with_values(vec![
                    Symbol("quote".into(), None),
                    list_with_values(vec![Number(1), Number(2), Number(3)]),
                ])],
                "(quote (1 2 3))",
            ),
            (
                "`1",
                vec![list_with_values(vec![
                    Symbol("quasiquote".into(), None),
                    Number(1),
                ])],
                "(quasiquote 1)",
            ),
            (
                "`(1 2 3)",
                vec![list_with_values(vec![
                    Symbol("quasiquote".into(), None),
                    list_with_values(vec![Number(1), Number(2), Number(3)]),
                ])],
                "(quasiquote (1 2 3))",
            ),
            (
                "~1",
                vec![list_with_values(vec![
                    Symbol("unquote".into(), None),
                    Number(1),
                ])],
                "(unquote 1)",
            ),
            (
                "~(1 2 3)",
                vec![list_with_values(vec![
                    Symbol("unquote".into(), None),
                    list_with_values(vec![Number(1), Number(2), Number(3)]),
                ])],
                "(unquote (1 2 3))",
            ),
            (
                "`(1 ~a 3)",
                vec![list_with_values(vec![
                    Symbol("quasiquote".into(), None),
                    list_with_values(vec![
                        Number(1),
                        list_with_values(vec![
                            Symbol("unquote".into(), None),
                            Symbol("a".into(), None),
                        ]),
                        Number(3),
                    ]),
                ])],
                "(quasiquote (1 (unquote a) 3))",
            ),
            (
                "~@(1 2 3)",
                vec![list_with_values(vec![
                    Symbol("splice-unquote".into(), None),
                    list_with_values(vec![Number(1), Number(2), Number(3)]),
                ])],
                "(splice-unquote (1 2 3))",
            ),
            (
                "~@,,,,,(1 2 3)",
                vec![list_with_values(vec![
                    Symbol("splice-unquote".into(), None),
                    list_with_values(vec![Number(1), Number(2), Number(3)]),
                ])],
                "(splice-unquote (1 2 3))",
            ),
            (
                "~@,  ,,(1 2 3)",
                vec![list_with_values(vec![
                    Symbol("splice-unquote".into(), None),
                    list_with_values(vec![Number(1), Number(2), Number(3)]),
                ])],
                "(splice-unquote (1 2 3))",
            ),
            (
                "~@    (1 2 3)",
                vec![list_with_values(vec![
                    Symbol("splice-unquote".into(), None),
                    list_with_values(vec![Number(1), Number(2), Number(3)]),
                ])],
                "(splice-unquote (1 2 3))",
            ),
            ("1 #_(1 2 3) 3", vec![Number(1), Number(3)], "1 3"),
            (
                "1 (1 2 #_[1 2 3 :keyw]) 3",
                vec![
                    Number(1),
                    list_with_values([Number(1), Number(2)].iter().cloned()),
                    Number(3),
                ],
                "1 (1 2) 3",
            ),
            (
                "1 (1 2 #_[1 2 3 :keyw]) ,,,,,,, #_3",
                vec![
                    Number(1),
                    list_with_values([Number(1), Number(2)].iter().cloned()),
                ],
                "1 (1 2)",
            ),
            (
                "1 (1 2 #_[1 2 3 :keyw]) ,,,,,,, #_3        \n\n4",
                vec![
                    Number(1),
                    list_with_values([Number(1), Number(2)].iter().cloned()),
                    Number(4),
                ],
                "1 (1 2) 4",
            ),
        ];
        for (input, expected_read, expected_print) in cases {
            match read(input) {
                Ok(result) => {
                    assert_eq!(result, expected_read);
                    if expected_print != "" {
                        let print = result
                            .iter()
                            .map(|elem| elem.to_readable_string())
                            .join(" ");
                        assert_eq!(print, expected_print);
                    }
                }
                Err(err) => {
                    let context = err.context(input);
                    panic!(
                        "error reading `{}`: {} while reading `{}`",
                        input, err, context
                    );
                }
            }
        }
    }
}
