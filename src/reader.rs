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

fn is_token(input: char) -> bool {
    !is_whitespace(input) && !is_comment(input)
}

fn is_numeric(input: char) -> bool {
    char::is_numeric(input)
}

fn is_symbolic(input: char) -> bool {
    match input {
        '*' | '+' | '!' | '-' | '_' | '\'' | '?' | '<' | '>' | '=' | '/' | '&' | ':' | '$'
        | '#' => true,
        _ => char::is_alphanumeric(input),
    }
}

fn parse_identifier_and_optional_namespace(
    symbolic: &str,
) -> Result<(String, Option<String>), ReaderError> {
    if let Some((ns, identifier)) = symbolic.split_once('/') {
        if ns.is_empty() {
            if identifier.is_empty() {
                return Ok(("/".to_string(), None));
            }
            return Err(ReaderError::MissingNamespace);
        }
        Ok((identifier.to_string(), Some(ns.to_string())))
    } else {
        Ok((symbolic.to_string(), None))
    }
}

fn parse_symbolic_with_namespace(symbolic: &str) -> Result<Value, ReaderError> {
    if let Some(symbolic) = symbolic.strip_prefix(':') {
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
enum Span {
    ToEnd(usize),
    Slice(usize, usize),
}

#[derive(Debug, Error)]
pub enum ReaderError {
    #[error("error parsing number: {0}")]
    CouldNotParseNumber(#[from] ParseIntError),
    #[error("error negating number: {0}")]
    CouldNotNegateNumber(i64),
    #[error("unexpected input found after parsing all forms before EOF")]
    UnexpectedAdditionalInput,
    #[error("expected further input but found EOF")]
    ExpectedMoreInput,
    #[error("expected a namespace but none was given")]
    MissingNamespace,
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
    #[error("internal reader error: {0}")]
    Internal(String),
}

#[derive(Copy, Clone)]
enum ParseState {
    Reading,
    Exiting,
}

impl Default for ParseState {
    fn default() -> Self {
        Self::Reading
    }
}

#[derive(Default)]
struct Reader<'a> {
    input: &'a str,
    forms: Vec<Value>,
    whitespace_spans: Vec<Span>,
    comment_spans: Vec<Span>,
    line_count: usize,
    parse_state: ParseState,
}

impl<'a> Reader<'a> {
    fn new() -> Self {
        Self::default()
    }

    fn read_whitespace(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        let (start, ch) = stream.next().expect("from peek");
        let mut end = None;
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
            Span::Slice(start, end)
        } else {
            Span::ToEnd(start)
        };
        self.whitespace_spans.push(span);
        Ok(())
    }

    fn read_comment(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        let (start, _) = stream.next().expect("from peek");
        let mut end = None;

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
            Span::Slice(start, end)
        } else {
            Span::ToEnd(start)
        };
        self.comment_spans.push(span);
        Ok(())
    }

    fn read_number(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        let (start, _) = stream.next().expect("from peek");
        let mut end = None;
        while let Some((index, ch)) = stream.peek() {
            end = Some(*index);
            if is_numeric(*ch) {
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
            self.forms.push(Value::Number(n));
            Ok(())
        } else {
            Err(ReaderError::ExpectedMoreInput)
        }
    }

    fn read_symbolic(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        let (start, _) = stream.next().expect("from peek");
        let mut end = None;
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
            self.forms.push(value);
            Ok(())
        } else {
            Err(ReaderError::ExpectedMoreInput)
        }
    }

    fn read_string(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        let (start, _) = stream.next().expect("from peek");
        let end = find_string_close(stream)?;
        // start at character after first '"'
        let source = &self.input[start + 1..end];
        let escaped_string = apply_string_escapes(source);
        self.forms.push(Value::String(escaped_string));
        Ok(())
    }

    fn read_number_and_negate(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        self.read_number(stream)?;
        let number = self.forms.last_mut().expect("did read number");
        match number {
            Value::Number(n) => {
                let neg_n = n
                    .checked_neg()
                    .ok_or_else(|| ReaderError::CouldNotNegateNumber(*n))?;
                *n = neg_n;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn read_symbolic_and_prepend_dash(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        self.read_symbolic(stream)?;
        let symbol = self.forms.last_mut().expect("did read symbol");
        match symbol {
            Value::Symbol(identifier, ..) => {
                identifier.insert(0, '-');
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn disambiguate_dash(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        stream.next().expect("from peek");
        if let Some((_, next)) = stream.peek() {
            match *next {
                ch if is_numeric(ch) => self.read_number_and_negate(stream)?,
                ch if is_symbolic(ch) => self.read_symbolic_and_prepend_dash(stream)?,
                _ => {
                    let value = Value::Symbol('-'.to_string(), None);
                    self.forms.push(value);
                }
            }
        } else {
            let value = Value::Symbol('-'.to_string(), None);
            self.forms.push(value);
        }
        Ok(())
    }

    fn read_atom(&mut self, first_char: char, stream: &mut Stream) -> Result<(), ReaderError> {
        match first_char {
            ch if ch == '-' => self.disambiguate_dash(stream),
            ch if is_numeric(ch) => self.read_number(stream),
            ch if is_symbolic(ch) => self.read_symbolic(stream),
            ch => {
                panic!("unexpected character when reading atom: {}", ch);
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
        stream.next().expect("from peek");
        let start = self.forms.len();
        let previous_state = self.parse_state;
        self.parse_state = ParseState::Reading;
        self.read_from_stream(stream)?;
        self.parse_state = previous_state;
        let collection = collector(self.forms.drain(start..).collect())?;
        self.forms.push(collection);
        let (_, ch) = stream
            .next()
            .ok_or(ReaderError::UnbalancedCollection(terminal))?;
        if ch != terminal {
            return Err(ReaderError::UnbalancedCollection(terminal));
        }
        Ok(())
    }

    fn read_dispatch(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        let (_, next_ch) = stream.peek().ok_or(ReaderError::ExpectedMoreInput)?;
        match *next_ch {
            '{' => self.read_collection('}', stream, |elems| Ok(set_with_values(elems))),
            '\'' => {
                stream.next().expect("from peek");
                self.read_symbolic(stream)?;
                let symbol = self.forms.pop().expect("just read symbol");
                match symbol {
                    symbol @ Value::Symbol(..) => {
                        let expansion = list_with_values(
                            [Value::Symbol("var".to_string(), None), symbol]
                                .iter()
                                .cloned(),
                        );
                        self.forms.push(expansion);
                        Ok(())
                    }
                    other => Err(ReaderError::VarDispatchRequiresSymbol(other)),
                }
            }
            ch => Err(ReaderError::CouldNotParseDispatch(ch)),
        }
    }

    fn read_macro(&mut self, identifier: &str, stream: &mut Stream) -> Result<(), ReaderError> {
        let forms_count = self.forms.len();
        let previous_state = self.parse_state;
        self.parse_state = ParseState::Exiting;
        self.read_from_stream(stream)?;
        self.parse_state = previous_state;

        match self.forms.len() {
            len if len == forms_count => {
                return Err(ReaderError::ExpectedMoreInput);
            }
            len if len > forms_count + 1 => {
                return Err(ReaderError::Internal(
                    "read too many forms during reader macro".to_string(),
                ));
            }
            len if len < forms_count => {
                return Err(ReaderError::Internal(
                    "unexpectedly dropped forms during reader macro".to_string(),
                ));
            }
            _ => {}
        }
        let form = self.forms.pop().expect("just read form");
        let expansion = list_with_values(
            [Value::Symbol(identifier.to_string(), None), form]
                .iter()
                .cloned(),
        );
        self.forms.push(expansion);
        Ok(())
    }

    fn read_form(&mut self, next_char: char, stream: &mut Stream) -> Result<(), ReaderError> {
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
                self.read_dispatch(stream)?;
            }
            '@' => {
                stream.next().expect("from peek");
                self.read_macro("deref", stream)?;
            }
            '\'' => {
                stream.next().expect("from peek");
                self.read_macro("quote", stream)?;
            }
            '`' => {
                stream.next().expect("from peek");
                self.read_macro("quasiquote", stream)?;
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
                self.read_macro(identifier, stream)?;
            }
            '"' => self.read_string(stream)?,
            ch if is_token(ch) => self.read_atom(ch, stream)?,
            _ => unreachable!(),
        }
        Ok(())
    }

    fn read_from_stream(&mut self, stream: &mut Stream) -> Result<(), ReaderError> {
        while let Some((_, ch)) = stream.peek() {
            let ch = *ch;
            if is_whitespace(ch) {
                self.read_whitespace(stream)?;
                continue;
            }
            if is_comment(ch) {
                self.read_comment(stream)?;
                continue;
            }
            self.read_form(ch, stream)?;
            if matches!(self.parse_state, ParseState::Exiting) {
                break;
            }
        }
        Ok(())
    }

    fn read(mut self, input: &'a str) -> Result<Vec<Value>, ReaderError> {
        self.input = input;
        let mut stream = input.char_indices().peekable();
        self.read_from_stream(&mut stream)?;
        if stream.next().is_some() {
            return Err(ReaderError::UnexpectedAdditionalInput);
        }
        Ok(self.forms)
    }
}

pub fn read(input: &str) -> Result<Vec<Value>, ReaderError> {
    Reader::new().read(input)
}

#[cfg(test)]
mod tests {
    use super::{
        list_with_values, map_with_values, read, set_with_values, vector_with_values, Value::*,
    };
    use itertools::Itertools;

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
            ("-", vec![Symbol("-".into(), None)], "-"),
            ("-baz", vec![Symbol("-baz".into(), None)], "-baz"),
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
                Err(e) => {
                    panic!("{} (from {})", e, input);
                }
            }
        }
    }
}
