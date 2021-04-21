use crate::value::{list_with_values, map_with_values, set_with_values, vector_with_values, Value};
use combine::error::ParseError;
use combine::error::StreamError;
use combine::parser;
use combine::parser::{
    char::alpha_num, char::char, char::digit, char::space, choice::choice, choice::optional,
    combinator::attempt, range::recognize, repeat::many, repeat::many1, repeat::skip_many,
    repeat::skip_many1, repeat::skip_until, sequence::between, token::eof, token::one_of,
};
use combine::stream::position::Stream;
use combine::stream::StreamErrorFor;
use combine::{EasyParser, Parser, RangeStream};
use std::string::String as StdString;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReaderError {
    #[error("{0}")]
    ParserError(String),
}

#[inline]
fn read_form<'a, Input>() -> impl Parser<Input, Output = Value> + 'a
where
    Input: RangeStream<Token = char, Range = &'a str> + 'a,
    Input::Error: ParseError<Input::Token, Input::Range, Input::Position>,
{
    let whitespace = || skip_many(space().or(char(',')));
    whitespace().with(read_form_inner()).skip(whitespace())
}

parser! {
    #[inline]
    fn read_form_inner['a, Input]()(Input) -> Value
    where [ Input: RangeStream<Token = char, Range = &'a str> + 'a]
    {
        let number = optional(char('-')).and(recognize(skip_many1(digit()))).map(|(sign, digits): (Option<char>, &str)| {
            let value = digits.parse::<i64>().unwrap();
            let result = if sign.is_some() {
                -value
            } else {
                value
            };
            Value::Number(result)
        });

        let string =
            (char('"'), recognize(skip_until(char('"'))), char('"')).map(|(_, s, _): (_, &str, _)| Value::String(s.to_string()));
        let identifier_tokens = || alpha_num().or(one_of("*+!-_'?<>=/&".chars()));
        let identifier = || many1(identifier_tokens());
        let identifier_with_optional_namespace = || identifier().and_then(|id_with_maybe_ns: StdString| {
            let partitions = id_with_maybe_ns.split(|c| c == '/').collect::<Vec<&str>>();
            match partitions.len() {
                1 => Ok((partitions[0].to_string(), None)), // identifier without namespace
                2 => {
                    let ns = &partitions[0];
                    let id = &partitions[1];
                    if ns.is_empty() {
                        if id.is_empty() {
                            // identifier "/" without namespace
                            Ok(("/".to_string(), None))
                        } else {
                            Err(StreamErrorFor::<Input>::expected_static_message("unexpected / in symbol"))
                        }
                    } else {
                        if id.is_empty() {
                            Err(StreamErrorFor::<Input>::expected_static_message("unexpected / in symbol"))
                        } else {
                            // identifier with namespace
                            Ok((id.to_string(), Some(ns.to_string())))
                        }
                    }
                }
                3 => {
                    let ns = &partitions[0];
                    let sep = &partitions[1];
                    let id = &partitions[1];
                    if sep.is_empty() && id.is_empty() {
                        // identifier "/" with namespace
                        Ok(("/".to_string(), Some(ns.to_string())))
                    } else {
                            Err(StreamErrorFor::<Input>::expected_static_message("unexpected / in symbol"))
                    }
                }
                _ => Err(StreamErrorFor::<Input>::expected_static_message("unexpected / in symbol")),
            }
        });

        let symbol = identifier_with_optional_namespace().and_then(|(id, ns)| {
            match id.as_ref() {
                "nil" => {
                    if ns.is_none() {
                        Ok(Value::Nil)
                    } else {
                        Err(StreamErrorFor::<Input>::expected_static_message("can only attach namespace to symbol"))
                    }
                }
                "true" => {
                    if ns.is_none() {
                        Ok(Value::Bool(true))
                    } else {
                        Err(StreamErrorFor::<Input>::expected_static_message("can only attach namespace to symbol"))
                    }
                }
                "false" => {
                    if ns.is_none() {
                        Ok(Value::Bool(false))
                    } else {
                        Err(StreamErrorFor::<Input>::expected_static_message("can only attach namespace to symbol"))
                    }
                }
                _ => {
                    Ok(Value::Symbol(id, ns))
                }
            }
        });
        let keyword = (char(':'), identifier_with_optional_namespace()).map(|(_, (id, ns))| Value::Keyword(id, ns));

        let list = between(char('('), char(')'), many::<Vec<_>, _,_>(read_form())).map(list_with_values);
        let vector = between(char('['), char(']'), many::<Vec<_>, _,_>(read_form())).map(vector_with_values);
        let map = between(char('{'), char('}'), many::<Vec<_>, _, _>((read_form(), read_form()))).map(map_with_values);
        let set = (char('#'), between(char('{'), char('}'), many::<Vec<_>, _, _>(read_form()))).map(|(_, elems)| set_with_values(elems));
        let deref = char('@').with(read_form()).map(|form| list_with_values([Value::Symbol("deref".to_string(), None), form].iter().map(|f| f.clone())));
        let quote = char('\'').with(read_form()).map(|form| list_with_values([Value::Symbol("quote".to_string(), None), form].iter().map(|f| f.clone())));
        let quasiquote = char('`').with(read_form()).map(|form| list_with_values([Value::Symbol("quasiquote".to_string(), None), form].iter().map(|f| f.clone())));
        let unquote_with_optional_splice = (char('~'), optional(char('@')), read_form()).map(|(_, splice_opt, form)| {
            if splice_opt.is_some() {
                list_with_values([Value::Symbol("splice-unquote".to_string(), None), form].iter().cloned())
            } else {
                list_with_values([Value::Symbol("unquote".to_string(), None), form].iter().cloned())
            }
        });

        choice((
            attempt(number),
            string,
            keyword,
            deref,
            quote,
            quasiquote,
            unquote_with_optional_splice,
            symbol,
            list,
            vector,
            map,
            set,
        ))
    }
}

pub fn read(input: &str) -> Result<Vec<Value>, ReaderError> {
    let comment = (char(';'), skip_until(eof()));
    let whitespace = || skip_many(space().or(char(',')));
    let mut parser =
        (whitespace(), many(read_form()), optional(comment), eof()).map(|(_, result, _, _)| result);
    let parse_result = parser.easy_parse(Stream::new(input));
    match parse_result {
        Ok((forms, _)) => Ok(forms),
        Err(error) => Err(ReaderError::ParserError(error.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        list_with_values, map_with_values, read, set_with_values, vector_with_values, Value::*,
    };
    use itertools::join;

    #[test]
    fn test_read_basic() {
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
            ("-", vec![Symbol("-".into(), None)], "-"),
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
            ("\"hi\"", vec![String("hi".into())], "\"hi\""),
            (
                "\"123foo\" true",
                vec![String("123foo".into()), Bool(true)],
                "\"123foo\" true",
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
            ("#{}", vec![set_with_values(vec![])], "#{}"),
            ("#{1}", vec![set_with_values(vec![Number(1)])], "#{1}"),
            ("#{   1}", vec![set_with_values(vec![Number(1)])], "#{1}"),
            ("#{   1  }", vec![set_with_values(vec![Number(1)])], "#{1}"),
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
        ];
        for (input, expected_read, expected_print) in cases {
            match read(input) {
                Ok(result) => {
                    assert_eq!(result, expected_read);
                    if expected_print != "" {
                        let print = join(result, " ");
                        assert_eq!(print, expected_print);
                    }
                }
                Err(e) => {
                    dbg!(input);
                    dbg!(e);
                    panic!();
                }
            }
        }
    }
}
