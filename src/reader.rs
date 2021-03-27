use combine::eof;
use combine::from_str;
use combine::parser::{
    char::alpha_num, char::char, char::digit, char::space, choice::choice, choice::or,
    range::range, range::recognize, repeat::many1, repeat::skip_many1, repeat::skip_until,
};
use combine::stream::position::Stream;
use combine::EasyParser;
use combine::Parser;
use thiserror::Error;

#[derive(Debug, PartialEq)]
pub enum Form<'a> {
    Nil,
    Bool(bool),
    Number(u64),
    String(&'a str),
    Keyword(&'a str),
    Symbol(&'a str),
    Comment(&'a str),
    Whitespace,
}

#[derive(Debug, Error)]
pub enum ReaderError {
    #[error("reader could not parse the full input")]
    ExtraneousInput,
    #[error("{0}")]
    ParserError(String),
    #[error("unknown error occurred")]
    Unknown,
}

pub fn read(input: &str) -> Result<Vec<Form>, ReaderError> {
    let number = from_str(recognize(skip_many1(digit()))).map(Form::Number);
    let nil = recognize(range("nil")).map(|_| Form::Nil);
    let true_parser = recognize(range("true")).map(|_| Form::Bool(true));
    let false_parser = recognize(range("false")).map(|_| Form::Bool(false));
    let whitespace = recognize(skip_many1(or(space(), char(',')))).map(|_| Form::Whitespace);
    let keyword = (char(':'), recognize(skip_many1(alpha_num()))).map(|(_, s)| Form::Keyword(s));
    let string =
        (char('"'), recognize(skip_until(char('"'))), char('"')).map(|(_, s, _)| Form::String(s));
    let symbol = recognize(skip_many1(alpha_num())).map(Form::Symbol);

    let forms = choice((
        whitespace,
        number,
        nil,
        true_parser,
        false_parser,
        string,
        keyword,
        symbol,
    ));
    let eof = recognize(eof()).map(|_| vec![Form::Whitespace]);

    let mut parser = or(eof, many1::<Vec<_>, _, _>(forms));
    let parse_result = parser.easy_parse(Stream::new(input));
    match parse_result {
        Ok((result, _)) => Ok(result),
        Err(error) => Err(ReaderError::ParserError(error.to_string())),
    }
}
