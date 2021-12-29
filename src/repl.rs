use crate::interpreter::{EvaluationError, Interpreter, InterpreterBuilder, SymbolIndex};
use crate::reader::{is_structural, is_symbolic, is_token, read, ReadError};
use crate::value::Value;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter};
use rustyline::{Context, Editor};
use rustyline_derive::{Helper, Hinter, Validator};
use std::borrow::Cow;
use std::cell::RefCell;
use std::default::Default;
use std::env::Args;
use std::fmt::{self, Debug, Write};
use std::fs;
use std::io;
use std::path::Path;
use std::rc::Rc;
use thiserror::Error;

const DEFAULT_HISTORY_PATH: &str = ".sigil.history";
const DEFAULT_SOURCE_SPAN_LEN: usize = 64;

#[derive(Error, Debug)]
pub enum ReplError<'a> {
    #[error("error reading: {0}")]
    Read(ReadError, &'a str),
    #[error("error evaluating: {0}")]
    Eval(EvaluationError, Value),
    #[error("error with I/O: {0}")]
    IO(#[from] io::Error),
    #[error("error with formatting: {0}")]
    Fmt(#[from] fmt::Error),
    #[error("error with readline: {0}")]
    Readline(#[from] ReadlineError),
}

pub struct StdRepl<P: AsRef<Path>> {
    editor: Editor<EditorHelper>,
    interpreter: Interpreter,
    history_path: P,
}

impl Default for StdRepl<&'static str> {
    fn default() -> Self {
        let interpreter = InterpreterBuilder::default().build();
        Self::new(interpreter, DEFAULT_HISTORY_PATH)
    }
}

impl<P: AsRef<Path>> Debug for StdRepl<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StdRepl {{ .. }}")
    }
}

#[derive(Helper, Hinter, Validator)]
struct EditorHelper {
    highlighter: MatchingBracketHighlighter,
    symbol_index: Rc<RefCell<SymbolIndex>>,
}

impl Highlighter for EditorHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        self.highlighter.highlight(line, pos)
    }

    fn highlight_char(&self, line: &str, pos: usize) -> bool {
        self.highlighter.highlight_char(line, pos)
    }
}

impl Completer for EditorHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        if line.is_empty() {
            return Ok((0, vec![]));
        }

        let mut start = line[..pos]
            .rfind(|ch| !is_token(ch) || is_structural(ch) || !is_symbolic(ch))
            .unwrap_or(0);
        let mut ch = line[start..start + 1].chars().take(1);
        if let Some(ch) = ch.next() {
            if is_structural(ch) {
                start += 1;
            }
        }
        let target = &line[start..pos];

        let mut matches = vec![];
        let index = self.symbol_index.borrow();
        for symbol in index.iter() {
            if symbol.starts_with(target) {
                matches.push(Pair {
                    display: symbol.clone(),
                    replacement: symbol.clone(),
                });
            }
        }
        Ok((start, matches))
    }
}

pub fn repl_with_interpreter(interpreter: Interpreter) -> StdRepl<&'static str> {
    StdRepl::new(interpreter, DEFAULT_HISTORY_PATH)
}

fn consume_error(err: ReplError) {
    match err {
        ReplError::Read(err, source) => {
            let context = err.context(&source);
            let span_len = std::cmp::min(context.len(), DEFAULT_SOURCE_SPAN_LEN);
            println!(
                "error reading: {} at {} from input:\n{}",
                err,
                &context[..span_len],
                source,
            );
        }
        ReplError::Eval(err, form) => {
            println!("error evaluating `{}`: {}", form.to_readable_string(), err);
        }
        other => println!("{}", other),
    }
}

impl<P: AsRef<Path>> StdRepl<P> {
    pub fn new(mut interpreter: Interpreter, history_path: P) -> Self {
        let symbol_index = Rc::new(RefCell::new(SymbolIndex::new()));
        interpreter.register_symbol_index(symbol_index.clone());

        let helper = EditorHelper {
            highlighter: MatchingBracketHighlighter::new(),
            symbol_index,
        };
        let mut editor = Editor::<EditorHelper>::new();
        editor.set_helper(Some(helper));

        Self {
            editor,
            interpreter,
            history_path,
        }
    }

    pub fn with_command_line_args(mut self, args: Args) -> Self {
        self.interpreter.intern_args(args);
        self
    }

    pub fn run_from_source<'a>(&mut self, source: &'a str) -> Result<Vec<Value>, ReplError<'a>> {
        let forms = read(&source).map_err(|err| ReplError::Read(err, source))?;
        let mut results = vec![];
        for form in forms.iter() {
            match self.interpreter.evaluate(form) {
                Ok(result) => {
                    results.push(result);
                }
                Err(err) => {
                    return Err(ReplError::Eval(err, form.clone()));
                }
            }
        }
        Ok(results)
    }

    pub fn run_from_file<Q: AsRef<Path>>(&mut self, path: Q) -> Result<(), ReplError> {
        let contents = fs::read_to_string(path)?;
        if let Err(err) = self.run_from_source(&contents) {
            consume_error(err);
        }
        Ok(())
    }

    pub fn run(&mut self) -> Result<(), ReplError> {
        let _ = self.editor.load_history(&self.history_path);

        let mut prompt_buffer = String::new();
        loop {
            prompt_buffer.clear();
            write!(prompt_buffer, "{}=> ", self.interpreter.current_namespace())?;
            let next_line = self.editor.readline(&prompt_buffer);
            match next_line {
                Ok(line) => {
                    self.editor.add_history_entry(line.as_str());
                    match self.run_from_source(&line) {
                        Ok(results) => {
                            for result in results {
                                println!("{}", result.to_readable_string());
                            }
                        }
                        Err(err) => {
                            consume_error(err);
                            continue;
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!("^D");
                    break;
                }
                Err(err) => {
                    println!("error reading line from repl: {}", err);
                    break;
                }
            }
        }
        self.editor.save_history(&self.history_path)?;
        Ok(())
    }
}
