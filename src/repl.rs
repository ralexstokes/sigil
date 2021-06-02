use crate::interpreter::Interpreter;
use crate::reader::read;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter};
use rustyline::history::Direction;
use rustyline::{Context, Editor};
use rustyline_derive::{Helper, Hinter, Validator};
use std::borrow::Cow;
use std::default::Default;
use std::env::Args;
use std::error::Error;
use std::fmt::Write;
use std::fs;

pub struct StdRepl {
    interpreter: Interpreter,
    history_path: String,
}

impl Default for StdRepl {
    fn default() -> Self {
        Self {
            interpreter: Interpreter::default(),
            history_path: "sigil.history".to_string(),
        }
    }
}

#[derive(Helper, Hinter, Validator)]
struct EditorHelper {
    highlighter: MatchingBracketHighlighter,
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
        _pos: usize,
        ctx: &Context<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        let history = ctx.history();
        let mut iter = line.chars().rev().enumerate();
        let mut target_index = None;
        // advance until not whitespace
        while let Some((index, ch)) = iter.next() {
            target_index = Some(index);
            if !ch.is_whitespace() {
                break;
            }
        }
        // advance until whitespace again...
        while let Some((index, ch)) = iter.next() {
            target_index = Some(index);
            if ch.is_whitespace() {
                break;
            }
        }
        if let Some(target_index) = target_index {
            let line_index = line.len() - target_index - 1;

            if let Some(completion_index) = history.starts_with(line, 0, Direction::Forward) {
                let history_match = &history[completion_index];
                let pair = Pair {
                    display: history_match.to_string(),
                    replacement: history_match.to_string(),
                };
                return Ok((line_index, vec![pair]));
            }
        }
        Ok((0, vec![]))
    }
}

impl StdRepl {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_command_line_args(mut self, args: Args) -> Self {
        self.interpreter.intern_args(args);
        self
    }

    pub fn run_file_from_args(&mut self) -> Result<(), Box<dyn Error>> {
        let path = self.interpreter.command_line_arg(1)?;
        let contents = fs::read_to_string(path)?;
        for line in contents.lines() {
            let forms = read(&line)?;
            for form in forms.iter() {
                self.interpreter.evaluate(form)?;
            }
        }
        Ok(())
    }

    pub fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let helper = EditorHelper {
            highlighter: MatchingBracketHighlighter::new(),
        };
        let mut editor = Editor::<EditorHelper>::new();
        editor.set_helper(Some(helper));

        let _ = editor.load_history(&self.history_path);

        let mut prompt_buffer = String::new();

        let mut current_namespace = self.interpreter.current_namespace();
        write!(prompt_buffer, "{}=> ", current_namespace)?;
        loop {
            let next_line = editor.readline(&prompt_buffer);
            match next_line {
                Ok(line) => {
                    editor.add_history_entry(line.as_str());
                    let forms = match read(&line) {
                        Ok(forms) => forms,
                        Err(e) => {
                            print!("{}", e);
                            continue;
                        }
                    };
                    for form in forms.iter() {
                        match self.interpreter.evaluate(form) {
                            Ok(result) => {
                                let mut line = String::new();
                                write!(line, "{}", result)?;
                                editor.add_history_entry(line.as_str());
                                println!("{}", line);
                            }
                            Err(e) => {
                                println!("{}", e);
                            }
                        }
                    }
                    if self.interpreter.current_namespace() != current_namespace {
                        current_namespace = self.interpreter.current_namespace();
                        prompt_buffer.clear();
                        write!(prompt_buffer, "{}=> ", current_namespace)?;
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
                    println!("error: {:?}", err);
                    break;
                }
            }
        }
        editor.save_history(&self.history_path)?;
        Ok(())
    }
}
