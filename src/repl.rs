use crate::interpreter::{Interpreter, InterpreterBuilder, SymbolIndex};
use crate::reader::{is_structural, is_symbolic, is_token, read};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter};
use rustyline::{Context, Editor};
use rustyline_derive::{Helper, Hinter, Validator};
use std::borrow::Cow;
use std::cell::RefCell;
use std::default::Default;
use std::env::Args;
use std::error::Error;
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::rc::Rc;

const DEFAULT_HISTORY_PATH: &str = ".sigil.history";
const DEFAULT_SOURCE_SPAN_LEN: usize = 64;

pub struct StdRepl<'a> {
    editor: Editor<EditorHelper>,
    interpreter: Interpreter,
    history_path: &'a str,
}

impl<'a> Default for StdRepl<'a> {
    fn default() -> Self {
        let interpreter = InterpreterBuilder::default().build();
        Self::with_interpreter(interpreter)
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

impl<'a> StdRepl<'a> {
    pub fn new(mut interpreter: Interpreter, history_path: &'a str) -> Self {
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

    pub fn with_interpreter(interpreter: Interpreter) -> Self {
        Self::new(interpreter, DEFAULT_HISTORY_PATH)
    }

    pub fn with_command_line_args(mut self, args: Args) -> Self {
        self.interpreter.intern_args(args);
        self
    }

    pub fn run_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn Error>> {
        let contents = fs::read_to_string(path)?;
        let forms = match read(&contents) {
            Ok(forms) => forms,
            Err(err) => {
                let context = err.context(&contents);
                let span_len = std::cmp::min(context.len(), DEFAULT_SOURCE_SPAN_LEN);
                println!("error reading source: {} at {}", err, &context[..span_len],);
                return Ok(());
            }
        };
        for form in forms.iter() {
            match self.interpreter.evaluate(form) {
                Ok(..) => {}
                Err(err) => {
                    println!("error evaluating `{}`: {}", form.to_readable_string(), err);
                    continue;
                }
            }
        }
        Ok(())
    }

    pub fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let _ = self.editor.load_history(&self.history_path);

        let mut prompt_buffer = String::new();
        loop {
            prompt_buffer.clear();
            write!(prompt_buffer, "{}=> ", self.interpreter.current_namespace())?;
            let next_line = self.editor.readline(&prompt_buffer);
            match next_line {
                Ok(line) => {
                    self.editor.add_history_entry(line.as_str());
                    let forms = match read(&line) {
                        Ok(forms) => forms,
                        Err(err) => {
                            let context = err.context(&line);
                            let span_len = std::cmp::min(context.len(), DEFAULT_SOURCE_SPAN_LEN);
                            println!(
                                "error reading `{}`: {} while reading {}",
                                &line,
                                err,
                                &context[..span_len]
                            );
                            continue;
                        }
                    };
                    for form in forms.iter() {
                        match self.interpreter.evaluate(form) {
                            Ok(result) => {
                                println!("{}", result.to_readable_string());
                            }
                            Err(e) => {
                                println!("error evaluating `{}`: {}", form.to_readable_string(), e);
                            }
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
