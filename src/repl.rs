use crate::interpreter::Interpreter;
use crate::reader::read;
use rustyline::error::ReadlineError;
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter};
use rustyline::Editor;
use rustyline_derive::{Completer, Helper, Hinter, Validator};
use std::borrow::Cow;
use std::default::Default;
use std::error::Error;
use std::fmt::Write;

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

#[derive(Completer, Helper, Hinter, Validator)]
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

impl StdRepl {
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
                    for form in forms {
                        match self.interpreter.evaluate(form) {
                            Ok(result) => println!("{}", result),
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
