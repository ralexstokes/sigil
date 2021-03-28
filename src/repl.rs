use crate::interpreter::Interpreter;
use crate::reader::read;
use rustyline::error::ReadlineError;
use rustyline::Editor;
use std::default::Default;
use std::error::Error;
use std::fmt::Write;

#[derive(Default)]
pub struct StdRepl {
    interpreter: Interpreter,
}

impl StdRepl {
    pub fn run(&mut self) -> Result<(), Box<dyn Error>> {
        let mut editor = Editor::<()>::new();
        let mut prompt_buffer = String::new();

        let mut current_namespace = self.interpreter.current_namespace();
        write!(prompt_buffer, "{}=> ", current_namespace)?;
        loop {
            let next_line = editor.readline(&prompt_buffer);
            match next_line {
                Ok(line) => {
                    let forms = match read(&line) {
                        Ok(forms) => forms,
                        Err(e) => {
                            println!("{}", e);
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
        Ok(())
    }
}
