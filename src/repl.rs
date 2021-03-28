use crate::interpreter::Interpreter;
use crate::reader::read;
use std::default::Default;
use std::io::{self, BufRead, Write};

#[derive(Default)]
pub struct StdRepl {
    interpreter: Interpreter,
}

impl StdRepl {
    pub fn run(&mut self) -> io::Result<()> {
        let stdin = io::stdin();
        let reader = stdin.lock();

        let mut current_namespace = &self.interpreter.current_namespace;

        print!("{}=> ", current_namespace);
        io::stdout().flush()?;
        for line in reader.lines() {
            let line = line.unwrap();
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
            current_namespace = &self.interpreter.current_namespace;
            print!("{}=> ", current_namespace);
            io::stdout().flush()?;
        }
        Ok(())
    }
}
