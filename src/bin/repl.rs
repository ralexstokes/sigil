use clap::{Parser, Subcommand};
use sigil::{repl_with_interpreter, Interpreter};
use std::env;
use std::error::Error;

#[derive(Parser)]
#[clap(about, version, author)]
struct Options {
    /// Points to a file containing the "core" source
    #[clap(long)]
    with_core_source: Option<String>,
    #[clap(subcommand)]
    from_file: Option<FromFileCommand>,
}

#[derive(Subcommand)]
enum FromFileCommand {
    /// Evaluates source from a file
    FromFile {
        /// the file path to read
        path: String,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse();

    let interpreter = Interpreter::default();
    let mut repl = repl_with_interpreter(interpreter).with_command_line_args(env::args());

    let result = if let Some(FromFileCommand::FromFile { path }) = options.from_file {
        repl.run_from_file(path)
    } else {
        repl.run()
    };
    if let Err(err) = result {
        println!("{}", err);
    }
    Ok(())
}
