use clap::{AppSettings, Clap};
use sigil::{InterpreterBuilder, StdRepl};
use std::env;
use std::error::Error;

#[derive(Clap)]
#[clap(setting = AppSettings::ColoredHelp)]
struct Options {
    #[clap(long)]
    /// Points to a file containing the "core" source
    with_core_source: Option<String>,
    #[clap(subcommand)]
    from_file: Option<FromFileCommand>,
}

#[derive(Clap)]
enum FromFileCommand {
    /// Evaluates source from a file
    FromFile {
        /// the file path to read
        path: String,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse();

    let mut builder = InterpreterBuilder::default();
    if let Some(core_source) = &options.with_core_source {
        builder.with_core_file_path(core_source);
    }
    let interpreter = builder.build();

    let mut repl = StdRepl::with_interpreter(interpreter).with_command_line_args(env::args());

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
