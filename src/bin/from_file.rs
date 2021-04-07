use sigil::repl::StdRepl;
use std::env;

fn main() {
    let args = env::args();
    let mut repl = StdRepl::new().with_command_line_args(args);

    if let Err(e) = repl.run_file_from_args() {
        println!("{}", e);
    }
}
