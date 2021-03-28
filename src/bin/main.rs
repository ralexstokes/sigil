use sigil::repl::StdRepl;

fn main() {
    match StdRepl::default().run() {
        Ok(_) => (),
        Err(e) => println!("{}", e),
    }
}
