use core::panic;
use sigil::{InterpreterBuilder, StdRepl};
use std::env;
use std::thread;
use tempfile::NamedTempFile;

const STACK_SIZE: usize = 4194304; // 4 MiB
const SELF_HOSTING_REPL_SOURCE: &str = include_str!("./self-hosted.sigil");

// Run some test code but from the context of the self-hosted interpreter
fn run_tests_as_self_hosted() {
    let mut interpreter = InterpreterBuilder::default().build();
    let mut args = env::args().into_iter().collect::<Vec<String>>();
    args.push(String::from("tests/tests.sigil"));
    interpreter.intern_args(args.into_iter());
    let temp_file = NamedTempFile::new().expect("can make temp file on host");
    let mut repl = StdRepl::new(interpreter, temp_file.path());
    if let Err(err) = repl.run_from_source(SELF_HOSTING_REPL_SOURCE) {
        panic!("{}", err);
    }
}

#[test]
fn verify_tests_self_hosted() {
    // NOTE: The test exercising the self-hosted implementation overflows the default stack
    // provided by a new thread under `cargo test`. Run this test in a thread with
    // a much larger stack of `STACK_SIZE` to avoid a stack overflow.
    // See here for more info about the default stack size: https://doc.rust-lang.org/std/thread/index.html#stack-size
    //
    // NOTE: the test performs fine with the `release` build.
    // NOTE: the self-hosted implementation assumes TCO; determine if the interpreter
    // still consumes this much stack space after TCO is implemented.
    let builder = thread::Builder::new().stack_size(STACK_SIZE);
    let handler = builder.spawn(run_tests_as_self_hosted).unwrap();
    handler.join().unwrap();
}
