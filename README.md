# sigil

a lisp oxide

# about

`sigil` is a lisp intended to bridge the gap between "low-level" code written in Rust and places where "high-level" code is desired.

if you have a question about how `sigil` works as a language, you should assume it has [Clojure](https://clojure.org/) semantics.

this crate provides an interpreter meant to be embedded into other Rust code. an example of such an embedding can be found with the repl under `/examples`.

# contributing, etc.

contributions are very welcome! if you are keen to join, i'd suggest messaging me on twitter or somewhere to see the current status.

# usage

To open a repl:

`cargo run --example repl`

To run a file:

`cargo run --example repl -- from-file $FILE_PATH`

# status

things are still very much a work-in-progress still. the "core" interpreter is written but there are several features worth adding:

- [ ] tools for interop with Rust
- [ ] metadata
- [ ] namespaces
- [ ] language-level features
  - [ ] other primitives
  - [ ] "collections are functions of their keys"
  - [ ] argument destructuring
  - [ ] rich `fn` macro
  - [ ] some reader macros e.g. `#( %1 )`
- [ ] precise errors, in the style of Rust
- [ ] GC and related memory story

i have vague plans to accelerate computation with a web-assembly VM at some point.
