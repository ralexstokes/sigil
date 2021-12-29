test:
    cargo test
fmt:
    cargo fmt
lint: fmt
    cargo clippy
build:
    cargo build
run-ci: lint build test
repl:
    cargo run --features repl
