use sigil::reader::read;

fn main() {
    for input in vec![
        "1337",
        "nil",
        "true",
        "false",
        ",,,",
        "  ",
        " , ",
        "true, , , false",
        "",
        "foobar true",
        "foo/bar",
        "\"\"",
        "\"hi\"",
        "\"123foo\" true",
        r#""abc""#,
        ":foobar",
        ":net/hi",
        ":a0987234",
        "foo/bar true ;; some comment",
        "foo/bar true ;; some comment\nhi/baz",
        "()",
        "(a b c)",
        "(12 :foo/bar \"extra\")",
        "[]",
        "[a b c]",
        "[12 :foo/bar \"extra\"]",
        "{}",
        "{:a b :c 12}",
        "{12 :foo/bar \"extra\" bar}",
        "() []",
        "(())",
        "(([]))",
        "(([]))()",
        "(12 (true [34 false]))() (7)",
    ] {
        match read(input) {
            Ok(result) => println!("{:?}", result),
            Err(e) => println!("{:?}", e),
        }
    }
}
