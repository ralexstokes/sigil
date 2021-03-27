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
    ] {
        match read(input) {
            Ok(result) => println!("{:?}", result),
            Err(e) => println!("{:?}", e),
        }
    }
}
