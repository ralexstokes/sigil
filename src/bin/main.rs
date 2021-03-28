use sigil::reader::read;

fn main() {
    for input in vec![
        "nil true,,,, false 12 \"hi\" :bar/foo baz (+ [] {:a 2} #{1 2 3}) ;; comment",
        "(defn foo [a] (+ a 1))",
    ] {
        match read(input) {
            Ok(result) => {
                for form in result  {
                    println!("{}", form);
                }
            },
            Err(e) => println!("{:?}", e),
        }
    }
}
