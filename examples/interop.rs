use sigil::{InterpreterBuilder, Value};
use ssz_rs::prelude::*;

#[derive(Debug, SimpleSerialize)]
struct ExampleContainer {
    a: usize,
}

impl From<&Value> for ExampleContainer {
    fn from(value: &Value) -> Self {
        let key = Value::String("a".to_string());
        let a = match value {
            Value::Map(m) => match m.get(&key).unwrap() {
                Value::Number(a) => *a,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        ExampleContainer {
            a: a.try_into().unwrap(),
        }
    }
}

fn main() {
    let mut interpreter = InterpreterBuilder::default().build();

    let source = r#"
    (ns
      (:require
        [github.com/ralexstokes/ssz_rs :as ssz]
        [std/io :as io]))


    (def ssz-source "example.ssz")
    (defn load-ssz-container [path schema]
      (ssz/deserialize (io/read-bytes path) schema))
    "#;

    let result: ExampleContainer = interpreter
        .evaluate_from_source(source)
        .unwrap()
        .first()
        .unwrap()
        .into();
    dbg!(result);
}
