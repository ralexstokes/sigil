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

fn ssz_deserialize_example_container(
    _: &mut Interpreter,
    args: &[Value],
) -> EvaluationResult<Value> {
    let data = &args[0];
    // bytes from data
    let bytes = &[];
    let container = ExampleContainer::deserialize(bytes).expect("can deserialize");
    Ok(container.into())
}

fn main() {
    let ssz_rs_bindings = [(
        "deserialize-example-container",
        Value::Primitive(ssz_deserialize_example_container),
    )];
    let ssz_rs_namespace = Namespace::from(ssz_rs_bindings);
    let namespaces = [("github.com/ralexstokes/ssz_rs", ssz_rs_namespace)];
    let mut interpreter = InterpreterBuilder::default()
        .with_namespaces(namespaces)
        .build();

    let source = r#"
    (ns
      (:require
        [github.com/ralexstokes/ssz_rs :as ssz]
        [std/io :as io]))


    (defn load-ssz-example-container [path]
      (ssz/deserialize-example-container (io/read-bytes path)))

    (def ssz-source "example.ssz")
    (load-ssz-example-container ssz-source)
    "#;

    let result: ExampleContainer = interpreter
        .evaluate_from_source(source)
        .unwrap()
        .first()
        .unwrap()
        .into();
    dbg!(result);
}
