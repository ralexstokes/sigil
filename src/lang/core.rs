pub const SOURCE: &str = include_str!("./core.sigil");

#[cfg(test)]
mod tests {
    use crate::interpreter::InterpreterBuilder;
    use crate::reader::read;
    use crate::value::Value;

    #[test]
    fn test_core_macros() -> Result<(), Box<dyn std::error::Error>> {
        let source = "(defn f [x] (let [y 29] (+ x y))) (f 1)";
        let expected_result = Value::Number(30);

        let builder = InterpreterBuilder::default();
        let mut interpreter = builder.build();

        let forms = read(source)?;
        let (first, rest) = forms.split_first().expect("some form");
        let result = interpreter.evaluate(first)?;
        let result = rest
            .iter()
            .try_fold(result, |_, form| interpreter.evaluate(form))?;
        assert_eq!(result, expected_result);
        Ok(())
    }
}
