## Change log 5.6 -> 7

- `Session::ask()` -> `Session::ask_parts()`
- `Session::ask_string()` -> `Session::ask()` and can take any part valid apart from just `String`
- `Part` -> `PartType` inside a struct. To migrate, just use `.into()`. Eg.  
    **Before**
    ```rust
    session.ask(vec![Part::InlineData(InlineData::new(base64, mime)), ..])
    ```
    **Now**
    ```rust
    session.ask_parts(vec![InlineData::new(base64, mime).into(), ..])
    ```
- `response.get_text_no_think("\n")` -> `response.get_chat().get_text_no_think("\n")`
