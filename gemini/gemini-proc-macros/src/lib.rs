use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Attribute, Data, DeriveInput, Fields, FnArg, ItemFn, Lit, Meta, Pat, Type, parse_macro_input,
};

/// Attribute macro to mark a function as callable by Gemini.
///
/// This macro generates a schema for the function and an `execute` method to call it
/// with JSON arguments.
///
/// # Requirements
/// - Function arguments must be owned types that implement `GeminiSchema` (e.g., `String`, `i32`, `bool`).
/// - References are not supported.
/// - The function can be `async` and can return a `Result` (the `Ok` value must implement `Serialize`).
///
/// # Example
/// ```ignore
/// #[gemini_function]
/// /// Returns the current weather for a location.
/// fn get_weather(location: String) -> String {
///     format!("The weather in {} is sunny.", location)
/// }
/// ```
#[proc_macro_attribute]
pub fn gemini_function(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let args_struct_name = syn::Ident::new(&format!("{}_args", fn_name), fn_name.span());
    let fn_description = extract_doc_comments(&input_fn.attrs);

    let mut properties = Vec::new();
    let mut required = Vec::new();
    let mut param_names = Vec::new();
    let mut param_types = Vec::new();

    for arg in input_fn.sig.inputs.iter_mut() {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = pat_ident.ident.clone();
                let param_name_str = param_name.to_string();
                let param_type = (*pat_type.ty).clone();
                let param_desc = extract_doc_comments(&pat_type.attrs);

                if has_reference(&param_type) {
                    return syn::Error::new_spanned(
                        &param_type,
                        "references are not supported in gemini_function. Use owned types like String instead.",
                    )
                    .to_compile_error()
                    .into();
                }

                // Remove doc attributes from the function signature so it compiles
                pat_type.attrs.retain(|attr| !attr.path().is_ident("doc"));

                let is_optional = is_option(&param_type);

                properties.push(quote! {
                    let mut schema = <#param_type as GeminiSchema>::gemini_schema();
                    if !#param_desc.is_empty() {
                        if let Some(obj) = schema.as_object_mut() {
                            obj.insert("description".to_string(), serde_json::json!(#param_desc));
                        }
                    }
                    props.insert(#param_name_str.to_string(), schema);
                });

                if !is_optional {
                    required.push(param_name_str);
                }

                param_names.push(param_name);
                param_types.push(param_type);
            }
        }
    }

    let fn_name_str = fn_name.to_string();
    let is_async = input_fn.sig.asyncness.is_some();
    let call_await = if is_async {
        quote! { .await }
    } else {
        quote! {}
    };

    let is_result = match &input_fn.sig.output {
        syn::ReturnType::Default => false,
        syn::ReturnType::Type(_, ty) => {
            let s = quote!(#ty).to_string();
            s.contains("Result")
        }
    };

    let result_handling = if is_result {
        quote! {
            match result {
                Ok(v) => Ok(serde_json::json!(v)),
                Err(e) => Err(e.to_string()),
            }
        }
    } else {
        quote! {
            Ok(serde_json::json!(result))
        }
    };

    let expanded = quote! {
        #input_fn

        #[allow(non_camel_case_types)]
        pub struct #fn_name { }

        #[allow(non_camel_case_types)]
        #[derive(gemini_client_api::serde::Deserialize)]
        pub struct #args_struct_name {
            #(pub #param_names: #param_types,)*
        }

        impl GeminiSchema for #fn_name {
            fn gemini_schema() -> serde_json::Value {
                use serde_json::{json, Map};
                let mut props = Map::new();
                #(#properties)*

                json!({
                    "name": #fn_name_str,
                    "description": #fn_description,
                    "parameters": {
                        "type": "OBJECT",
                        "properties": props,
                        "required": [#(#required),*]
                    }
                })
            }
        }

        impl #fn_name {
            pub async fn execute(args: serde_json::Value) -> Result<serde_json::Value, String> {
                use gemini_client_api::serde::Deserialize;
                let args = #args_struct_name::deserialize(&args).map_err(|e| e.to_string())?;
                let result = #fn_name(#(args.#param_names),*) #call_await;
                #result_handling
            }
            pub async fn execute_with_closure<F, Fut, T>(args: serde_json::Value, f: F) -> Result<T, String>
            where
                F: FnOnce(#(#param_types),*) -> Fut,
                Fut: std::future::Future<Output = T>,
            {
                use gemini_client_api::serde::Deserialize;
                let args = #args_struct_name::deserialize(&args).map_err(|e| e.to_string())?;
                Ok(f(#(args.#param_names),*).await)
            }
        }
    };

    TokenStream::from(expanded)
}

/// Macro to execute function calls requested by Gemini and update the session history, with a custom callback for results.
///
/// # Usage
/// `execute_function_calls_with_callback!(session, callback, function1, function2, ...)`
///
/// The `callback` should be a closure or function that takes `(String, Result<serde_json::Value, String>)`
/// and returns `serde_json::Value`.
/// (function_name, result) is passed to it
///
/// # Returns
/// A `Vec<Option<Result<serde_json::Value, String>>>` containing the results of each function call.
/// - `Some(Ok(value))` if the function was called and succeeded.
/// - `Some(Err(err))` if the function was called but failed.
/// - `None` if the function was not called.
///
/// # Note
/// The `session` is automatically updated with the `FunctionResponse`.
/// The `callback` is invoked with the result of the function execution (whether `Ok` or `Err`)
/// and its return value is used to update the session.
#[proc_macro]
pub fn execute_function_calls_with_callback(input: TokenStream) -> TokenStream {
    use syn::parse::{Parse, ParseStream};
    use syn::{Expr, Token};

    struct ExecuteWithCallbackInput {
        session: Expr,
        _comma1: Token![,],
        callback: Expr,
        _comma2: Token![,],
        functions: syn::punctuated::Punctuated<syn::Path, Token![,]>,
    }

    impl Parse for ExecuteWithCallbackInput {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(ExecuteWithCallbackInput {
                session: input.parse()?,
                _comma1: input.parse()?,
                callback: input.parse()?,
                _comma2: input.parse()?,
                functions: input.parse_terminated(syn::Path::parse, Token![,])?,
            })
        }
    }

    let input = parse_macro_input!(input as ExecuteWithCallbackInput);
    generate_execute_logic(&input.session, &input.callback, &input.functions)
}


/// Attribute macro to derive the `GeminiSchema` trait for a struct or enum.
///
/// This allows the type to be used in structured output (`set_json_mode`) or as a parameter
/// in a `gemini_function`.
///
/// # Requirements
/// - For structs: only named fields are supported.
/// - For enums: only unit variants (no data) are supported.
/// - Field/variant types must also implement `GeminiSchema`.
/// - Doc comments on fields and variants are extracted as descriptions in the schema.
///
/// # Example
/// ```ignore
/// #[gemini_schema]
/// struct SearchResult {
///     /// The title of the page.
///     title: String,
///     /// The URL of the page.
///     url: String,
/// }
/// ```
#[proc_macro_attribute]
pub fn gemini_schema(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let description = extract_doc_comments(&input.attrs);

    let expanded = match &input.data {
        Data::Struct(data) => {
            let mut properties = Vec::new();
            let mut required = Vec::new();

            match &data.fields {
                Fields::Named(fields) => {
                    for field in &fields.named {
                        let field_name = field.ident.as_ref().unwrap();
                        let field_name_str = field_name.to_string();
                        let field_type = &field.ty;
                        let field_desc = extract_doc_comments(&field.attrs);

                        if has_reference(field_type) {
                            return syn::Error::new_spanned(
                                field_type,
                                "references are not supported in gemini_schema. Use owned types instead.",
                            )
                            .to_compile_error()
                            .into();
                        }

                        let is_optional = is_option(field_type);

                        properties.push(quote! {
                            let mut schema = <#field_type as GeminiSchema>::gemini_schema();
                            if !#field_desc.is_empty() {
                                if let Some(obj) = schema.as_object_mut() {
                                    obj.insert("description".to_string(), serde_json::json!(#field_desc));
                                }
                            }
                            props.insert(#field_name_str.to_string(), schema);
                        });

                        if !is_optional {
                            required.push(field_name_str);
                        }
                    }
                }
                _ => panic!("gemini_schema only supports named fields in structs"),
            }

            quote! {
                impl #impl_generics GeminiSchema for #name #ty_generics #where_clause {
                    fn gemini_schema() -> serde_json::Value {
                        use serde_json::{json, Map};
                        let mut props = Map::new();
                        #(#properties)*

                        let mut schema = json!({
                            "type": "OBJECT",
                            "properties": props,
                            "required": [#(#required),*]
                        });

                        if !#description.is_empty() {
                            if let Some(obj) = schema.as_object_mut() {
                                obj.insert("description".to_string(), json!(#description));
                            }
                        }
                        schema
                    }
                }
            }
        }
        Data::Enum(data) => {
            let mut variants = Vec::new();
            for variant in &data.variants {
                if !matches!(variant.fields, Fields::Unit) {
                    panic!("gemini_schema only supports unit variants in enums");
                }
                variants.push(variant.ident.to_string());
            }

            quote! {
                impl #impl_generics GeminiSchema for #name #ty_generics #where_clause {
                    fn gemini_schema() -> serde_json::Value {
                        use serde_json::json;
                        let mut schema = json!({
                            "type": "STRING",
                            "enum": [#(#variants),*]
                        });

                        if !#description.is_empty() {
                            if let Some(obj) = schema.as_object_mut() {
                                obj.insert("description".to_string(), json!(#description));
                            }
                        }
                        schema
                    }
                }
            }
        }
        _ => panic!("gemini_schema only supports structs and enums"),
    };

    let output = quote! {
        #input
        #expanded
    };

    TokenStream::from(output)
}

fn extract_doc_comments(attrs: &[Attribute]) -> String {
    let mut doc_comments = Vec::new();
    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let Meta::NameValue(nv) = &attr.meta {
                if let syn::Expr::Lit(expr_lit) = &nv.value {
                    if let Lit::Str(lit_str) = &expr_lit.lit {
                        doc_comments.push(lit_str.value().trim().to_string());
                    }
                }
            }
        }
    }
    doc_comments.join("\n")
}

fn is_option(ty: &Type) -> bool {
    if let Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
            return seg.ident == "Option";
        }
    }
    false
}

fn has_reference(ty: &Type) -> bool {
    match ty {
        Type::Reference(_) => true,
        Type::Path(tp) => {
            for seg in &tp.path.segments {
                if let syn::PathArguments::AngleBracketed(ab) = &seg.arguments {
                    for arg in &ab.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            if has_reference(inner) {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Macro to execute function calls requested by Gemini and update the session history.
///
/// # Usage
/// `execute_function_calls!(session, function1, function2, ...)`
///
/// # Returns
/// A `Vec<Option<Result<serde_json::Value, String>>>` containing the results of each function call.
/// The length of the vector matches the number of functions provided.
/// - `Some(Ok(value))` if the function was called and succeeded.
/// - `Some(Err(err))` if the function was called but failed.
/// - `None` if the function was not called.
///
/// # Note
/// The `session` is automatically updated with the `FunctionResponse` for successful calls.
/// If a function call fails, the error is converted to a JSON object `{"Error": error_message}`
/// and sent to the session as the function response.
#[proc_macro]
pub fn execute_function_calls(input: TokenStream) -> TokenStream {
    use syn::parse::{Parse, ParseStream};
    use syn::{Expr, Token};

    struct ExecuteInput {
        session: Expr,
        _comma: Token![,],
        functions: syn::punctuated::Punctuated<syn::Path, Token![,]>,
    }

    impl Parse for ExecuteInput {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(ExecuteInput {
                session: input.parse()?,
                _comma: input.parse()?,
                functions: input.parse_terminated(syn::Path::parse, Token![,])?,
            })
        }
    }

    let input = parse_macro_input!(input as ExecuteInput);
    let callback: Expr = syn::parse_quote! {
        |_name: String, result: Result<gemini_client_api::serde_json::Value, String>| {
            match result {
                Ok(value) => value,
                Err(e) => gemini_client_api::serde_json::json!({"Error": e}),
            }
        }
    };

    generate_execute_logic(&input.session, &callback, &input.functions)
}

fn generate_execute_logic(
    session: &syn::Expr,
    callback: &syn::Expr,
    functions: &syn::punctuated::Punctuated<syn::Path, syn::Token![,]>,
) -> TokenStream {
    let num_funcs = functions.len();

    let match_arms = functions.iter().enumerate().map(|(i, path)| {
        let name_str = path.segments.last().unwrap().ident.to_string();
        quote! {
            #name_str => {
                let args = call.args().clone().unwrap_or(gemini_client_api::serde_json::json!({}));
                let fut: gemini_client_api::futures::future::BoxFuture<'static, (usize, String, Result<gemini_client_api::serde_json::Value, String>)> = Box::pin(async move {
                    (#i, #name_str.to_string(), #path::execute(args).await)
                });
                futures.push(fut);
            }
        }
    });

    let expanded = quote! {
        {
            let mut results_array = vec![None; #num_funcs];
            // Define callback here to ensure it's available
            let mut result_callback = #callback;

            if let Some(chat) = #session.get_last_chat() {
                let mut futures = Vec::new();
                for part in chat.parts() {
                    if let gemini_client_api::gemini::types::request::PartType::FunctionCall(call) = part.data() {
                        match call.name().as_str() {
                            #(#match_arms)*
                            _ => {}
                        }
                    }
                }
                if !futures.is_empty() {
                    let results = gemini_client_api::futures::future::join_all(futures).await;
                    for (idx, name, res) in results {
                        // Invoke callback regardless of success or failure
                        let val_to_add = result_callback(name.clone(), res.clone());

                        if let Err(e) = #session.add_function_response(name.clone(), val_to_add) {
                             results_array[idx] = Some(Err(format!(
                                "failed to add function response for `{}`: {}",
                                name, e
                            )));
                            continue;
                        }
                        results_array[idx] = Some(res);
                    }
                }
            }
            results_array
        }
    };

    TokenStream::from(expanded)
}
