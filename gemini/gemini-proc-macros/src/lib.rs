use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse_macro_input, Attribute, Data, DeriveInput, Fields, FnArg, ItemFn, Lit, Meta, Pat, Type,
};

#[proc_macro_attribute]
pub fn gemini_function(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let fn_description = extract_doc_comments(&input_fn.attrs);

    let mut properties = Vec::new();
    let mut required = Vec::new();

    for arg in input_fn.sig.inputs.iter_mut() {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = pat_ident.ident.to_string();
                let param_type = &pat_type.ty;
                let param_desc = extract_doc_comments(&pat_type.attrs);

                // Remove doc attributes from the function signature so it compiles
                pat_type.attrs.retain(|attr| !attr.path().is_ident("doc"));

                let is_optional = is_option(param_type);

                properties.push(quote! {
                    let mut schema = <#param_type as GeminiSchema>::gemini_schema();
                    if !#param_desc.is_empty() {
                        if let Some(obj) = schema.as_object_mut() {
                            obj.insert("description".to_string(), serde_json::json!(#param_desc));
                        }
                    }
                    props.insert(#param_name.to_string(), schema);
                });

                if !is_optional {
                    required.push(param_name);
                }
            }
        }
    }

    let fn_name_str = fn_name.to_string();

    let expanded = quote! {
        #input_fn

        #[allow(non_camel_case_types)]
        pub struct #fn_name { }

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
    };

    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn gemini_schema(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let name = &input.ident;
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
                impl GeminiSchema for #name {
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
                impl GeminiSchema for #name {
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
