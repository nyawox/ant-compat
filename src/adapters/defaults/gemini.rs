use crate::{adapters::traits::Adapter, conversion::request::Request};
use serde_json::{Map, Value};

pub struct GeminiToolSchemaAdapter;

impl Adapter for GeminiToolSchemaAdapter {
    fn adapt_tool_schema(&self, schema: &Value, _request: &Request) -> Value {
        struct Walker<'a> {
            root: &'a Value,
        }
        impl<'a> Walker<'a> {
            fn resolve_ref(&self, p: &str) -> Option<&'a Value> {
                p.strip_prefix("#/")?
                    .split('/')
                    .try_fold(self.root, |c, part| c.get(part))
            }
            fn walk(&self, node: &Value) -> Value {
                match node {
                    Value::Object(m) => {
                        let merged: Map<String, Value> = match m
                            .get("$ref")
                            .and_then(Value::as_str)
                            .and_then(|p| self.resolve_ref(p))
                            .and_then(Value::as_object)
                        {
                            Some(r) => m
                                .iter()
                                .chain(r)
                                .filter(|(k, _)| *k != "$ref")
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect(),
                            None => m.clone(),
                        };
                        let merged = match merged.get("allOf").and_then(Value::as_array) {
                            Some(all) => {
                                let props: Map<String, Value> = all
                                    .iter()
                                    .map(|it| self.walk(it))
                                    .filter_map(|w| {
                                        w.get("properties").and_then(Value::as_object).cloned()
                                    })
                                    .fold(Map::new(), |mut acc, p| {
                                        acc.extend(p);
                                        acc
                                    });
                                let mut obj = merged.clone();
                                obj.remove("allOf");
                                if !props.is_empty() {
                                    if let Some(p) = obj
                                        .entry("properties".to_string())
                                        .or_insert(Value::Object(Map::new()))
                                        .as_object_mut()
                                    {
                                        p.extend(props);
                                    }
                                    obj.entry("type".to_string())
                                        .or_insert(Value::String("object".to_string()));
                                }
                                obj
                            }
                            None => merged,
                        };
                        let mut cleaned = merged
                            .into_iter()
                            .filter(|(k, _)| {
                                !["$schema", "additionalProperties", "definitions"]
                                    .contains(&k.as_str())
                            })
                            .map(|(k, v)| (k, self.walk(&v)))
                            .collect::<Map<_, _>>();
                        if let Some(first) = cleaned
                            .get("type")
                            .and_then(Value::as_array)
                            .and_then(|arr| arr.iter().find(|v| !v.is_null()).cloned())
                        {
                            cleaned.insert("type".to_string(), first);
                        }
                        if cleaned.get("type").and_then(Value::as_str) == Some("string")
                            && cleaned
                                .get("format")
                                .and_then(Value::as_str)
                                // Remove format unless it is a date-time or enum,
                                // since Gemini does not support these formats
                                // the rest of cleanups are not really needed for cc,
                                // but they are kept to support other clients (like zed)
                                .is_some_and(|f| !["date-time", "enum"].contains(&f))
                        {
                            cleaned.remove("format");
                        }
                        Value::Object(cleaned)
                    }
                    Value::Array(a) => Value::Array(a.iter().map(|v| self.walk(v)).collect()),
                    _ => node.clone(),
                }
            }
        }
        Walker { root: schema }.walk(schema)
    }
}
