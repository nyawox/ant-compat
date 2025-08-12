use crate::lazy_regex;
use llm_json::{RepairOptions, loads};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::sync::LazyLock;

static BRACKET_ARGS_REGEX: LazyLock<Regex> = lazy_regex!(
    r#"(?s)(?P<key>\w+)\s*=\s*(?:"""(?P<value_multi>.*?)"""|"(?P<value_single>(?:\\.|[^"])*)")"#
);

static JSON_CODE_BLOCK_REGEX: LazyLock<Regex> =
    lazy_regex!(r"(?s)^\s*```json\s*\r?\n(.*?)\r?\n```\s*$");

static BRACKET_TOOLS_REGEX: LazyLock<Regex> = lazy_regex!(r"\[tool\((?s)(.*?)\)\]");

const XML_TOOLS_START: &str = "<function_calls>";
const XML_TOOLS_END: &str = "</function_calls>";
const BRACKET_TOOLS_START: &str = "---TOOLS---";
const BRACKET_TOOLS_END: &str = "---END_TOOLS---";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedToolCall {
    pub name: String,
    pub args: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedToolResponse {
    pub text: Option<String>,
    pub tool_calls: Option<Vec<ParsedToolCall>>,
}

#[must_use]
pub fn parse_antml(response_text: &str) -> ParsedToolResponse {
    if let Some(tool_calls_start) = response_text.find(XML_TOOLS_START) {
        parse_antml_content(response_text, tool_calls_start)
    } else {
        ParsedToolResponse {
            text: Some(response_text.to_string()),
            tool_calls: None,
        }
    }
}

fn parse_antml_content(response_text: &str, tool_calls_start: usize) -> ParsedToolResponse {
    let text = Some(response_text[..tool_calls_start].to_string());
    let end_index = response_text
        .find(XML_TOOLS_END)
        .unwrap_or(response_text.len());
    let tool_calls_content = &response_text[tool_calls_start..end_index];

    let mut reader = quick_xml::Reader::from_str(tool_calls_content);
    let mut buffer = Vec::new();
    let mut current_tool_call: Option<ParsedToolCall> = None;
    let mut current_param_name: Option<String> = None;
    let mut tool_calls = None;

    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(quick_xml::events::Event::Start(event)) => {
                if event.name().as_ref() == b"invoke" {
                    for attr in event.attributes().flatten() {
                        if attr.key.as_ref() == b"name" {
                            current_tool_call = Some(ParsedToolCall {
                                name: String::from_utf8_lossy(&attr.value).into_owned(),
                                args: Map::new(),
                            });
                        }
                    }
                } else if event.name().as_ref() == b"parameter" {
                    for attr in event.attributes().flatten() {
                        if attr.key.as_ref() == b"name" {
                            current_param_name =
                                Some(String::from_utf8_lossy(&attr.value).into_owned());
                        }
                    }
                }
            }
            Ok(quick_xml::events::Event::Text(event)) => {
                if let (Some(tool_call), Some(param_name)) =
                    (current_tool_call.as_mut(), current_param_name.as_ref())
                {
                    let value_str = String::from_utf8_lossy(event.as_ref()).into_owned();
                    let mut is_json = false;
                    let processed_value_str =
                        if let Some(captures) = JSON_CODE_BLOCK_REGEX.captures(&value_str) {
                            is_json = true;
                            captures.get(1).map_or("", |mat| mat.as_str())
                        } else {
                            value_str.trim()
                        };

                    tool_call.args.insert(
                        param_name.clone(),
                        parse_value(processed_value_str, is_json),
                    );
                }
            }
            Ok(quick_xml::events::Event::End(event)) => {
                if event.name().as_ref() == b"invoke" {
                    if let Some(tool_call) = current_tool_call.take() {
                        tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                    }
                } else if event.name().as_ref() == b"parameter" {
                    current_param_name = None;
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => {
                break;
            }
            _ => (),
        }
        buffer.clear();
    }

    ParsedToolResponse { text, tool_calls }
}

#[must_use]
pub fn parse_bracket_tools(response_text: &str) -> ParsedToolResponse {
    if let Some(start_pos) = response_text.find(BRACKET_TOOLS_START) {
        parse_bracket_tools_content(response_text, start_pos)
    } else {
        ParsedToolResponse {
            text: Some(response_text.to_string()),
            tool_calls: None,
        }
    }
}

fn parse_bracket_tools_content(response_text: &str, start_pos: usize) -> ParsedToolResponse {
    let text = Some(response_text[..start_pos].to_string());
    let tools_content = &response_text[start_pos + BRACKET_TOOLS_START.len()..];
    let end_pos = tools_content
        .find(BRACKET_TOOLS_END)
        .unwrap_or(tools_content.len());
    let tools_content = &tools_content[..end_pos].trim();

    let mut calls = Vec::new();
    for cap in BRACKET_TOOLS_REGEX.captures_iter(tools_content) {
        let matched_text = &cap[0];
        if let Some(call) = parse_single_bracket_tool_call(matched_text) {
            calls.push(call);
        }
    }

    let tool_calls = if calls.is_empty() { None } else { Some(calls) };
    ParsedToolResponse { text, tool_calls }
}

pub fn parse_single_bracket_tool_call(slice: &str) -> Option<ParsedToolCall> {
    if !slice.starts_with("[tool(") || !slice.ends_with(")]") {
        return None;
    }
    let inner = &slice["[tool(".len()..slice.len() - 2];
    let mut in_str = false;
    let mut in_triple = false;
    let mut escape = false;
    let mut split_at: Option<usize> = None;
    for (index, _) in inner.char_indices() {
        let rem = &inner[index..];
        if in_triple {
            if rem.starts_with(r#"""""#) {
                in_triple = false;
            }
            continue;
        }
        let character = rem.chars().next().unwrap_or_default();
        if in_str {
            if !escape && character == '"' {
                in_str = false;
            }
            escape = !escape && character == '\\';
            continue;
        }
        if character == '"' {
            if rem.starts_with(r#"""""#) {
                in_triple = true;
            } else {
                in_str = true;
            }
            continue;
        }
        if character == ',' {
            split_at = Some(index);
            break;
        }
    }
    let (name, args_str) = if let Some(pos) = split_at {
        (inner[..pos].trim(), inner[pos + 1..].trim())
    } else {
        (inner.trim(), "")
    };
    if name.is_empty() {
        return None;
    }
    let mut args = Map::new();
    if !args_str.is_empty() {
        for cap in BRACKET_ARGS_REGEX.captures_iter(args_str) {
            if let Some(key) = cap.name("key") {
                let key = key.as_str().to_string();
                let value_str = if let Some(value) = cap.name("value_multi") {
                    value.as_str().to_string()
                } else if let Some(value) = cap.name("value_single") {
                    value.as_str().replace(r#"\""#, r#"""#)
                } else {
                    continue;
                };
                args.insert(key, parse_value(&value_str, true));
            }
        }
    }
    Some(ParsedToolCall {
        name: name.to_string(),
        args,
    })
}

#[must_use]
pub fn parse_single_xml_tool_call(slice: &str) -> Option<ParsedToolCall> {
    let wrapped = format!("<function_calls>\n{slice}\n</function_calls>");
    let resp = parse_antml_content(&wrapped, 0);
    resp.tool_calls.and_then(|mut value| value.pop())
}

#[must_use]
pub fn build_non_stream_response(parsed: ParsedToolResponse) -> Value {
    let mut message_map = serde_json::Map::new();

    if let Some(text) = parsed.text
        && !text.is_empty()
    {
        message_map.insert("content".to_string(), Value::String(text));
    }

    if let Some(tool_calls) = parsed.tool_calls {
        message_map.insert(
            "tool_calls".to_string(),
            Value::Array(
                tool_calls
                    .into_iter()
                    .map(|tc| {
                        json!({
                            "id": format!("call_{}", rand::random::<u32>()),
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": serde_json::to_string(&tc.args).unwrap_or_default(),
                            }
                        })
                    })
                    .collect(),
            ),
        );
    }

    Value::Object(message_map)
}

fn parse_value(value_str: &str, is_json: bool) -> Value {
    if is_json {
        let wrapped_str = format!(r#"{{"data":{value_str}}}"#);
        if let Ok(parsed) = loads(&wrapped_str, &RepairOptions::default())
            && let Some(data) = parsed.get("data")
            && (!data.is_null() || value_str.trim() == "null")
        {
            return data.clone();
        }
    }
    Value::String(value_str.to_string())
}
