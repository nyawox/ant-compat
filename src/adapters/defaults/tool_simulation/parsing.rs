use super::streaming::{BracketGrammar, ToolGrammar, XmlGrammar};
use chumsky::prelude::*;
use llm_json::{RepairOptions, loads};
use memchr::{memchr, memmem::Finder};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

const XML_TOOLS_START: &str = "<function_calls>";
const XML_TOOLS_END: &str = "</function_calls>";
const BRACKET_TOOLS_START: &str = "---TOOLS---";
const BRACKET_TOOLS_END: &str = "---END_TOOLS---";
const INVOKE_CLOSE: &str = "</invoke>";
const PARAMETER_OPEN: &[u8] = b"<parameter ";
const PARAMETER_CLOSE: &[u8] = b"</parameter>";

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

fn some_if_not_empty<T>(items: Vec<T>) -> Option<Vec<T>> {
    if items.is_empty() { None } else { Some(items) }
}

fn strip_json_fence(input: &str) -> Option<String> {
    if input.len() >= 7 && input[..7].eq_ignore_ascii_case("```json") {
        let mut json_payload = input[7..]
            .trim_start_matches([' ', '\t', '\r', '\n'])
            .to_string();
        if let Some(pos) = json_payload.rfind("\n```") {
            json_payload.truncate(pos);
        } else if json_payload.ends_with("```") {
            json_payload.truncate(json_payload.len().saturating_sub(3));
        }
        Some(json_payload)
    } else {
        None
    }
}

fn find_quoted_attr_value<'a>(input: &'a str, attr: &str) -> Option<&'a str> {
    let key = {
        let mut attribute_key = String::with_capacity(attr.len() + 1);
        attribute_key.push_str(attr);
        attribute_key.push('=');
        attribute_key
    };
    let key_position = input.find(&key)?;
    let quote_byte = input.as_bytes().get(key_position + key.len())?;
    if *quote_byte != b'"' && *quote_byte != b'\'' {
        return None;
    }
    let value_start = key_position + key.len() + 1;
    let value_end_offset = input[value_start..].find(*quote_byte as char)?;
    Some(&input[value_start..value_start + value_end_offset])
}

struct ParamIter<'a> {
    body_bytes: &'a [u8],
    cursor: usize,
}

impl<'a> ParamIter<'a> {
    fn new(body: &'a str) -> Self {
        Self {
            body_bytes: body.as_bytes(),
            cursor: 0,
        }
    }
}

impl Iterator for ParamIter<'_> {
    type Item = (String, String);
    fn next(&mut self) -> Option<Self::Item> {
        let open_tag_finder = Finder::new(PARAMETER_OPEN);
        let open_tag_offset = open_tag_finder.find(&self.body_bytes[self.cursor..])?;
        let parameter_start = self.cursor + open_tag_offset;
        let header_start = parameter_start + PARAMETER_OPEN.len();

        let right_angle_bracket_offset = memchr(b'>', &self.body_bytes[header_start..])?;
        let header_end = header_start + right_angle_bracket_offset;
        let header = std::str::from_utf8(&self.body_bytes[parameter_start..header_end]).ok()?;
        let parameter_name = find_quoted_attr_value(header, "name")?.to_string();

        let content_start = header_end + 1;
        let close_tag_finder = Finder::new(PARAMETER_CLOSE);
        let content_end_offset = close_tag_finder.find(&self.body_bytes[content_start..])?;
        let content_end = content_start + content_end_offset;
        let parameter_value = std::str::from_utf8(&self.body_bytes[content_start..content_end])
            .ok()?
            .to_string();

        self.cursor = content_end + PARAMETER_CLOSE.len();
        Some((parameter_name, parameter_value))
    }
}

#[must_use]
pub fn parse_antml(response_text: &str) -> ParsedToolResponse {
    if let Some(tool_calls_start) = response_text.find(XML_TOOLS_START) {
        let text = Some(response_text[..tool_calls_start].to_string());
        let end_index = response_text
            .find(XML_TOOLS_END)
            .unwrap_or(response_text.len());
        let mut buffer = response_text[tool_calls_start..end_index].to_string();
        let mut calls: Vec<ParsedToolCall> = Vec::new();

        drain_complete_calls(&XmlGrammar, &mut buffer, &mut calls, parse_xml_tool);

        let tool_calls = some_if_not_empty(calls);
        ParsedToolResponse { text, tool_calls }
    } else {
        ParsedToolResponse {
            text: Some(response_text.to_string()),
            tool_calls: None,
        }
    }
}

#[must_use]
pub fn parse_bracket_tools(response_text: &str) -> ParsedToolResponse {
    if let Some(start_pos) = response_text.find(BRACKET_TOOLS_START) {
        let text = Some(response_text[..start_pos].to_string());
        let tools_content = &response_text[start_pos + BRACKET_TOOLS_START.len()..];
        let end_pos = tools_content
            .find(BRACKET_TOOLS_END)
            .unwrap_or(tools_content.len());
        let mut buffer = tools_content[..end_pos].trim().to_string();

        let mut calls = Vec::new();
        drain_complete_calls(&BracketGrammar, &mut buffer, &mut calls, parse_bracket_tool);

        let tool_calls = some_if_not_empty(calls);
        ParsedToolResponse { text, tool_calls }
    } else {
        ParsedToolResponse {
            text: Some(response_text.to_string()),
            tool_calls: None,
        }
    }
}

pub(crate) fn drain_complete_calls(
    grammar: &dyn ToolGrammar,
    buffer: &mut String,
    out: &mut Vec<ParsedToolCall>,
    parse_fn: fn(&str) -> Option<ParsedToolCall>,
) {
    let mut start_at = 0usize;
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    while let Some((start, end)) = grammar.extract_one_call_from(buffer, start_at) {
        let slice = &buffer[start..end];
        if let Some(call) = parse_fn(slice) {
            out.push(call);
        }
        ranges.push((start, end));
        start_at = end;
    }
    if ranges.is_empty() {
        return;
    }
    let mut new_buf = String::with_capacity(buffer.len());
    let mut prev = 0usize;
    for (start, end) in ranges {
        if start > prev {
            new_buf.push_str(&buffer[prev..start]);
        }
        prev = end;
    }
    if prev < buffer.len() {
        new_buf.push_str(&buffer[prev..]);
    }
    *buffer = new_buf;
}

#[must_use]
pub fn parse_bracket_tool(slice: &str) -> Option<ParsedToolCall> {
    enum ArgValue {
        Quoted(String),
        Triple(String),
        Bare,
    }

    let tool_name = any()
        .filter(|c: &char| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .repeated()
        .at_least(1)
        .collect::<String>()
        .padded();

    let backslash_then_char = just('\\').ignore_then(any());
    let quoted_char = backslash_then_char.or(any().filter(|c: &char| *c != '"' && *c != '\\'));
    let quoted = just('"')
        .ignore_then(quoted_char.repeated().collect::<String>())
        .then_ignore(just('"'))
        .map(ArgValue::Quoted);

    let triple_end = just("\"\"\"");
    let triple_char = any().and_is(triple_end.not());
    let triple = just("\"\"\"")
        .ignore_then(triple_char.repeated().collect::<String>())
        .then_ignore(just("\"\"\""))
        .map(ArgValue::Triple);

    // accept bare tokens to keep tool parsing from failing. we skip them later
    let bare = any()
        .filter(|c: &char| !c.is_whitespace() && *c != ',' && *c != ')' && *c != ']')
        .repeated()
        .at_least(1)
        .collect::<String>()
        .map(|_| ArgValue::Bare);

    let value = triple.or(quoted).or(bare).padded();

    let arg_key = text::ascii::ident::<&str, extra::Default>()
        .map(|s: &str| s.to_string())
        .padded();

    let arg_pair = arg_key.then_ignore(just('=').padded()).then(value).padded();

    let arg_pairs = arg_pair
        .separated_by(just(',').padded())
        .allow_trailing()
        .collect::<Vec<(String, ArgValue)>>();

    let parser = just("[tool(")
        .ignore_then(tool_name)
        .then(
            just(',')
                .padded()
                .ignore_then(arg_pairs)
                .or_not()
                .map(Option::unwrap_or_default),
        )
        .then_ignore(just(")]"));

    if let Ok((name, argument_pairs)) = parser.parse(slice).into_result() {
        let args = argument_pairs.into_iter().fold(
            Map::new(),
            |mut arguments_map, (argument_name, argument_value)| {
                match argument_value {
                    ArgValue::Quoted(value_text) | ArgValue::Triple(value_text) => {
                        let is_json = is_potential_json_literal(&value_text);
                        arguments_map.insert(argument_name, parse_value(&value_text, is_json));
                    }
                    ArgValue::Bare => {}
                }
                arguments_map
            },
        );
        Some(ParsedToolCall { name, args })
    } else {
        None
    }
}

#[must_use]
pub fn parse_xml_tool(slice: &str) -> Option<ParsedToolCall> {
    let invoke_start = slice.find("<invoke ")?;
    let invoke_after = &slice[invoke_start..];
    let open_rel = invoke_after.find('>')?;
    let header = &invoke_after[..open_rel];
    let name = find_quoted_attr_value(header, "name")?;
    let body_all = &invoke_after[open_rel + 1..];
    let close_rel = body_all.find(INVOKE_CLOSE)?;
    let body = &body_all[..close_rel];

    let args =
        ParamIter::new(body).fold(Map::new(), |mut args_map, (parameter_name, raw_value)| {
            let trimmed = raw_value.trim_start();
            if let Some(json_payload) = strip_json_fence(trimmed) {
                args_map.insert(parameter_name, parse_value(&json_payload, true));
            } else {
                let is_json = is_potential_json_literal(trimmed);
                args_map.insert(parameter_name, parse_value(trimmed, is_json));
            }
            args_map
        });

    Some(ParsedToolCall {
        name: name.to_string(),
        args,
    })
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
                    .map(|tool_call| {
                        json!({
                            "id": format!("call_{}", rand::random::<u32>()),
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": serde_json::to_string(&tool_call.args).unwrap_or_default(),
                            }
                        })
                    })
                    .collect(),
            ),
        );
    }

    Value::Object(message_map)
}

pub(crate) fn is_potential_json_literal(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.starts_with('{')
        || trimmed.starts_with('[')
        || trimmed.ends_with('}')
        || trimmed.ends_with(']')
    {
        return true;
    }
    matches!(trimmed, "true" | "false" | "null") || serde_json::from_str::<Value>(trimmed).is_ok()
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
