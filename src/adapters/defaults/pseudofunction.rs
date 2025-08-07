use crate::{
    adapters::traits::Adapter,
    conversion::request::Request,
    models::{
        claude::{ClaudeTool, ClaudeToolChoice},
        openai::{
            OpenAIContent, OpenAIDelta, OpenAIMessage, OpenAIStreamChoice, OpenAIStreamChunk,
            OpenAIStreamFunction, OpenAIStreamToolCall,
        },
    },
};
use async_stream::stream;
use futures_util::stream::{Stream, StreamExt};
use llm_json::{RepairOptions, loads};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::pin::Pin;

const XML_TOOLS_SUFFIX: &str = "-xml-tools";
const XML_TOOLS_START: &str = "<function_calls>";
const XML_TOOLS_END: &str = "</function_calls>";
const XML_TOOLS_PROMPT: &str = r#"In this environment you have access to a set of tools you can use to answer the user's question. You can invoke functions by writing a "<function_calls>" block like the following as part of your reply.

<function_calls>
  <invoke name="$FUNCTION_NAME">
    <parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
...
  </invoke>
  <invoke name="$ANOTHER_FUNCTION_NAME">
    <parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
...
  </invoke>
</function_calls>

You can invoke multiple functions in parallel by including multiple <invoke> blocks within the same <function_calls> wrapper.

Parameter Formatting Rules:
1.  **Scalar Values**: For simple types like strings, numbers, or booleans, provide the value directly.
2.  **JSON Values**: For complex types like objects or arrays, you MUST wrap the JSON content in a markdown code block with a `json` tag.

Example of a complex JSON parameter:
<function_calls>
  <invoke name="TodoWrite">
    <parameter name="todos">
```json
[{"id":"123","content":"A complex task","status":"pending","priority":"high"}]
```
    </parameter>
  </invoke>
</function_calls>

When you call tools, place the <function_calls> block at the end of your response. Do not generate any text after the closing </function_calls> tag.

Here are the functions available in JSONSchema format:
<functions>
{}
</functions>"#;

const BRACKET_TOOLS_SUFFIX: &str = "-bracket-tools";
const BRACKET_TOOLS_START: &str = "---TOOLS---";
const BRACKET_TOOLS_END: &str = "---END_TOOLS---";
const BRACKET_TOOLS_PROMPT: &str = r#"You have access to a set of tools to answer questions and complete tasks. Invoke them at the end of your response using this format:

---TOOLS---
[tool(ToolName, parameter="value", another_parameter="value")]
[tool(AnotherTool, param1="value1", param2="value2")]
[tool(ThirdTool, param="value")]
[tool(FourthTool, setting="value", option="value")]
---END_TOOLS---

## Rules
- Place tool calls at the very end of your response, after a '---TOOLS---' separator
- Each tool call must be on a new line
- All parameter values must use double quotes
- For multi-line content or strings with quotes, use triple quotes: """content"""
- Arrays and objects: wrap JSON payload in triple quotes: """[...]""" or """{}"""
- When possible, use parallel tool calls to perform multiple independent operations

## Examples

Reading multiple files:
---TOOLS---
[tool(Read, file_path="/home/user/project/config.yaml")]
[tool(Read, file_path="/home/user/project/data.csv")]
[tool(Read, file_path="/home/user/project/settings.ini")]
---END_TOOLS---

Writing todos with complex data:
---TOOLS---
[tool(TodoWrite, todos="""[{"id": "1", "content": "Review PR", "status": "pending"}]""")]
---END_TOOLS---

Multiple tools:
---TOOLS---
[tool(Read, file_path="/config.json")]
[tool(TodoWrite, todos="""[{"id": "2", "content": "Update config", "priority": "high"}]""")]
---END_TOOLS---

## Available Tools
{}"#;

pub struct PseudoFunctionToolAdapter;

impl Adapter for PseudoFunctionToolAdapter {
    fn adapt_tools(
        &self,
        _tools: Option<Vec<ClaudeTool>>,
        _request: &Request,
    ) -> Option<Vec<ClaudeTool>> {
        None
    }

    fn adapt_tool_choice(
        &self,
        _tool_choice: Option<ClaudeToolChoice>,
        _request: &Request,
    ) -> Option<ClaudeToolChoice> {
        None
    }
}

pub struct PseudoFunctionAdapter;

impl Adapter for PseudoFunctionAdapter {
    fn adapt_messages(
        &self,
        messages: Vec<OpenAIMessage>,
        request: &Request,
    ) -> Vec<OpenAIMessage> {
        if request.model.ends_with("-xml-tools") || request.model.ends_with("-bracket-tools") {
            Self::aggregate_tool_results(messages, request)
        } else {
            messages
        }
    }

    fn adapt_system_prompt(&self, system_prompt: &str, request: &Request) -> String {
        if request.model.ends_with("-xml-tools") {
            Self::xmltools_system_prompt(system_prompt, request)
        } else if request.model.ends_with("-bracket-tools") {
            Self::brackettools_system_prompt(system_prompt, request)
        } else {
            system_prompt.to_string()
        }
    }
}

impl PseudoFunctionAdapter {
    fn aggregate_tool_results(
        messages: Vec<OpenAIMessage>,
        request: &Request,
    ) -> Vec<OpenAIMessage> {
        messages
            .into_iter()
            .map(|message| match message.role.as_str() {
                "assistant" if message.tool_calls.is_some() => {
                    Self::convert_assistant_tool_call(message, request)
                }
                "tool" => Self::convert_tool_message(message, request),
                _ => message,
            })
            .collect()
    }

    fn convert_assistant_tool_call(mut message: OpenAIMessage, request: &Request) -> OpenAIMessage {
        if let Some(tool_calls) = message.tool_calls.take() {
            let formatted_calls = if request.model.ends_with("-xml-tools") {
                let calls = tool_calls
                    .into_iter()
                    .map(|call| {
                        let name = call.function.name;
                        let arguments = call.function.arguments;
                        format!(
                            "<invoke name=\"{name}\"><parameters>{arguments}</parameters></invoke>"
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("<function_calls>\n{calls}\n</function_calls>")
            } else {
                let calls = tool_calls
                    .into_iter()
                    .map(|call| {
                        let name = call.function.name;
                        let args = call.function.arguments;
                        let Ok(args_map): Result<Map<String, Value>, _> =
                            serde_json::from_str(&args)
                        else {
                            return format!("[tool({name})]");
                        };
                        let params = args_map
                            .into_iter()
                            .map(|(k, v)| {
                                if v.is_string() {
                                    let s = v.as_str().unwrap_or_default();
                                    format!("{k}=\"{s}\"")
                                } else {
                                    let s = serde_json::to_string(&v).unwrap_or_default();
                                    format!("{k}=\"\"\"{s}\"\"\"")
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(", ");

                        if params.is_empty() {
                            format!("[tool({name})]")
                        } else {
                            format!("[tool({name}, {params})]")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{BRACKET_TOOLS_START}\n{calls}\n{BRACKET_TOOLS_END}")
            };

            let new_content = if let Some(OpenAIContent::Text(existing)) = message.content.take() {
                if existing.trim().is_empty() {
                    formatted_calls
                } else {
                    format!("{existing}\n\n{formatted_calls}")
                }
            } else {
                formatted_calls
            };
            message.content = Some(OpenAIContent::Text(new_content));
        }
        message
    }

    fn convert_tool_message(message: OpenAIMessage, request: &Request) -> OpenAIMessage {
        let content = if let (Some(id), Some(OpenAIContent::Text(text))) =
            (message.tool_call_id, message.content)
        {
            let name = request.find_tool_name_by_id(&id).unwrap_or_default();
            if request.model.ends_with("-xml-tools") {
                format!(
                    "<function_results>\n<result name=\"{name}\">{text}</result>\n</function_results>"
                )
            } else {
                format!("[tool_result(name=\"{name}\")]\n{text}\n[/tool_result]")
            }
        } else {
            String::new()
        };

        OpenAIMessage {
            role: "user".to_string(),
            content: Some(OpenAIContent::Text(content)),
            ..Default::default()
        }
    }

    fn xmltools_system_prompt(system_prompt: &str, request: &Request) -> String {
        generate_system_prompt(system_prompt, request, XML_TOOLS_PROMPT, "\n\n", |tool| {
            let func = json!({
                "name": tool.name.clone(),
                "description": tool.description.clone().unwrap_or_default(),
                "parameters": tool.input_schema.clone(),
            });
            format!("<function>\n{}\n</function>", serde_json::to_string_pretty(&func).unwrap_or_default())
        })
    }

    fn brackettools_system_prompt(system_prompt: &str, request: &Request) -> String {
        generate_system_prompt(system_prompt, request, BRACKET_TOOLS_PROMPT, "\n\n", |tool| {
            format!(
                "#**Tool**: `{}`\n#**Description:** {}\n*#*Schema:**\n```json\n{}\n```",
                tool.name,
                tool.description.as_deref().unwrap_or("No description provided."),
                serde_json::to_string_pretty(&tool.input_schema).unwrap_or_default()
            )
        })
    }
}

pub struct PseudoFunctionModelAdapter;

impl Adapter for PseudoFunctionModelAdapter {
    fn adapt_model(&self, model: &str, _request: &Request) -> String {
        model
            .strip_suffix("-xml-tools")
            .or_else(|| model.strip_suffix("-bracket-tools"))
            .unwrap_or(model)
            .to_string()
    }
}

pub struct PseudoFunctionResponseAdapter;

impl Adapter for PseudoFunctionResponseAdapter {
    fn adapt_non_stream_response(&self, response: Value, request: &Request) -> Value {
        let mut response = response.clone();
        if let Some(text) = response["choices"][0]["message"]["content"].as_str() {
            let parsed = match request.model.as_str() {
                m if m.ends_with(XML_TOOLS_SUFFIX) => parse_antml(text),
                m if m.ends_with(BRACKET_TOOLS_SUFFIX) => parse_bracket_tools(text),
                _ => return response,
            };
            response["choices"][0]["message"] = build_non_stream_response(parsed);
        }
        response
    }
    fn adapt_chunk_stream(
        &self,
        stream: Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>>,
        request: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<OpenAIStreamChunk, reqwest::Error>> + Send>> {
        let Some(mut parser) = initialize_parser(request) else {
            return stream;
        };

        Box::pin(stream! {
            let mut chunk_stream = stream;
            let mut last_chunk_for_metadata: Option<OpenAIStreamChunk> = None;

            while let Some(chunk_result) = chunk_stream.next().await {
                let mut chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(e);
                        continue;
                    }
                };
                last_chunk_for_metadata = Some(chunk.clone());

                if let Some(choice) = chunk.choices.first_mut() {
                    if let Some(content) = choice.delta.content.take() {
                        let new_chunks = handle_delta(&content, &chunk, &mut parser);
                        for new_chunk in new_chunks {
                            yield Ok(new_chunk);
                        }
                    } else {
                        yield Ok(chunk);
                    }
                } else {
                    yield Ok(chunk);
                }
            }

            if let Some(chunk) = finalize_stream(parser, last_chunk_for_metadata) {
                yield Ok(chunk);
            }
        })
    }
}

fn initialize_parser(request: &Request) -> Option<StreamingToolParser> {
    match request.model.as_str() {
        m if m.ends_with(XML_TOOLS_SUFFIX) => {
            Some(StreamingToolParser::new(XML_TOOLS_START, XML_TOOLS_END))
        }
        m if m.ends_with(BRACKET_TOOLS_SUFFIX) => Some(StreamingToolParser::new(
            BRACKET_TOOLS_START,
            BRACKET_TOOLS_END,
        )),
        _ => None,
    }
}

fn build_tool_call_choice(tool_calls: Vec<ParsedToolCall>) -> OpenAIStreamChoice {
    OpenAIStreamChoice {
        index: 0,
        delta: OpenAIDelta {
            tool_calls: Some(
                tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(i, tc)| OpenAIStreamToolCall {
                        index: u32::try_from(i).unwrap_or(0),
                        id: Some(format!("call_{}", rand::random::<u32>())),
                        call_type: Some("function".to_string()),
                        function: Some(OpenAIStreamFunction {
                            name: Some(tc.name),
                            arguments: Some(serde_json::to_string(&tc.args).unwrap_or_default()),
                        }),
                    })
                    .collect(),
            ),
            ..Default::default()
        },
        finish_reason: Some("tool_calls".to_string()),
    }
}

fn handle_delta(
    content: &str,
    chunk: &OpenAIStreamChunk,
    parser: &mut StreamingToolParser,
) -> Vec<OpenAIStreamChunk> {
    let mut chunks_to_yield = Vec::new();
    let (text_to_yield, tool_calls) = parser.process(content);

    if !text_to_yield.is_empty() {
        let mut text_chunk = chunk.clone();
        if let Some(text_choice) = text_chunk.choices.first_mut() {
            text_choice.delta.content = Some(text_to_yield);
            text_choice.finish_reason = None;
        }
        chunks_to_yield.push(text_chunk);
    }

    if !tool_calls.is_empty() {
        let tool_choice = build_tool_call_choice(tool_calls);
        let mut tool_chunk = chunk.clone();
        tool_chunk.choices = vec![tool_choice];
        chunks_to_yield.push(tool_chunk);
    }

    chunks_to_yield
}

fn finalize_stream(
    mut parser: StreamingToolParser,
    last_chunk_opt: Option<OpenAIStreamChunk>,
) -> Option<OpenAIStreamChunk> {
    let tool_calls = parser.finalize();
    if !tool_calls.is_empty() {
        if let Some(mut last_chunk) = last_chunk_opt {
            let tool_choice = build_tool_call_choice(tool_calls);
            last_chunk.choices = vec![tool_choice];
            return Some(last_chunk);
        }
    } else if let Some(mut last_chunk) = last_chunk_opt {
        if let Some(choice) = last_chunk.choices.first_mut()
            && choice.finish_reason.is_none()
        {
            choice.finish_reason = Some("stop".to_string());
        }
        return Some(last_chunk);
    }
    None
}

enum ParserState {
    Passthrough,
    MatchingStart { matched_len: usize },
    InToolBlock { buffer: String },
    MatchingEnd { buffer: String, matched_len: usize },
}

struct StreamingToolParser {
    state: ParserState,
    start_delimiter: &'static str,
    end_delimiter: &'static str,
}

impl StreamingToolParser {
    fn new(start_delimiter: &'static str, end_delimiter: &'static str) -> Self {
        Self {
            state: ParserState::Passthrough,
            start_delimiter,
            end_delimiter,
        }
    }

    fn process(&mut self, text: &str) -> (String, Vec<ParsedToolCall>) {
        let mut text_to_yield = String::new();
        let mut tools_to_yield = Vec::new();

        for character in text.chars() {
            let current_state = std::mem::replace(&mut self.state, ParserState::Passthrough);
            self.state = match current_state {
                ParserState::Passthrough => {
                    if let Some(first_char) = self.start_delimiter.chars().next() {
                        if character == first_char {
                            ParserState::MatchingStart { matched_len: 1 }
                        } else {
                            text_to_yield.push(character);
                            ParserState::Passthrough
                        }
                    } else {
                        text_to_yield.push(character);
                        ParserState::Passthrough
                    }
                }
                ParserState::MatchingStart { matched_len } => {
                    if self.start_delimiter.chars().nth(matched_len) == Some(character) {
                        let new_len = matched_len + 1;
                        if new_len == self.start_delimiter.len() {
                            ParserState::InToolBlock {
                                buffer: String::new(),
                            }
                        } else {
                            ParserState::MatchingStart {
                                matched_len: new_len,
                            }
                        }
                    } else {
                        text_to_yield.push_str(&self.start_delimiter[..matched_len]);
                        text_to_yield.push(character);
                        ParserState::Passthrough
                    }
                }
                ParserState::InToolBlock { mut buffer } => {
                    if let Some(first_char) = self.end_delimiter.chars().next() {
                        if character == first_char {
                            ParserState::MatchingEnd {
                                buffer,
                                matched_len: 1,
                            }
                        } else {
                            buffer.push(character);
                            ParserState::InToolBlock { buffer }
                        }
                    } else {
                        buffer.push(character);
                        ParserState::InToolBlock { buffer }
                    }
                }
                ParserState::MatchingEnd {
                    mut buffer,
                    matched_len,
                } => {
                    if self.end_delimiter.chars().nth(matched_len) == Some(character) {
                        let new_len = matched_len + 1;
                        if new_len == self.end_delimiter.len() {
                            let tool_parser = if self.start_delimiter == XML_TOOLS_START {
                                parse_antml
                            } else {
                                parse_bracket_tools
                            };
                            let start = self.start_delimiter;
                            let end = self.end_delimiter;
                            let response = tool_parser(&format!("{start}{buffer}{end}"));
                            if let Some(tools) = response.tool_calls {
                                tools_to_yield.extend(tools);
                            }
                            if let Some(text) = response.text {
                                text_to_yield.push_str(&text.replace(self.start_delimiter, ""));
                            }
                            ParserState::Passthrough
                        } else {
                            ParserState::MatchingEnd {
                                buffer,
                                matched_len: new_len,
                            }
                        }
                    } else {
                        buffer.push_str(&self.end_delimiter[..matched_len]);
                        buffer.push(character);
                        ParserState::InToolBlock { buffer }
                    }
                }
            };
        }
        (text_to_yield, tools_to_yield)
    }

    fn finalize(&mut self) -> Vec<ParsedToolCall> {
        let final_buffer = match &self.state {
            ParserState::InToolBlock { buffer } | ParserState::MatchingEnd { buffer, .. } => {
                buffer.clone()
            }
            _ => String::new(),
        };

        if !final_buffer.is_empty() {
            let tool_parser = if self.start_delimiter == XML_TOOLS_START {
                parse_antml
            } else {
                parse_bracket_tools
            };
            let start = self.start_delimiter;
            let end = self.end_delimiter;
            let response = tool_parser(&format!("{start}{final_buffer}{end}"));
            return response.tool_calls.unwrap_or_default();
        }
        vec![]
    }
}

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

pub fn parse_antml(response_text: &str) -> ParsedToolResponse {
    if let Some(tool_calls_start) = response_text.find("<function_calls>") {
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
    let end_idx = response_text
        .find("</function_calls>")
        .unwrap_or(response_text.len());
    let tool_calls_content = &response_text[tool_calls_start..end_idx];

    let mut reader = quick_xml::Reader::from_str(tool_calls_content);
    let mut buf = Vec::new();
    let mut current_tool_call: Option<ParsedToolCall> = None;
    let mut current_param_name: Option<String> = None;
    let mut tool_calls = None;
    let Ok(json_regex) = regex::Regex::new(r"(?s)^\s*```json\s*\r?\n(.*?)\r?\n```\s*$") else {
        return ParsedToolResponse {
            text: Some(response_text.to_string()),
            tool_calls: None,
        };
    };

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Start(e)) => {
                if e.name().as_ref() == b"invoke" {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"name" {
                            current_tool_call = Some(ParsedToolCall {
                                name: String::from_utf8_lossy(&attr.value).into_owned(),
                                args: Map::new(),
                            });
                        }
                    }
                } else if e.name().as_ref() == b"parameter" {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"name" {
                            current_param_name =
                                Some(String::from_utf8_lossy(&attr.value).into_owned());
                        }
                    }
                }
            }
            Ok(quick_xml::events::Event::Text(e)) => {
                if let (Some(tool_call), Some(param_name)) =
                    (current_tool_call.as_mut(), current_param_name.as_ref())
                {
                    let value_str = String::from_utf8_lossy(e.as_ref()).into_owned();
                    let mut is_json = false;
                    let processed_value_str =
                        if let Some(captures) = json_regex.captures(&value_str) {
                            is_json = true;
                            captures.get(1).map_or("", |m| m.as_str())
                        } else {
                            value_str.trim()
                        };

                    tool_call.args.insert(
                        param_name.clone(),
                        parse_value(processed_value_str, is_json),
                    );
                }
            }
            Ok(quick_xml::events::Event::End(e)) => {
                if e.name().as_ref() == b"invoke" {
                    if let Some(tool_call) = current_tool_call.take() {
                        tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                    }
                } else if e.name().as_ref() == b"parameter" {
                    current_param_name = None;
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => {
                break;
            }
            _ => (),
        }
        buf.clear();
    }

    ParsedToolResponse { text, tool_calls }
}

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
    if let Ok(re) = regex::Regex::new(r"\[tool\((?s)(.*?)\)\]")
        && let Ok(args_re) = regex::Regex::new(
            r#"(?s)(?P<key>\w+)\s*=\s*(?:"""(?P<value_multi>.*?)"""|"(?P<value_single>(?:\\.|[^"])*)")"#,
        )
    {
        for cap in re.captures_iter(tools_content) {
            let inner_content = &cap[1];
            let (name, args_str) = match inner_content.split_once(',') {
                Some((n, a)) => (n.trim().to_string(), a),
                None => (inner_content.trim().to_string(), ""),
            };

            let mut args = Map::new();
            for arg_cap in args_re.captures_iter(args_str) {
                if let Some(key_match) = arg_cap.name("key") {
                    let key = key_match.as_str().to_string();
                    let value_str = if let Some(val) = arg_cap.name("value_multi") {
                        val.as_str().to_string()
                    } else if let Some(val) = arg_cap.name("value_single") {
                        val.as_str().replace(r#"\""#, r#"""#)
                    } else {
                        continue;
                    };

                    args.insert(key, parse_value(&value_str, true));
                }
            }
            calls.push(ParsedToolCall { name, args });
        }
    }

    let tool_calls = if calls.is_empty() { None } else { Some(calls) };
    ParsedToolResponse { text, tool_calls }
}

fn build_non_stream_response(parsed: ParsedToolResponse) -> Value {
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

fn generate_system_prompt(
    system_prompt: &str,
    request: &Request,
    template: &str,
    separator: &str,
    tool_formatter: impl Fn(&ClaudeTool) -> String,
) -> String {
    let tools = match &request.tools {
        Some(t) if !t.is_empty() => t,
        _ => return system_prompt.to_string(),
    };

    let tools_list = tools
        .iter()
        .map(tool_formatter)
        .collect::<Vec<String>>()
        .join(separator);
    let function_definitions = template.replace("{}", &tools_list);

    format!("{function_definitions}\n\n{system_prompt}")
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
