use crate::{
    adapters::traits::Adapter,
    conversion::request::Request,
    models::{
        claude::ClaudeTool,
        openai::{OpenAIContent, OpenAIMessage},
    },
};
use serde_json::{Map, Value, json};

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
1.  **Scalar Values**: For simple types like strings, numbers, or booleans, provide the value directly between parameter tags.
2.  **JSON Values**: For complex types like objects or arrays, place a markdown code block with a `json` tag directly between parameter tags.

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
- ALL parameter values must use quotes:
  * Strings: double quotes ("value")
  * Numbers: double quotes ("123", "42")
  * Booleans: double quotes ("true", "false")
- For multi-line content or strings with nested quotes, use triple quotes: """content"""
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

pub struct ToolSimulationRequestAdapter;

impl Adapter for ToolSimulationRequestAdapter {
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

impl ToolSimulationRequestAdapter {
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
                            .map(|(key, value)| {
                                if value.is_string() {
                                    let string_value = value.as_str().unwrap_or_default();
                                    format!("{key}=\"{string_value}\"")
                                } else {
                                    let string_value =
                                        serde_json::to_string(&value).unwrap_or_default();
                                    format!("{key}=\"\"\"{string_value}\"\"\"")
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
                format!("---TOOLS---\n{calls}\n---END_TOOLS---")
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
            format!(
                "<function>\n{}\n</function>",
                serde_json::to_string_pretty(&func).unwrap_or_default()
            )
        })
    }

    fn brackettools_system_prompt(system_prompt: &str, request: &Request) -> String {
        generate_system_prompt(
            system_prompt,
            request,
            BRACKET_TOOLS_PROMPT,
            "\n\n",
            |tool| {
                format!(
                    "#**Tool**: `{}`\n#**Description:** {}\n*#*Schema:**\n```json\n{}\n```",
                    tool.name,
                    tool.description
                        .as_deref()
                        .unwrap_or("No description provided."),
                    serde_json::to_string_pretty(&tool.input_schema).unwrap_or_default()
                )
            },
        )
    }
}

fn generate_system_prompt(
    system_prompt: &str,
    request: &Request,
    template: &str,
    separator: &str,
    tool_formatter: impl Fn(&ClaudeTool) -> String,
) -> String {
    let tools = match &request.tools {
        Some(tools) if !tools.is_empty() => tools,
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
