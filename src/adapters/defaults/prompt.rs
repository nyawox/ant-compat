use crate::{adapters::traits::Adapter, conversion::request::Request, lazy_regex};
use regex::Regex;
use std::{borrow::Cow, sync::LazyLock};

static DEFENSIVE_SECURITY: LazyLock<Regex> = lazy_regex!(
    r"IMPORTANT: Assist with defensive security tasks only\. Refuse to create, modify, or improve code that may be used maliciously\. .*Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation\.\n"
);

static MINIMIZE_OUTPUT: LazyLock<Regex> = lazy_regex!(
    r"IMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy\. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request\. If you can answer in 1-3 sentences or a short paragraph, please do\."
);

static SHORT_RESPONSE_CMDLINE: LazyLock<Regex> = lazy_regex!(
    r"IMPORTANT: Keep your responses short, since they will be displayed on a command line interface\."
);

static SHORT_RESPONSE_1: LazyLock<Regex> = lazy_regex!(
    r"(?s)You should be concise, direct, and to the point.*Avoid introductions, conclusions, and explanations\. "
);

static SHORT_RESPONSE_2: LazyLock<Regex> = lazy_regex!(
    r"You MUST answer concisely with fewer than 4 lines of text \(not including tool use or code generation\), unless user asks for detail\."
);

// static SHORT_RESPONSE_3: LazyLock<Regex> =
//     lazy_regex!(r"(You MUST avoid text before/after your response, such as)");

static SINGLE_MESSAGE_TOOL_CALL: LazyLock<Regex> = lazy_regex!(
    r"(?s)- You have the capability to call multiple tools in a single response.*send a single message with two tool calls to run the calls in parallel\."
);

static FEEDBACK: LazyLock<Regex> = lazy_regex!(
    r"(?s)If the user asks for help or wants to give feedback inform them of the following.*claude_code_docs_map\.md\."
);

static PLAN_MODE_REMINDER: LazyLock<Regex> = lazy_regex!(
    r"(?s)<system-reminder>.*When you're done researching.*in any way until the user has confirmed the plan\..*</system-reminder>"
);

// fix gemini and o3 plan mode madness
const BETTER_PLAN_MODE_REMINDER: &str = "
<system-reminder>Plan mode is active. The user indicated that they do not want you to execute yet -- you MUST NOT make any edits, run any non-readonly tools (including changing configs or making commits), or otherwise make any changes to the system. This supercedes any other instructions you have received (for example, to make edits). Instead, you should:
1. Answer the user's query comprehensively through extensive research
2. Only when you have a complete implementation plan that requires writing code, you can call the ExitPlanMode tool to present it for user confirmation
IMPORTANT: Batch all related tool calls in single function call block - avoid sequential execution.
**NEVER** make any file changes or run any tools that modify the system state in any way.</system-reminder>";

static TASKS_SEARCH: LazyLock<Regex> = lazy_regex!(
    r"- Use the available search tools to understand the codebase and the user's query\. You are encouraged to use the search tools extensively both in parallel and sequentially\."
);

const SEARCH_PARALLEL: &str  = "- Use search tools to understand the codebase and user query - execute LS, Read, Grep, and Glob operations in parallel.
- Prefer Grep for finding functions, symbols, or patterns. As you learn the project structure, scope searches to targeted subtrees.";

// const NO_APOLOGIZE: &str = "
// When encountering unexpected results or errors, state what happened and proceed without apologizing. Skip phrases like \"You're absolutely right to be frustruated\" or \"I apologize\". Just try your best to address the situation directly without apologizing.
// $1";

const REMOTE_TEAMMATE: &str = "IMPORTANT: Write as a collaborative remote teammate - warm and helpful while staying direct and solution-focused.";

const TOOL_POLICY: &str = "- When approaching any task, identify all needed operations upfront then execute them together:
  - Codebase exploration: read multiple files + search patterns + list directories in parallel
  - Research: search multiple topics + find documentation + gather examples in parallel
  - Debugging: check logs + run tests + inspect state in parallel
  - Start every research or implementation task with comprehensive parallel web searches and codebase analysis. Your knowledge cutoff makes extensive searching essential - execute multiple queries and follow relevant links across related topics.
- Always use Read tool for viewing file contents, Grep tool for searching, and Glob tool for file pattern matching. ***NEVER*** invoke cat/grep/find as Bash commands.
- When tool calls fail, (file not found, string not found in file), use Read tool to refresh your understanding before retrying.
- Execute tool operations directly without narration - skip phrases like \"I will use MultiEdit to...\" or \"Let me apply these changes\". Only describe actions when specific permission is required.
- IMPORTANT: When there are no dependencies between tools, always batch multiple independent operations in single blocks. Never use sequential execution for operations that can run in parallel. This includes codebase exploration, web searches, and git operations.

## Examples

Research task (parallel web searches):
<example>
[tool(WebSearch, query=\"rust Arc RwLock\")]
[tool(WebSearch, query=\"tokio state management\")]
[tool(WebSearch, query=\"axum shared state\")]
[tool(WebSearch, query=\"ArcSwap benchmarks\")]
[tool(WebSearch, query=\"dashmap concurrent\")]
</example>

Git operations (batch commands):
<example>
[tool(Bash, command=\"git status\")]
[tool(Bash, command=\"git diff --cached\")]
[tool(Bash, command=\"git log --oneline -10\")]
[tool(Bash, command=\"git branch -a\")]
[tool(Bash, command=\"git remote -v\")]
</example>

Codebase exploration (parallel reads):
<example>
[tool(Read, file_path=\"Cargo.toml\")]
[tool(Read, file_path=\"src/main.rs\")]
[tool(Read, file_path=\"src/lib.rs\")]
</example>
";

fn is_openai_model(model: &str) -> bool {
    match model {
        "o3" | "o3-mini" | "o4-mini" => true,
        _ if model.contains("gpt-") || model.contains("openai") => true,
        _ => false,
    }
}

struct Prompter<'a> {
    replacements: &'a [(&'static LazyLock<Regex>, &'static str)],
}

impl Prompter<'_> {
    fn prompt<'b>(&self, text: &'b str) -> Cow<'b, str> {
        let mut prompt = Cow::Borrowed(text);
        for (regex, replacement) in self.replacements {
            prompt = Cow::Owned(regex.replace_all(&prompt, *replacement).into_owned());
        }
        prompt
    }
}

pub struct DefaultSystemPromptAdapter;
impl Adapter for DefaultSystemPromptAdapter {
    fn adapt_system_prompt(&self, system_prompt: &str, request: &Request) -> String {
        let minimal_cleanup = Prompter {
            replacements: &[(&DEFENSIVE_SECURITY, ""), (&FEEDBACK, "")],
        };

        let mut prompt = minimal_cleanup.prompt(system_prompt).trim().to_string();

        if is_openai_model(&request.model) {
            let oai_prompter = Prompter {
                replacements: &[
                    (&MINIMIZE_OUTPUT, ""),
                    (&SHORT_RESPONSE_CMDLINE, ""),
                    (&SHORT_RESPONSE_1, REMOTE_TEAMMATE),
                    (&SHORT_RESPONSE_2, ""),
                ],
            };
            prompt = oai_prompter.prompt(&prompt).trim().to_string();
        }

        let default_improvements = Prompter {
            replacements: &[
                (&SINGLE_MESSAGE_TOOL_CALL, TOOL_POLICY),
                // (&SHORT_RESPONSE_3, NO_APOLOGIZE),
                (&TASKS_SEARCH, SEARCH_PARALLEL),
            ],
        };
        default_improvements.prompt(&prompt).trim().to_string()
    }
}

pub struct DefaultUserPromptAdapter;
impl Adapter for DefaultUserPromptAdapter {
    fn adapt_user_prompt(&self, user_prompt: &str, _request: &Request) -> String {
        let prompter = Prompter {
            replacements: &[(&PLAN_MODE_REMINDER, BETTER_PLAN_MODE_REMINDER)],
        };
        prompter.prompt(user_prompt).trim().to_string()
    }
}
