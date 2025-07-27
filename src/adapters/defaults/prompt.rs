use crate::{adapters::traits::Adapter, conversion::request::Request};
use regex::Regex;
use std::{borrow::Cow, sync::LazyLock};

macro_rules! lazy_regex {
    ($s:expr) => {
        LazyLock::new(|| Regex::new($s).expect("Static regex pattern must be valid"))
    };
}

static DEFENSIVE_SECURITY: LazyLock<Regex> = lazy_regex!(
    r"IMPORTANT: Assist with defensive security tasks only\. Refuse to create, modify, or improve code that may be used maliciously\. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation\.\n"
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

static SHORT_RESPONSE_3: LazyLock<Regex> =
    lazy_regex!(r"(You MUST avoid text before/after your response, such as)");

static SINGLE_MESSAGE_TOOL_CALL: LazyLock<Regex> = lazy_regex!(
    r"(?s)- You have the capability to call multiple tools in a single response.*send a single message with two tool calls to run the calls in parallel\."
);

static FEEDBACK: LazyLock<Regex> = lazy_regex!(
    r"(?s)If the user asks for help or wants to give feedback inform them of the following.*cli-usage"
);

static PLAN_MODE_REMINDER: LazyLock<Regex> = lazy_regex!(
    r"(?s)<system-reminder>.*When you're done researching.*in any way until the user has confirmed the plan\..*</system-reminder>"
);

// fix gemini and o3 plan mode madness
const BETTER_PLAN_MODE_REMINDER: &str = "
<system-reminder>Plan mode is active. The user indicated that they do not want you to execute yet -- you MUST NOT make any edits, run any non-readonly tools (including changing configs or making commits), or otherwise make any changes to the system. This supercedes any other instructions you have received (for example, to make edits). Instead, you should:
1. Answer the user's query comprehensively through extensive research
2. Only when you have a complete implementation plan that requires writing code, you can call the ExitPlanMode tool to present it for user confirmation
IMPORTANT: Batch all related tool calls in single function_calls blocks - avoid sequential execution.
**NEVER** make any file changes or run any tools that modify the system state in any way.</system-reminder>";

static TASKS_SEARCH: LazyLock<Regex> = lazy_regex!(
    r"- Use the available search tools to understand the codebase and the user's query\. You are encouraged to use the search tools extensively both in parallel and sequentially\."
);

const NO_APOLOGIZE: &str = "
When encountering unexpected results or errors, state what happened and proceed without apologizing. Skip phrases like \"You're absolutely right to be frustruated\" or \"I apologize\". Just try your best to address the situation directly without apologizing.
$1";

const SEARCH_PARALLEL: &str  = "- Use search tools to understand the codebase and user query - execute LS, Read, Grep, and Glob operations in parallel.
- Prefer Grep for finding functions, symbols, or patterns. As you learn the project structure, scope searches to targeted subtrees.";

const REMOTE_TEAMMATE: &str = "IMPORTANT: Write as a collaborative remote teammate - warm and helpful while staying direct and solution-focused.";

// hotfix for lazy tool use capability
// grep tool line is inspired by zed
// ironically sonnet benefits the most from this
// qwen and kimi become very snappy
// now it can at least proactively research the web if you tell so
// instead of insisting on using outdated coding practices from their own datasets
// gemini and o3 become usable... sorta, if we ignore gemini escape sequence issue https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/utils/editCorrector.ts
const TOOL_POLICY: &str = "- When approaching any task, identify all needed operations upfront then execute them together:
  - Codebase exploration: read multiple files + search patterns + list directories in parallel
  - Research: search multiple topics + find documentation + gather examples in parallel
  - Debugging: check logs + run tests + inspect state in parallel
  - Start every research or implementation task with comprehensive parallel web searches and codebase analysis. Your knowledge cutoff makes extensive searching essential - execute multiple queries and follow relevant links across related topics.
- Always use LS tool for listing files, Read tool for viewing file contents, Grep tool for searching, and Glob tool for file pattern matching. ***NEVER*** invoke ls/cat/head/tail/grep/find as Bash commands.
- When tool calls fail, (file not found, string not found in file), use Read tool to refresh your understanding before retrying.
- Execute tool operations directly without narration - skip phrases like \"I will use MultiEdit to...\" or \"Let me apply these changes\". Only describe actions when specific permission is required.
- VERY IMPORTANT: Execute multiple independent tool calls in a single function_calls block. Avoid sequential execution.

<example>
Research task:
**WRONG** - Sequential execution (3 separate blocks):
<function_calls>
[WebSearch(\"rust Arc RwLock\")]
</function_calls>
<commentary>Wait for results, analyze, then make next call</commentary>
<function_calls>
[WebSearch(\"tokio state management\")]
</function_calls>
<commentary>Wait for results, analyze, then make next call</commentary>
<function_calls>
[WebSearch(\"axum shared state\")]
</function_calls>

**CORRECT** - Parallel execution (5 searches in 1 block):
<function_calls>
[
  WebSearch(\"rust Arc RwLock\"),
  WebSearch(\"tokio state management\"),
  WebSearch(\"axum shared state\"),
  WebSearch(\"ArcSwap benchmarks\"),
  WebSearch(\"dashmap concurrent\")
]
</function_calls>
</example>

<example>
Rust project analysis:
**WRONG** - Sequential execution (3 separate blocks):
<function_calls>
[Read(\"Cargo.toml\")]
</function_calls>
<commentary>Analyze results, then proceed</commentary>
<function_calls>
[Read(\"src/main.rs\")]
</function_calls>
<commentary>Analyze results, then proceed</commentary>
<function_calls>
[LS(\"src/\")]
</function_calls>

**CORRECT** - Parallel execution (8 operations in 1 block):
<function_calls>
[
  Read(\"Cargo.toml\"),
  Read(\"src/main.rs\"),
  Read(\"src/lib.rs\"),
  Read(\"src/error.rs\"),
  Read(\"src/config.rs\"),
  Read(\"src/handlers/mod.rs\"),
  LS(\"src/\"),
  LS(\"tests/\")
]
</function_calls>
</example>

<example>
Go project exploration:
**WRONG** - Sequential execution (3 separate blocks):
<function_calls>
[Read(\"go.mod\")]
</function_calls>
<commentary>Analyze structure, then search</commentary>
<function_calls>
[Grep(\"func main\", \"cmd/\")]
</function_calls>
<commentary>Wait for results, then read files</commentary>
<function_calls>
[Read(\"cmd/api/main.go\")]
</function_calls>

**CORRECT** - Parallel execution (7 operations in 1 block):
<function_calls>
[
  Read(\"go.mod\"),
  Read(\"cmd/api/main.go\"),
  Read(\"internal/server/server.go\"),
  Read(\"internal/database/db.go\"),
  Read(\"internal/auth/auth.go\"),
  Grep(\"func main\", \"cmd/\"),
  LS(\"internal/\")
]
</function_calls>
</example>

<example>
Git operations:
**WRONG** - Sequential execution (3 separate blocks):
<function_calls>
[Bash(\"git status\")]
</function_calls>
<commentary>Check output before proceeding</commentary>
<function_calls>
[Bash(\"git diff --cached\")]
</function_calls>
<commentary>Review changes before continuing</commentary>
<function_calls>
[Bash(\"git log --oneline -10\")]
</function_calls>

**CORRECT** - Parallel execution (5 commands in 1 block):
<function_calls>
[
  Bash(\"git status\"),
  Bash(\"git diff --cached\"),
  Bash(\"git log --oneline -10\"),
  Bash(\"git branch -a\"),
  Bash(\"git remote -v\")
]
</function_calls>
</example>

<example>
Subagent delegation:
**WRONG** - Sequential execution (2 separate blocks):
<function_calls>
[Task(description=\"Database optimization\", prompt=\"Analyze SQL queries for performance improvements\", subagent_type=\"database_expert\")]
</function_calls>
<commentary>Wait for completion before starting next task</commentary>
<function_calls>
[Task(description=\"Security review\", prompt=\"Check for XSS vulnerabilities in frontend code\", subagent_type=\"security_expert\")]
</function_calls>

**CORRECT** - Parallel execution (3 tasks in 1 block):
<function_calls>
[
  Task(description=\"Database optimization\", prompt=\"Analyze SQL queries for performance improvements\", subagent_type=\"database_expert\"),
  Task(description=\"Security review\", prompt=\"Check for XSS vulnerabilities in frontend code\", subagent_type=\"security_expert\"),
  Task(description=\"API consistency\", prompt=\"Review REST endpoints for naming conventions\", subagent_type=\"api_expert\")
]
</function_calls>
</example>";

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
                (&SHORT_RESPONSE_3, NO_APOLOGIZE),
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
