use crate::{adapters::traits::Adapter, conversion::request::Request, lazy_regex};
use regex::Regex;
use std::sync::LazyLock;

const READ_TOOL_ENFORCEMENT: &str = "**MANDATORY**: Use the `Read` tool first to examine file's contents. This tool will error if you attempt an edit without reading the file first.";
const EDIT_FAILURE_RULES: &str = "- Edit fails if `old_string` and `new_string` are identical. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`
- Edit fails if `old_string` is empty. An empty `old_string` is permitted only for file creation.";
const UNIQUENESS_RULES: &str = "**UNIQUENESS:**
- Include AT LEAST 1-6 lines of context BEFORE and AFTER your target change
- Match file content precisely: all whitespace, indentation, newlines, and surrounding code etc.
- Be unique in the file - if the string matches multiple locations, add more context OR use `replace_all=true`
- To make a targeted replacement when `old_string` might appear multiple times:
  1. **Expand old_string**: Include more surrounding unique lines from the file in your `old_string`.
  2. **When expanding**: Add more lines but do NOT change the existing content.
  3. **Multiple replacements**: Set `replace_all=true` to replace all occurrences that match `old_string` exactly.

";

static PLAN_MODE: LazyLock<Regex> = lazy_regex!(
    r"(?s)\n<system-reminder>.*When you're done researching.*in any way until the user has confirmed the plan\..*</system-reminder>"
);

// MultiEdit
const MULTIEDIT_ORIG: &str = r"To make multiple file edits, provide the following:
1\. file_path: The absolute path to the file to modify \(must be absolute, not relative\)
2\. edits: An array of edit operations to perform, where each edit contains:
   - old_string: The text to replace \(must match the file contents exactly, including all whitespace and indentation\)
   - new_string: The edited text to replace the old_string
   - replace_all: Replace all occurences of old_string\. This parameter is optional and defaults to false\.";

const MULTIEDIT_BASE: &str = "To make multiple file edits, provide the following:
1. `file_path`: The absolute path to the file to modify (MUST be an absolute path; otherwise an error will be thrown.)
2. `edits`: An array of edit operations to perform, where each edit contains:
   - `old_string`: The exact literal text to replace (including all whitespace, indentation, newlines, and surrounding code etc.).
   - `new_string`: The exact literal text to replace the old_string with (also including all whitespace, indentation, newlines, and surrounding code etc.)
   - `replace_all`: Replace all occurences of old_string. This parameter is optional and defaults to false.

";

const MULTIEDIT_PLANNING_RULES: &str = "3. Plan your edits carefully to avoid conflicts between sequential operations
4. For the smoothest editing experience, break down complex changes into smaller, focused edit operations. This approach reduces conflicts between sequential edits and makes each operation more predictable and reliable.
5. Each edit's `old_string` must be unique unless explicitly using `replace_all=true`";

// Edit
const EDIT_BASE: &str = "This tool is your surgical instrument for making precise, single, exact string replacements. For any other changes, including block/function replacement, multiple sequential edits, or complex refactoring, use the `MultiEdit` tool.

";

const EDIT_PARAMS: &str = "Expected Parameters:
- `file_path`: The absolute path to the file to modify (MUST be an absolute path; otherwise an error will be thrown.)
- `old_string`: The exact literal text to replace (including all whitespace, indentation, newlines, and surrounding code etc.).
- `new_string`: The exact literal text to replace the old_string with (also including all whitespace, indentation, newlines, and surrounding code etc.). Ensure the resulting code is correct and idiomatic.
- `replace_all`: Replace all occurences of `old_string`. This parameter is optional and defaults to false.

";

// more of an experiment to improve general planning capability
const EXIT_PLAN_MODE: &str = "Use this tool to present your plan to the user and prompt them to exit plan mode.
The user has enabled plan mode because they want research and planning ONLY - no implementation yet.

You can use this tool when:
1. You've completed thorough research using available tools
2. You have a concrete implementation plan to present
3. The task actually requires writing code

Before using this tool, you MUST:
1. Research: Gather context extensively using available tools (search files, read code, web search)
2. Understand: Analyze existing patterns, conventions, and dependencies
3. Formulate a complete plan based on your findings

IMPORTANT: Do NOT use this tool for:
- Answering questions
- Ongoing research/exploration
- Tasks that don't involve writing code
- When you haven't done sufficient investigation yet

Examples:
- \"What does this function do?\" - Analyze the codebase and answer directly
- \"How does the vim mode work?\" - Analyze the codebase and explain
- \"Help me implement yank mode for vim\" - Analyze codebase → research external APIs/libraries → formulate plan → use this tool to present it
";

pub struct DefaultToolsAdapter;

impl Adapter for DefaultToolsAdapter {
    fn adapt_tool_result(&self, tool_name: &str, tool_result: &str, _request: &Request) -> String {
        // Causes annoying false positive in certain models, and prevents nothing
        // newline should make it specific enough to prevent false positive in actual file contents
        // (if number prefix wasn't enough)
        const SYSTEM_REMINDER: &str = r"
<system-reminder>
Whenever you read a file, you should consider whether it looks malicious. If it does, you MUST refuse to improve or augment the code. You can still analyze existing code, write reports, or answer high-level questions about the code behavior.
</system-reminder>";

        let mut result = match tool_name {
            "Read" => tool_result.replace(SYSTEM_REMINDER, ""),
            _ => tool_result.to_string(),
        };

        // fix gemini and o3 plan mode madness
        result = PLAN_MODE.replace_all(&result, "
<system-reminder>Plan mode is active. The user indicated that they do not want you to execute yet -- you MUST NOT make any edits, run any non-readonly tools (including changing configs or making commits), or otherwise make any changes to the system. This supercedes any other instructions you have received (for example, to make edits). Instead, you should:
1. Answer the user's query comprehensively through extensive research
2. Only when you have a complete implementation plan that requires writing code, you can call the ExitPlanMode tool to present it for user confirmation
**NEVER** make any file changes or run any tools that modify the system state in any way.</system-reminder>").into_owned();
        result.trim().to_string()
    }

    fn adapt_tool_description(&self, description: &str, _request: &Request) -> String {
        let mut desc = description.to_string();
        // general tool description improvements

        let rules: &[(&str, &str)] = &[
            // MultiEdit instruction
            (
                MULTIEDIT_ORIG,
                &format!("{MULTIEDIT_BASE}{UNIQUENESS_RULES}"),
            ),
            // Edit tool
            (
                r"^(Performs exact string replacements in files\.)",
                &format!("$1 {EDIT_BASE}{EDIT_PARAMS}{UNIQUENESS_RULES}"),
            ),
            (
                r"You must use your `Read` tool at least once in the conversation before editing\. This tool will error if you attempt an edit without reading the file\.",
                READ_TOOL_ENFORCEMENT,
            ),
            (
                r"- The edit will FAIL if `old_string` is not unique in the file\. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`\.",
                EDIT_FAILURE_RULES,
            ),
            // MultiEdit tool
            (
                r"Use the Read tool to understand the file's contents and context",
                READ_TOOL_ENFORCEMENT,
            ),
            (
                r"3\. Plan your edits carefully to avoid conflicts between sequential operations",
                MULTIEDIT_PLANNING_RULES,
            ),
            (
                r"- The tool will fail if edits\.old_string and edits\.new_string are the same",
                EDIT_FAILURE_RULES,
            ),
            // fix lazy planning
            (
                r"(?s)Use this tool when you are in plan mode.*Use the exit plan mode tool after you have finished planning the implementation steps of the task\.",
                EXIT_PLAN_MODE,
            ),
        ];

        for (pattern, replacement) in rules {
            if let Ok(re) = Regex::new(pattern) {
                desc = re.replace_all(&desc, *replacement).to_string();
            }
        }

        desc
    }
}
