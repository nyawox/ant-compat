# ant-compat

A /v1/messages compatible API layer that adds cc support to existing llm gateway deployments.
built for centralized, multi-user deployments, it passes API keys directly to the backend without manual provider/model mapping.
Simply configure your environment variables `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, choose the model `/model name` and start coding.

## Environment Variables:

- `OPENAI_BASE_URL` Sets the upstream URL for the backend API.
- `HAIKU_MODEL` Sets the default model to be used for background haiku requests. Defaults to `openai/gpt-4.1-mini`
- `DISABLE_GROQ_MAX_TOKENS` Self explanatory, disables the groq kimi k2 workaround
- `CONNECTION_TIMEOUT` Timeout for establishing the initial TCP connection. Defaults to `10`.
- `IDLE_CONNECTION_TIMEOUT` How long an idle, keep-alive connection can remain before being closed. Defaults to `60`.
- `LISTEN` Sets the server's listening address and port. Defaults to `0.0.0.0:33332`.
- `DISABLE_DEFAULT_ADAPTERS` Disables the default prompt and tool adapters when set to `true` or `1`.

## Features:

- Multi user support with API key passthrough: Forwards API keys directly, no manual provider/model mapping.
- Schema Cleanup: Cleans tool schema for compatibility with gemini models
- Instruction Cleanup: Remove certain unnecessary (and problematic) default system instructions
- Opinionated Prompt: Improves prompt and tool descriptions for better performance on less-capable models (see `src/adapters/defaults/`).
- Simulated Function Calling: Append a suffix to the model name to enable simulated function calls:
  - `-xml-tools`: a XML-based tool-calling format, similar to the native antml or roocode/cline implementations.
  - `-bracket-tools` a tool-calling format inspired by aider's NavigatorCoder PR #3781. Recommended for gemini models. It handles escaping issues very well without [client side workarounds](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/utils/editCorrector.ts). Overall, it feels more robust than the native tool_code
- Request parameter modification via CLAUDE.md/subagent instructions: Add configuration directives directly in your CLAUDE.md to override model parameters and settings:
- /v1/responses support

Note: /compact currently doesn't support aliased (not recognized on backend) model names from directives, like gemini-bt in this example.

function calling suffix are unaffected

`~/.claude/CLAUDE.md`

```text
--- PROXY DIRECTIVE ---
{
  "global": {
    "temperature": 0.1
  },
  "rules": [
    {
      "if": { "modelContains": "gemini-bt" },
      "apply": {
        "model": "google/gemini-2.5-pro-bracket-tools",
        "max_tokens": 65536,
        "temperature": 0.7,
        "top_p": 0.95,
        "reasoning_effort": "high"
      }
    },
    {
      "if": { "modelContains": "vertex-gemini" },
      "apply": {
        "model": "vertex/gemini-2.5-pro",
        "max_tokens": 65536,
        "temperature": 0.7,
        "top_p": 0.95,
        "reasoning_effort": "high"
      }
    },
    {
      "if": { "modelContains": "gpt-5" },
      "apply": {
        "model": "openai/gpt-5",
        "responses": {
          "enable": true,
          "reasoning_summary": "detailed"
        }
      }
    },
    {
      "if": { "modelContains": "fw-glm-4.5" },
      "apply": {
        "model": "accounts/fireworks/models/glm-4p5"
      }
    }
  ]
}
--- END DIRECTIVE ---

original instructions unchanged (e.g. # Collaborative Development Framework)
```

`~/.claude/agents/example-agent.md`

```text
---
name: example-agent
description: Use this agent when edit operations fail repeatedly (usual description here)
tools: (usual tools here)
model: sonnet
color: pink
---

--- PROXY DIRECTIVE ---
{
  "global": {
    "model": "zai-org/glm-4.5",
    "max_tokens": 65536,
    "temperature": 1,
    "top_p": 1
  }
}
--- END DIRECTIVE ---

(usual instructions unchanged, You are a specialized code editor...)
```

Must be enclosed between `--- PROXY DIRECTIVE ---` and `--- END DIRECTIVE ---` delimiters.
The JSON parameters will be deserialized and applied to the request.

## Goals

- [ ] Implement multi user configuration with per-user api endpoints
- [ ] Configuration interface with oauth for managing model mappings and custom adapters
- [ ] Prioritize high availability/multi instance support over all else,
- [ ] Helm chart for kubernetes deployment
- [ ] Build a mobile PWA with features like a reasoning effort toggle
- [ ] Implement signature for proxy directives
- [ ] Experiment with various crates and have fun

## Non-goals

- [ ] Any features targetting local single user setups, (e.g. TOML-based model mapping configuration)
- [ ] Docker compose
- [ ] Server-side API key configuration
- [ ] Multi provider support or api key load balancing (use the right tool for the job)
- [ ] Cloudflare Worker, Vercel deployment

## TODO:

- [ ] "Magic words" or syntaxes, similar to "ultrathink"/"think harder" that let you enable/disable adapters, so you can create your own "modes":
  - [ ] "Research mode", which enforces external research with parallel tool calls.
  - [ ] Investigate a tool use enforcement mode, which sets tool choice to "required", and adds a new followup question tool inspired by roocode https://github.com/RooCodeInc/Roo-Code/blob/main/src/core/prompts/tools/ask-followup-question.ts
- [ ] Detect WebSearch tool call, request searxng and append results as a tool result
- [ ] Input sanitization (Common API key patterns, ssh keys, certs etc)
- [ ] Moderation support (using OpenAI compatible moderation API or similar)
- [ ] /v1/messages/count_tokens endpoint support
- [ ] More tests, hooks, hooks, and hooks. Guardrails are the only way to make llm-assisted coding practical.
- [ ] Maybe a client companion
- [ ] We currently prioritize connection stability over correctness, emitting `message_start` delta immediately instead of waiting the first streaming chunk. this lets the client wait through potential background retries. Because of this, the model name in both stream/non_stream responses uses pre `adapt_model` name. i'm not sure how we should really be handling this.
