# ant-compat

A /v1/messages compatible API layer that adds cc support to existing llm gateway deployments.
built for centralized, multi-user deployments, it passes API keys directly to the backend without manual provider/model mapping.
Simply configure your environment variables `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, choose the model `/model name` and start coding.

## Environment Variables:
- `OPENAI_BASE_URL` Sets the upstream URL for the backend API
- `HAIKU_MODEL` Sets the default model to be used for background haiku requests
- `DISABLE_GROQ_MAX_TOKENS` Self explanatory, disables the groq kimi k2 workaround

## Features:
- Multi user support with API key passthrough: Forwards API keys directly, no manual provider/model mapping.
- Schema Cleanup: Cleans tool schema for compatibility with gemini models
- Instruction Cleanup: Remove certain unnecessary (and problematic) default system instructions
- Opinionated Prompt: Improves prompt and tool descriptions for better performance on less-capable models (see `src/adapters/defaults/`).
- Simulated Function Calling: Append a suffix to the model name to enable simulated function calls:
  - `-xml-tools`: a XML-based tool-calling format, similar to the native antml or roocode/cline implementations.
  - `-bracket-tools` a tool-calling format inspired by aider's NavigatorCoder PR #3781. Recommended for gemini models. It handles escaping issues very well without [client side workarounds](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/utils/editCorrector.ts). Overall, it feels more robust than the native tool_code
- TODO: Make adapters optional, at least via env var

## Goals
- [ ] Implement multi user configuration with per-user api endpoints
- [ ] Configuration interface with oauth for managing model mappings and custom adapters
- [ ] Prioritize high availability/multi instance support over all else,
- [ ] Helm chart for kubernetes deployment
- [ ] Build a mobile PWA with features like a reasoning effort toggle
## Non-goals
- [ ] Any features targetting local single user setups, (e.g. TOML-based model mapping configuration)
- [ ] Docker compose
- [ ] Server-side API key configuration
- [ ] Multi provider support or api key load balancing (use the right tool for the job)
- [ ] Cloudflare, Vercel deployment

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
