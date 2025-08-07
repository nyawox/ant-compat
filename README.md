# ant-compat

Compatible layer for individuals and organizations that use internal llm gateways.
API key passthrough, no manual sonnet/opus model mapping.
Simply configure your environment variables `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, choose the model `/model name` and start coding.


- `OPENAI_BASE_URL` Sets the upstream URL for the backend api
- `HAIKU_MODEL` Sets the default model to be used for haiku requests

## Features:
- Multi user support with API key passthrough: Forwards API keys directly, no manual provider/model mapping.
- Schema Cleanup: Cleans tool schema for compatibility with gemini models
- Instruction Cleanup: Remove certain unnecessary (problematic) default system instructions
- Opinionated Prompt: prompt/tool description modifications for better performance in lesser capable models (check src/adapters/defaults/)
- Simulated Function Calling: Append a suffix to the model name to enable simulated function calls:
  - `-xml-tools`: a XML-based tool-calling format, similar to the native antml or roocode/cline implementations.
  - `-bracket-tools` a tool-calling format inspired by aider's NavigatorCoder PR #3781. Recommended for gemini models. It handles escaping issues very well without [client side workarounds](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/utils/editCorrector.ts). Overall, it feels more robust than the native tool_code
- TODO: Make adapters optional, at least via env var

## Goals
- [ ] Implement multi user configuration, per-user endpoints
- [ ] Configuration interface with oauth for managing model mappings and custom adapters
- [ ] Prioritize high availability/multi instance support over all else,
- [ ] Helm chart for kubernetes deployment
- [ ] Build a mobile PWA with features like reasoning effort toggle and more
## Non-goals
- [ ] Any feature targetting local single user setups, (e.g. TOML-based model mapping configuration)
- [ ] Docker compose
- [ ] Server-side API key configuration
- [ ] Multi provider support, api key load balancing (use the right tool for the job)
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
