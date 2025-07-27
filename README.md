# ant-compat

Built from the ground up to act as an intermediary compatible layer for individuals and organizations that utilizes internal llm gateways.
API key passthrough, no manual sonnet/opus model mapping.
Just set `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, choose the model `/model name` and start coding.


`OPENAI_BASE_URL` for setting the upstream url,
`HAIKU_MODEL` for setting the default model mapped to haiku.

I'm aware of existing options but all of them had its own quirks, such as manual model mapping, non-streaming, subtle issues with specific providers, and more.
which is an absolute dealbreaker for me.
All of this just because i can't afford to keep my private fork of cc up to date😿

## Features:
- Cleans up tool schema for gemini models
- Remove certain unnecessary system instructions
- Opionated prompt/tool instructions modifications for better performance (see src/adapters/defaults/)
- TODO: Make adapters optional, at least via env var

## TODO:
- [ ] Per-user (self service) frontend with oauth for managing adapters
- [ ] Per-user endpoint with adapter configs, let's call it "profile"
- [ ] PWA for phone, so you can easily set reasoning effort without having to prompt "think harder" or "ultrathink" every time
- [ ] "Magic words" or syntaxes, similar to "ultrathink"/"think harder" that let's you enable/disable adapters, so you can create your own "modes":
  - [ ] "Research mode", which enforces external research with parallel tool calls.
  - [ ] Investigate a tool use enforcement mode, which sets tool choice to "required", and adds a new followup question tool inspired by roocode https://github.com/RooCodeInc/Roo-Code/blob/main/src/core/prompts/tools/ask-followup-question.ts
- [ ] Detect WebSearch tool call, request searxng and append results as a tool result
- [ ] Input sanitization (Common API key patterns, ssh keys, certs etc)
- [ ] Moderation support (using OpenAI compatible moderation API or similar)
- [ ] /v1/messages/count_tokens endpoint support
- [ ] More tests, hooks, hooks, and hooks. Guardrails are the only way to make llm-assisted coding practical.
- [ ] Maybe a client companion, which auto-corrects indentation and backslash escaping, literal \n vs newline handling for lesser capable llms? (gemini really suck at this)
