## Project Overview

This is a Rust-based API compatibility layer that translates between Anthropic's Claude API format and OpenAI-compatible endpoints. The server accepts Claude API requests and converts them to OpenAI format before forwarding to a configured backend.

## Development Commands

### Building and Running

```bash
# Build the project (REQUIRED - never use cargo build)
nix build

# Run the development server
nix run
```

### Testing & Linting

```bash
# Run tests and review snapshots
cargo insta test

# Accept all snapshots
cargo insta accept

# Show a specific snapshot
cargo insta show

# Reject all snapshots
cargo insta reject

# Print a summary of all pending snapshots
cargo insta pending-snapshots

# Run clippy (required before commits)
nix build .#checks.x86_64-linux.clippy
```

## Code Architecture

### Core Components

- `main.rs`: Server setup with Axum HTTP framework, CORS configuration, and application state
- `http/routes.rs`: Main message handling logic with Claude-to-OpenAI conversion
- `models/`: Type definitions for both Claude and OpenAI API formats
  - `claude.rs`: Claude API request/response structures
  - `openai.rs`: OpenAI API request/response structures
  - `shared.rs`: Shared types and utilities
- `conversion/`: Request and response conversion logic
  - `request.rs`: Claude-to-OpenAI request conversion
  - `non_stream.rs`: Non-streaming response conversion
  - `stream.rs`: Streaming response conversion
- `adapters/`: Model-specific adaptations and parameter adjustments.
  - `defaults/`: Opinionated, non-configurable adapters that serve as temporary placeholders until a full user-configurable system is implemented.
- `logging.rs`: Structured logging setup

### Request Flow

1. Receive Claude API request at `/v1/messages`
2. Extract API key from `x-api-key` header
3. Convert Claude request format to OpenAI format
4. Forward to configured OpenAI-compatible endpoint
5. Convert OpenAI response back to Claude format
6. Return response to client

### Configuration

- `OPENAI_BASE_URL`: Backend endpoint URL (default: `http://127.0.0.1:10152/v1`)
- Server binds to `0.0.0.0:8080`
- Haiku model requests are mapped to `openai/gpt-4.1-mini`

### Adapter Design Philosophy

The adapter system is designed to be self-contained and robust. Each parameter-specific adapter is optional. If an adapter is not implemented for a given parameter, the system defaults to a pass-through behavior, using the value from the original request without modification.

The `adapters/defaults` module is a temporary home for specific, opinionated adapters that will eventually be replaced by a user-configurable system. It is not intended for simple pass-through logic, and new pass-through defaults **MUST NOT** be added.

## Project Standards

- Inline variables in format macros:
  ```rust
  debug!("Event: {event:?}")
  ```
- Use positional arguments only for data manipulation:
  ```rust
  debug!("Count: {}", items.len())
  ```
- **NO** single function exceeding 50 lines.
- **NO** single file exceeding 1000 lines.
- **NO** redundant prefixes in variable names (e.g., `currentHandleIndex` should be `handleIndex`).
- **Type Safety**: Use strong typing systems.
- **Maintainability**: Write modular, scalable, clear code.
- **DRY**: Use symbolic reasoning to identify and remove redundancy.
- **Clean Implementation**: Replace, don't extend. When improving code, completely replace the old implementation.
- **NEVER** edit `Cargo.toml` manually; always use `cargo add`.
- Run clippy via `nix build -L .#checks.x86_64-linux.clippy`.
- **NEVER** use `#[allow(clippy::...)]` under any circumstances.
- **NEVER** use `panic!()`, `unwrap()`, or `expect()` (enforced by clippy).
- Use proper error handling with `Result<T, E>` and the `?` operator.

## Key Implementation Details

### Request & Response Conversion
- Adapters: A modular adapter system allows for targeted modification of requests. Adapters can modify:
  - Request parameters like `temperature`, `top_p`, `max_tokens`.
  - System and user prompts to optimize model behavior.
  - Tool schemas and tool results.
- Message Content:
  - Claude system prompts are converted to OpenAI system messages.
  - Claude content blocks (text, image) are converted to OpenAI content parts.
  - Base64 image data is formatted as `data:{media_type};base64,{data}`.
- Tool Handling:
  - `tool_use` and `tool_result` blocks are converted to OpenAI `tool_calls` and `tool` messages.
  - `tool_choice` is converted from Claude's format to OpenAI's.
- Reasoning Effort: The `thinking` parameter in Claude requests is mapped to `reasoning_effort` for supported OpenAI backends.

### Model-Specific Adaptations
- Gemini Schema Cleaning: For Gemini models, tool schemas undergo a cleaning process that:
  - Resolves `$ref` and `allOf` directives to create a flattened schema.
  - Removes unsupported fields like `$schema`, `additionalProperties`, and `definitions`.
  - Strips unsupported `format` values from string properties.
- Prompt Cleaning: System prompts are automatically cleaned of instructions that are not relevant to the backend model's expected behavior.
