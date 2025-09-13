## Project Overview

This is a Rust-based API compatibility layer that translates between Anthropic's Claude API format and OpenAI-compatible endpoints. The server accepts Claude API requests and converts them to OpenAI format before forwarding to a configured backend.

## Development Commands

### Building and Running

```bash
# Note: ALWAYS use clippy for code validation instead of building the entire project

# Build the project (REQUIRED - never use cargo build)
nix build

# Run the development server
nix run
```

### Testing & Linting

```bash
# Run tests and review snapshots
cargo insta test --check

# Accept all snapshots
cargo insta accept

# Run clippy (required before commits)
nix build .#checks.x86_64-linux.clippy
```

## Code Architecture

### Core Components

- `main.rs`: Server setup with Axum HTTP framework, CORS configuration, and application state
- `lib.rs`: Exports used for tests only. Do not use this library path within the binary. unify all imports to crate::
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

### Adapter Design Philosophy

The adapter system is designed to be self-contained and robust. Each parameter-specific adapter is optional. If an adapter is not implemented for a given parameter, the system defaults to a pass-through behavior, using the value from the original request without modification.

The `adapters/defaults` module is a temporary home for specific, opinionated adapters that will eventually be replaced by a user-configurable system. It is not intended for simple pass-through logic, and new pass-through defaults **MUST NOT** be added.
