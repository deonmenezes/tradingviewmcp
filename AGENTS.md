# tradingviewmcp

An MCP (Model Context Protocol) bridge that connects Claude Code to TradingView Desktop via Chrome DevTools Protocol. Enables AI-assisted chart analysis, Pine Script development, and workflow automation against your locally running TradingView app. This is a fork of [tradesdontlie/tradingview-mcp](https://github.com/tradesdontlie/tradingview-mcp).

## Tech Stack

- **Language:** JavaScript (ES modules, Node.js)
- **Protocol:** MCP via `@modelcontextprotocol/sdk`
- **Browser control:** `chrome-remote-interface` (Chrome DevTools Protocol)
- **Entry point:** `src/server.js`
- **CLI entry:** `src/cli/index.js` (binary: `tv`)

## Setup

```bash
npm install

# TradingView Desktop must be launched with CDP enabled:
# Add --remote-debugging-port=9222 to TradingView's launch flags
```

Requires a valid TradingView subscription and TradingView Desktop installed.

## Build / Run / Test

```bash
# Start the MCP server
npm start
# or: node src/server.js

# Run the CLI
npm run tv
# or: node src/cli/index.js

# Run all tests
npm test

# E2E tests only
npm run test:e2e

# Unit tests only
npm run test:unit

# CLI tests
npm run test:cli

# All tests with verbose output
npm run test:all
```

## Project Structure

```
src/
  server.js          # MCP server entry point
  cli/               # CLI interface (tv command)
  core/              # Core logic, exported as ./core
  connection.js      # CDP connection management
  tools/             # MCP tool definitions
  wait.js            # Utility: wait helpers
agents/              # Agent skill definitions
skills/              # Skill workflows (chart-analysis, pine-develop, etc.)
tests/               # E2E and unit tests
scripts/             # Utility scripts
CLAUDE.md            # Claude Code specific instructions
SETUP_GUIDE.md       # Detailed setup guide
RESEARCH.md          # Research notes
```

## Architecture & Key Files

- `src/server.js` — MCP server; registers all tools and starts the protocol loop
- `src/connection.js` — manages CDP connection to TradingView Desktop on port 9222
- `src/tools/` — individual MCP tool implementations (chart reading, Pine Script, etc.)
- `src/core/` — shared logic, re-exported as the `./core` package export
- `skills/` — higher-level workflows composed from tools (chart-analysis, pine-develop, multi-symbol-scan, etc.)
- Tests use Node's built-in `--test` runner; no additional test framework

## Conventions & Notes for Agents

- ES module project (`"type": "module"`) — use `import`/`export`, not `require()`
- TradingView Desktop must be running with `--remote-debugging-port=9222` before any tool works
- This tool does NOT connect to TradingView's servers or execute real trades
- All data processing is local; no market data leaves the machine
- Upstream fork: all original design and logic is by [tradesdontlie](https://github.com/tradesdontlie/tradingview-mcp)
- Read `CLAUDE.md` for Claude Code-specific workflow notes
- CDP APIs are undocumented internals of TradingView's Electron app; they can break on TradingView updates
