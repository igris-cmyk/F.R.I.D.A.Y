# F.R.I.D.A.Y. LLM Providers

FRIDAY uses a local provider router for reasoning text. Tool execution remains local and still goes through the planner, capability registry, `SecurityPolicy`, and `CapabilityExecutor`.

## Default Provider

DeepSeek is the default reasoning provider:

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_TIMEOUT_SECONDS=30
ENABLE_LOCAL_LLM=false
```

If `DEEPSEEK_API_KEY` is missing, FRIDAY returns a clean degraded message. It does not crash and does not fall back into unsafe tool execution.

## Local Ollama Path

Ollama remains available for future local-only reasoning:

```env
LLM_PROVIDER=ollama
ENABLE_LOCAL_LLM=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
```

When `ENABLE_LOCAL_LLM=false`, the provider router does not call Ollama for normal reasoning.

## Privacy Rules

Before cloud reasoning, FRIDAY scans prompts for obvious secrets and credential-heavy content. Cloud calls are blocked when the prompt appears to contain:

- `DATABASE_URL`, `AUTH_SECRET`, `NEXTAUTH_SECRET`, `GITHUB_TOKEN`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`
- OAuth client secrets, bearer tokens, passwords, private keys, or raw `.env`-style content

When blocked, the user-facing response is:

```text
This request appears to contain secrets. Cloud reasoning was not used.
```

Do not paste raw `.env` files or credentials into cloud reasoning prompts.

## Safety Boundary

DeepSeek thinks; FRIDAY acts locally. DeepSeek responses cannot execute capabilities, approve actions, weaken policy, or bypass local security.

Known deterministic commands still skip provider reasoning:

```text
show git status
read apps/desktop/package.json
analyze repository architecture
explain memory subsystem
delete everything in this folder
```

`shell.execute` remains blocked by `SecurityPolicy`.
