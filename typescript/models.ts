/**
 * Model Initialization File
 *
 * Configures the LLM model used throughout the course.
 *
 * Default: Anthropic claude-haiku-4-5 (fast, cheap, great for learning).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *   ⚠  IMPORTANT: install the matching package BEFORE swapping providers
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   Provider              Package                      Installed?
 *   --------------------  ---------------------------  ---------------------
 *   Anthropic (default)   @langchain/anthropic          yes (default dep)
 *   OpenAI                @langchain/openai             yes (default dep)
 *   Ollama                @langchain/ollama             yes (default dep)
 *   AWS Bedrock           @langchain/aws               install separately
 *   Google Gemini         @langchain/google-genai       install separately
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * To swap providers:
 *   1. Comment out the active model line(s) below.
 *   2. Uncomment the section for your desired provider.
 *   3. Set the provider's env vars in `.env` (see notes inline).
 */

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { ChatAnthropic } from "@langchain/anthropic";
import { config } from "dotenv";
import { Agent, setGlobalDispatcher } from "undici";

// Force `.env` to win over any same-named variable already exported by the
// shell (default dotenv behavior leaves pre-existing shell vars in place,
// which silently ignores this file's values).
config({ path: join(dirname(fileURLToPath(import.meta.url)), ".env"), override: true });

// Node's default global fetch (undici) connection pool serializes concurrent
// requests once it fills, causing multi-minute client-side queueing when
// several subagents call the same LLM endpoint concurrently (see
// STALL_FIX_REPORT.md). Widen it once here, before any model is constructed,
// so every lesson that imports this file is covered.
setGlobalDispatcher(new Agent({ connections: 64 }));

// ═══ Snowflake Cortex ════════════════════════════════════════════════════════
// Mirrors python/models.py. Claude goes through Cortex's ANTHROPIC MESSAGES API,
// not the OpenAI-compatible Chat Completions endpoint. On Chat Completions,
// Claude-on-Cortex rejects parallel tool calls and cannot do prompt caching at
// all; on the Messages API both work. See SNOWFLAKE.md at the repo root.

const SNOWFLAKE_ACCOUNT = process.env.SNOWFLAKE_ACCOUNT;
const SNOWFLAKE_PAT = process.env.SNOWFLAKE_PAT;

if (!SNOWFLAKE_ACCOUNT || !SNOWFLAKE_PAT) {
  throw new Error(
    "Set SNOWFLAKE_ACCOUNT and SNOWFLAKE_PAT in typescript/.env. " +
      "See SNOWFLAKE.md for how to mint a scoped programmatic access token.",
  );
}

// The Anthropic SDK appends /v1/messages, so this stops at /api/v2/cortex.
const CORTEX_MESSAGES_URL = `https://${SNOWFLAKE_ACCOUNT}.snowflakecomputing.com/api/v2/cortex`;

// Context/output limits as documented by SNOWFLAKE for Cortex — not Anthropic's
// own numbers, which differ (Snowflake serves claude-sonnet-5 at a 1M context).
const CORTEX_LIMITS: Record<string, { maxInput: number; maxOutput: number }> = {
  "claude-opus-5": { maxInput: 1_000_000, maxOutput: 128_000 },
  "claude-opus-4-8": { maxInput: 1_000_000, maxOutput: 128_000 },
  "claude-sonnet-5": { maxInput: 1_000_000, maxOutput: 64_000 },
  "claude-sonnet-4-6": { maxInput: 1_000_000, maxOutput: 64_000 },
  "claude-haiku-4-5": { maxInput: 200_000, maxOutput: 64_000 },
};

/**
 * Cortex rejects a TOP-LEVEL `cache_control` on the request body with
 * 400 "cache_control: Extra inputs are not permitted". LangChain's
 * prompt-caching middleware sets it there in addition to the system blocks and
 * tools; Cortex accepts those two and only objects to the top-level key.
 *
 * The JS ChatAnthropic has no payload hook, so intercept at fetch and delete
 * just that key. Bisected on a real captured payload — removing the top-level
 * key alone is sufficient, and the system/tool markers are what earn cache hits.
 */
const cortexFetch: typeof fetch = async (input, init) => {
  if (init?.body && typeof init.body === "string") {
    try {
      const payload = JSON.parse(init.body);
      if (payload && typeof payload === "object" && "cache_control" in payload) {
        delete payload.cache_control;
        init = { ...init, body: JSON.stringify(payload) };
      }
    } catch {
      // Not JSON — pass through untouched.
    }
  }
  return fetch(input as RequestInfo, init);
};

function snowflakeAnthropicModel(
  modelName: string,
  opts: { timeout?: number; maxRetries?: number } = {},
): ChatAnthropic {
  const limits = CORTEX_LIMITS[modelName] ?? { maxInput: 200_000, maxOutput: 64_000 };
  return new ChatAnthropic({
    model: modelName,
    anthropicApiUrl: CORTEX_MESSAGES_URL,
    // The SDK requires a non-empty key but Snowflake authenticates via the
    // Bearer header below, so this value is never used.
    apiKey: "unused-snowflake-uses-bearer-header",
    maxTokens: Math.min(limits.maxOutput, 8192),
    timeout: opts.timeout ?? 120_000,
    maxRetries: opts.maxRetries ?? 2,
    clientOptions: {
      defaultHeaders: { Authorization: `Bearer ${SNOWFLAKE_PAT}` },
      fetch: cortexFetch,
    },
  });
}

// ═══ Default Models ══════════════════════════════════════════════════════════
// Timeouts are above the upstream 60s/120s: upstream paired 60s with the fast
// haiku-4-5, and the Module 4/5 research labs push large payloads.
export const model = snowflakeAnthropicModel("claude-haiku-4-5", { timeout: 120_000 });

// A more capable model for steps that need stronger reasoning
export const strongModel = snowflakeAnthropicModel("claude-opus-5", { timeout: 300_000 });

// ═══ Alternative Models ══════════════════════════════════════════════════════
// export const model = snowflakeAnthropicModel("claude-sonnet-5", { timeout: 180_000 });
// export const strongModel = snowflakeAnthropicModel("claude-opus-4-8", { timeout: 300_000 });

// ═════════════════════════════════════════════════════════════════════════════
//  ORIGINAL COURSE PROVIDERS (unchanged, for reference / reverting)
// ═════════════════════════════════════════════════════════════════════════════
// Requires ANTHROPIC_API_KEY in .env and `import { initChatModel } from "langchain"`:
//
// export const model = await initChatModel("anthropic:claude-haiku-4-5", { timeout: 60_000, maxRetries: 2 });
// export const strongModel = await initChatModel("anthropic:claude-sonnet-4-6", { timeout: 120_000, maxRetries: 2 });
//
// export const model = await initChatModel("openai:gpt-4.1-mini");
//
// Ollama: run models locally (no API key required)
// Install the Ollama app first: https://ollama.com ; then: ollama pull qwen2.5:7b
// export const model = await initChatModel("ollama:qwen2.5:7b");
//
// OpenRouter: hosted open-source models via OpenAI-compatible API
// Requires OPENROUTER_API_KEY in .env
// export const model = await initChatModel("openrouter:nvidia/nemotron-3-ultra-550b-a55b:free");
