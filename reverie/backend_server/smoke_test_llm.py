"""
Smoke test for the LLM backend configured in ../../openai_config.json.

Run from reverie/backend_server (same cwd the simulation uses):

    python smoke_test_llm.py

Exercises the three code paths the simulation relies on:
  1. ChatGPT_request      -> plain chat completion
  2. GPT_request          -> chat completion with the legacy gpt_parameter dict
  3. get_embedding        -> embedding vector
and reports latency for each, so the same script doubles as a quick
"is the model responsive enough" check when switching backends.
"""
import time

from persona.prompt_template.gpt_structure import (
  ChatGPT_request, GPT_request, get_embedding, openai_config, cost_logger,
)


def timed(label, fn):
  t0 = time.perf_counter()
  out = fn()
  dt = time.perf_counter() - t0
  print(f"[{label}] {dt*1000:.0f} ms")
  return out


def main():
  print(f"model={openai_config['model']}  base_url={openai_config.get('model-base-url', 'api.openai.com')}")
  print(f"embeddings={openai_config['embeddings']}  base_url={openai_config.get('embeddings-base-url', 'api.openai.com')}")
  print()

  r1 = timed("ChatGPT_request", lambda: ChatGPT_request(
    "Answer with a single word: what color is the sky on a clear day?"))
  print("  ->", repr(r1.strip()[:80]))

  gpt_param = {"engine": "gpt-35-turbo-0125", "max_tokens": 5,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
  # Same shape run_gpt_prompt.py uses for the wake-up-hour prompt.
  r2 = timed("GPT_request", lambda: GPT_request(
    "Isabella Rodriguez usually wakes up at 6am. In one line, what hour does she wake up? Answer: ",
    gpt_param))
  print("  ->", repr(r2.strip()[:80]))

  emb = timed("get_embedding", lambda: get_embedding("Isabella Rodriguez is opening Hobbs Cafe"))
  print(f"  -> vector of dim {len(emb)}")

  print()
  print(f"cost so far: ${cost_logger._OpenAICostLogger_Singleton__cost_logger.get_current_cost():.6f}")
  ok = all([
    r1 and "ERROR" not in r1,
    r2 and "TOKEN LIMIT EXCEEDED" not in r2,
    len(emb) > 0,
  ])
  print("SMOKE TEST", "PASSED" if ok else "FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
