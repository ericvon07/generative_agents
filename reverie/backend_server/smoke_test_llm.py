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

Against a routed remote backend it reports four more things, and they are
the reason 006-IP extended it: which provider actually answered, what the
call cost according to the router rather than according to a price table,
why generation stopped, and what the account's own /key and /credits
endpoints said immediately before and immediately after a paid call. That
last pair is the only way to answer the question 006-IP could not answer in
advance -- the account's balance reads 0 and paid calls succeed anyway, and
before a four-hour window is launched on that assumption it is worth
spending a thousandth of a cent to watch a number move.

This is the only thing that runs against a metered backend before a real
run does.
"""
import json
import time
import urllib.error
import urllib.request

from persona.prompt_template.gpt_structure import (
  ChatGPT_request, GPT_request, get_embedding, openai_config, cost_logger,
  provider_pin,
)
from persona.prompt_template import env_file, llm_trace


def timed(label, fn):
  t0 = time.perf_counter()
  out = fn()
  dt = time.perf_counter() - t0
  print(f"[{label}] {dt*1000:.0f} ms")
  return out


def router_base():
  """The router's API root, or None when this run is not on the router.

  The /key and /credits endpoints are OpenRouter's, not OpenAI's, so they
  are asked for only when the configured base_url is the router's.
  """
  base = (openai_config.get("model-base-url") or "").rstrip("/")
  return base if "openrouter.ai" in base else None


def account(base, label):
  """What the account says about itself: the allowance and the balance.

  Printed before and after a paid call so the reader can see which of the
  two moved. Never prints the key.
  """
  from persona.prompt_template.gpt_structure import _resolve_key
  try:
    key = _resolve_key(openai_config, "model-key", "model-key-env")
  except Exception as e:
    print(f"  {label}: no key ({e})")
    return
  for endpoint in ("key", "credits"):
    req = urllib.request.Request(f"{base}/{endpoint}",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
      with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r).get("data", {})
      print(f"  {label} /{endpoint}: "
            + ", ".join(f"{k}={v}" for k, v in sorted(data.items())
                        if k != "label"))
    except urllib.error.HTTPError as e:
      print(f"  {label} /{endpoint}: HTTP {e.code}")
    except Exception as e:
      print(f"  {label} /{endpoint}: {type(e).__name__}: {e}")


def facts(label):
  """What the last call's response said about itself."""
  f = llm_trace.last_call() or {}
  cost = f.get("cost")
  print(f"  {label}: provider={f.get('provider')}  "
        f"finish_reason={f.get('finish_reason')}  "
        f"cost={'-' if cost is None else f'${cost:.8f}'}")
  return f


def main():
  print(f"model={openai_config['model']}  base_url={openai_config.get('model-base-url', 'api.openai.com')}")
  print(f"embeddings={openai_config['embeddings']}  base_url={openai_config.get('embeddings-base-url', 'api.openai.com')}")
  key_env = openai_config.get("model-key-env")
  print(f"key from: {'$' + key_env if key_env else 'openai_config.json'}"
        f"   .env read: {', '.join(env_file.loaded()) or 'none'}")
  print(f"provider pin: {json.dumps(provider_pin()) if provider_pin() else 'none'}")
  print()

  base = router_base()
  if base:
    print("ACCOUNT")
    account(base, "before")
    print()

  r1 = timed("ChatGPT_request", lambda: ChatGPT_request(
    "Answer with a single word: what color is the sky on a clear day?"))
  print("  ->", repr(r1.strip()[:80]))
  f1 = facts("ChatGPT_request")

  gpt_param = {"engine": "gpt-35-turbo-0125", "max_tokens": 5,
               "temperature": 0, "top_p": 1, "stream": False,
               "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
  # Same shape run_gpt_prompt.py uses for the wake-up-hour prompt -- and with
  # 5 tokens it is also the shape that gets cut off, so finish_reason here is
  # expected to read "length" on a model that does not answer tersely.
  r2 = timed("GPT_request", lambda: GPT_request(
    "Isabella Rodriguez usually wakes up at 6am. In one line, what hour does she wake up? Answer: ",
    gpt_param))
  print("  ->", repr(r2.strip()[:80]))
  f2 = facts("GPT_request")

  emb = timed("get_embedding", lambda: get_embedding("Isabella Rodriguez is opening Hobbs Cafe"))
  print(f"  -> vector of dim {len(emb)}")

  print()
  if base:
    print("ACCOUNT")
    account(base, "after ")
    print()

  reported = [f.get("cost") for f in (f1, f2) if f.get("cost") is not None]
  if reported:
    print(f"router reported: ${sum(reported):.8f} over {len(reported)} calls")
  print(f"cost so far: ${cost_logger._OpenAICostLogger_Singleton__cost_logger.get_current_cost():.8f}")
  ok = all([r1, r2, len(emb) > 0])
  print("SMOKE TEST", "PASSED" if ok else "FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
