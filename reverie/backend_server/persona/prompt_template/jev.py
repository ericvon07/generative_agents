"""
File: jev.py
Description: Transport for TypeSafe's System One model (Jev).

One HTTP call carries one <state> and a map of typed <questions>; every
question is evaluated in parallel against that state and comes back with its
own typed answer and probabilities. Three question types exist:

  choice -> {"choice": <label>, "probabilities": {...}, "confidence": float}
  score  -> {"score": float in [0, n-1], "legend": {...}, "confidence": float}
  noul   -> {"noul": float in [0, 1]}

Why this module talks HTTP instead of importing typesafe-sdk: the official SDK
requires Python >= 3.10 and this simulator is pinned to 3.9 by the fork it came
from. The wire format is small and stable enough to call directly, and doing so
adds no dependency to a requirements.txt the project is trying to shrink. If the
simulator ever moves to 3.10+, this file is the only thing the SDK would
replace; ask() keeps the SDK's shape so the swap stays mechanical.

Configuration, in openai_config.json:

  "jev": {
      "model": "jev-latest",
      "base-url": "https://api.typesafe.ai/v1",
      "api-key-env": "TYPESAFE_API_KEY",
      "api-key": "<JEV-API-KEY>",
      "timeout": 30.0,
      "max-retries": 3
  }

The key comes from the environment variable named by "api-key-env" (default
TYPESAFE_API_KEY), and failing that from "api-key" in the config block -- the
same place openai_config.json already keeps the OpenAI key, and the same file
.gitignore already excludes. The environment wins, so one run can be pointed at
a different key without editing the file every other run shares. Environment
overrides: JEV_MODEL, JEV_BASE_URL.

Every call is written to the same llm_trace JSONL the LLM path uses, with
path="jev", so analyze_llm_trace.py reports both conditions side by side in
the same units: latency, input tokens, cost.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

from persona.prompt_template import llm_trace

_DEFAULTS = {
    "model": "jev-latest",
    "base-url": "https://api.typesafe.ai/v1",
    "api-key-env": "TYPESAFE_API_KEY",
    "api-key": "",
    "timeout": 30.0,
    "max-retries": 3,
    # $ per 1M tokens, as published for jev-1.12 in Sep 2026. Output is free.
    "costs": {"input": 0.042, "output": 0.0},
}


class JevError(Exception):
  """The request did not produce a usable answer. Raised after the retries
  are spent, so a caller can decide between the fail-safe and stopping."""


def _load_config():
  cfg = {}
  try:
    with open("../../openai_config.json", "r") as f:
      cfg = json.load(f).get("jev", {}) or {}
  except Exception:
    cfg = {}

  merged = dict(_DEFAULTS)
  merged.update({k: v for k, v in cfg.items() if v is not None})
  merged["model"] = os.environ.get("JEV_MODEL") or merged["model"]
  merged["base-url"] = (os.environ.get("JEV_BASE_URL")
                        or merged["base-url"]).rstrip("/")
  merged["timeout"] = float(merged["timeout"])
  merged["max-retries"] = int(merged["max-retries"])
  return merged


_config = _load_config()


def model():
  return _config["model"]


def costs():
  return _config["costs"]


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_env_name():
  """"api-key-env" names the variable; "api-key" holds the key. The two sit
  next to each other and the first mistake anyone makes -- the one made here
  on 22 Sep 2026 -- is putting the key in the wrong one. Say so, rather than
  look up an environment variable whose name is a secret and report that no
  key was found."""
  name = str(_config.get("api-key-env") or "")
  if name and not _ENV_NAME.match(name):
    raise JevError(
        "openai_config.json: \"jev\".\"api-key-env\" should be the NAME of an "
        "environment variable (e.g. TYPESAFE_API_KEY), not the key itself. "
        "The key goes in \"jev\".\"api-key\".")


def api_key():
  _check_env_name()
  from_env = os.environ.get(_config["api-key-env"], "").strip()
  if from_env:
    return from_env
  from_config = str(_config.get("api-key") or "").strip()
  # The example configs ship a placeholder; treat it as absent rather than
  # send it and read the 401 back as though it were an answer.
  if from_config.startswith("<") and from_config.endswith(">"):
    return ""
  return from_config


def configured():
  """Whether a key is present. Checked once at startup so a run does not get
  20 minutes in before discovering it has been measuring nothing."""
  return bool(api_key())


def config_error():
  """The configuration mistake to report, if there is one, or None."""
  try:
    _check_env_name()
  except JevError as e:
    return str(e)
  return None


def key_source():
  """Where the key came from, for the startup line. Never the key itself."""
  if os.environ.get(_config["api-key-env"], "").strip():
    return "$%s" % _config["api-key-env"]
  return "openai_config.json" if api_key() else "nowhere"


def key_env_name():
  return _config["api-key-env"]


# --- question constructors --------------------------------------------------
# Thin, so a call site reads like the primitive it is asking for.

def choice(instructions, criteria):
  """criteria: {label: description or None}. Up to 255 labels."""
  return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions, criteria):
  """criteria: an ordered list of rubric levels. The answer is a float in
  [0, len(criteria) - 1], not the level's own number."""
  return {"type": "score", "instructions": instructions, "criteria": criteria}


def noul(instructions):
  """A statement; the answer is the probability that it is true."""
  return {"type": "noul", "instructions": instructions}


# --- transport --------------------------------------------------------------

_RETRYABLE_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)


def _post(payload):
  body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
  request = urllib.request.Request(
      _config["base-url"] + "/systemone",
      data=body,
      headers={
          "Authorization": "Bearer %s" % api_key(),
          "Content-Type": "application/json",
      },
      method="POST")
  with urllib.request.urlopen(request, timeout=_config["timeout"]) as response:
    return json.loads(response.read().decode("utf-8"))


def ask(state, questions, label=None):
  """Evaluate <questions> against <state> in one request.

  ARGS:
    state: what the model may look at. A string, or any JSON-serialisable
           structure -- the simulator passes the same facts its prompt would
           have spelled out, unsummarised.
    questions: {key: question}, built with choice()/score()/noul().
    label: the decision's name, for the trace.

  RETURNS:
    {key: answer dict}, one per question, under the keys given.

  RAISES:
    JevError, once the retries are spent or the response does not parse.
  """
  return ask_full(state, questions, label=label)["answers"]


def ask_full(state, questions, label=None):
  """As ask(), but hands back the whole response -- answers, model and usage.

  Token counts are written to the trace on every call, which is where the
  simulation reads them from; this exists for the scripts that need them in
  the same breath as the answer, such as smoke_test_jev.py's cost report.
  Returning them rather than stashing them in a module global keeps this
  usable from the parallel persona loop the plan calls for."""
  if not api_key():
    raise JevError(
        "no API key: set %s in the environment" % _config["api-key-env"])

  payload = {"state": state, "model": _config["model"], "questions": questions}
  attempts = _config["max-retries"]
  last_error = None

  for attempt in range(1, attempts + 1):
    llm_trace.set_attempt(attempt)
    started = time.time()
    try:
      response = _post(payload)
      latency = time.time() - started
      usage = response.get("usage") or {}
      answers = response.get("answers")
      if not isinstance(answers, dict):
        raise JevError("response carried no answers map")
      _trace(label, payload, latency, usage, answers, None)
      return {"answers": answers,
              "usage": usage,
              "model": response.get("model"),
              "latency": latency}
    except Exception as e:
      latency = time.time() - started
      last_error = e
      _trace(label, payload, latency, None, None, e)
      if not _retryable(e) or attempt == attempts:
        break
      time.sleep(min(2.0 ** (attempt - 1), 8.0))

  raise JevError("%s: %s" % (type(last_error).__name__, last_error))


def _retryable(error):
  if isinstance(error, urllib.error.HTTPError):
    return error.code in _RETRYABLE_STATUS
  if isinstance(error, urllib.error.URLError):
    return True  # connection refused, DNS, timeout
  return isinstance(error, (OSError, ValueError))


def _trace(label, payload, latency, usage, answers, error):
  """One row per HTTP request, in the LLM path's own vocabulary so the two
  conditions land in the same table."""
  try:
    detail = error
    if isinstance(error, urllib.error.HTTPError):
      try:
        detail = "HTTP %s: %s" % (error.code, error.read().decode("utf-8")[:200])
      except Exception:
        detail = "HTTP %s" % error.code
      detail = JevError(detail)
    llm_trace.record_call(
        path="jev",
        model=_config["model"],
        prompt=json.dumps(payload.get("state"), ensure_ascii=False, default=str),
        response=None,
        latency=latency,
        error=detail,
        label=label,
        extra={
            "in_tok": None if not usage else usage.get("input_tokens"),
            "out_tok": None if not usage else usage.get("output_tokens"),
            "questions": sorted((payload.get("questions") or {}).keys()),
            "resp_head": None if answers is None else json.dumps(
                answers, ensure_ascii=False, default=str)[:400],
        })
  except Exception:
    pass


# --- answer readers ---------------------------------------------------------
# The API reference shows a noul both as {"noul": 0.999} and, in one abridged
# example, as {"value": true}. Read either rather than bet on one.

def read_choice(answer):
  """RETURNS: (label, probabilities dict, confidence)."""
  if not isinstance(answer, dict) or "choice" not in answer:
    raise JevError("expected a choice answer, got %r" % (answer,))
  return (answer["choice"],
          dict(answer.get("probabilities") or {}),
          answer.get("confidence"))


def read_score(answer):
  """RETURNS: (score as float over [0, n-1], confidence)."""
  if not isinstance(answer, dict) or "score" not in answer:
    raise JevError("expected a score answer, got %r" % (answer,))
  return float(answer["score"]), answer.get("confidence")


def read_noul(answer):
  """RETURNS: the probability that the statement is true, as a float."""
  if isinstance(answer, dict):
    if "noul" in answer:
      return float(answer["noul"])
    if "value" in answer:
      return 1.0 if answer["value"] else 0.0
  raise JevError("expected a noul answer, got %r" % (answer,))
