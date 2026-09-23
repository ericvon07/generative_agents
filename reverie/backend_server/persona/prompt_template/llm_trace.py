"""
File: llm_trace.py
Description: Per-call instrumentation for every LLM/embedding request.

Writes one JSONL record per event to <log dir>/<sim_code>_<timestamp>.jsonl.
Six event kinds:

  "manifest"-- the first line of every trace file: the seed, the commit, the
               models and the backend that produced everything below it.
               Written by run_manifest.py at boot; see there for why.
  "call"    -- one HTTP request actually sent to the model backend, with
               latency, token usage and a hash/preview of the payload.
  "reject"  -- a response that came back but was thrown away by the retry
               loop (unparseable JSON, failed validation, cleanup crash).
               Inside a decision_scope -- i.e. on one of the closed-set
               decisions -- it also carries "resp_full", the untruncated
               text, because the question "did the discarded answer already
               name a member of the offered set?" cannot be asked of a
               200-character head. Outside a decision scope the head is all
               that is kept: the long rejects are the generation ones.
  "outcome" -- the end of a logical prompt: how many attempts it took and
               whether it ended in the fail-safe value.
  "decision"-- one closed-set choice or rubric score, with the domain it was
               offered and whether the answer stayed inside it.
  "guard"   -- an out-of-domain answer that a call site's domain guard
               replaced, written after the decision it corrects.

The point is to answer, per prompt template: how many calls, how many
attempts per call, how much of the budget goes to format compliance rather
than judgement, and how latency splits between the day bootstrap and a
regular simulation step.

This module never raises into the caller: instrumentation must not be able
to break a simulation. All public functions swallow their own errors.

Configuration (all optional), in openai_config.json:

  "llm-trace": {
      "enabled": true,
      "dir": "../../logs/llm_trace",
      "head-chars": 200
  }

Environment overrides: LLM_TRACE_ENABLED=0|1, LLM_TRACE_DIR, LLM_TRACE_HEAD.
"""
import contextlib
import contextvars
import datetime
import hashlib
import json
import os
import sys
import threading
import time

# --- configuration ----------------------------------------------------------

_DEFAULT_DIR = "../../logs/llm_trace"
_DEFAULT_HEAD_CHARS = 200


def _load_config():
  cfg = {}
  try:
    with open("../../openai_config.json", "r") as f:
      cfg = json.load(f).get("llm-trace", {}) or {}
  except Exception:
    cfg = {}

  enabled = bool(cfg.get("enabled", True))
  env_enabled = os.environ.get("LLM_TRACE_ENABLED")
  if env_enabled is not None:
    enabled = env_enabled.strip().lower() not in ("0", "false", "no", "off", "")

  return {
      "enabled": enabled,
      "dir": os.environ.get("LLM_TRACE_DIR") or cfg.get("dir", _DEFAULT_DIR),
      "head_chars": int(os.environ.get("LLM_TRACE_HEAD")
                        or cfg.get("head-chars", _DEFAULT_HEAD_CHARS)),
  }


_config = _load_config()

# --- call context -----------------------------------------------------------
# Simulation-wide state (sim code, step) is process global: one ReverieServer
# per process. Per-agent state is a contextvar so it stays correct once the
# persona loop is parallelized (a fresh thread starts from the defaults, and
# an asyncio task inherits a copy).

_sim_code = "unset"
_step = None
_sim_time = None

_persona = contextvars.ContextVar("llm_trace_persona", default=None)
_phase = contextvars.ContextVar("llm_trace_phase", default=None)
_attempt = contextvars.ContextVar("llm_trace_attempt", default=None)
_watch = contextvars.ContextVar("llm_trace_decision_watch", default=None)


def set_simulation(sim_code):
  """Name the simulation being run. Rotates the trace file when it changes."""
  global _sim_code
  try:
    with _writer_lock:
      if sim_code and sim_code != _sim_code:
        _sim_code = sim_code
        _close_writer()
  except Exception:
    pass


def set_step(step, sim_time=None):
  """Record which simulation step (and in-world clock) calls belong to."""
  global _step, _sim_time
  try:
    _step = step
    if sim_time is not None:
      _sim_time = (sim_time.strftime("%Y-%m-%d %H:%M:%S")
                   if hasattr(sim_time, "strftime") else str(sim_time))
  except Exception:
    pass


@contextlib.contextmanager
def persona_scope(name):
  """Attribute the calls made inside the block to one agent."""
  token = None
  try:
    token = _persona.set(name)
  except Exception:
    pass
  try:
    yield
  finally:
    if token is not None:
      try:
        _persona.reset(token)
      except Exception:
        pass


@contextlib.contextmanager
def phase_scope(phase):
  """Label the calls made inside the block ("plan", "bootstrap", ...)."""
  token = None
  try:
    token = _phase.set(phase)
  except Exception:
    pass
  try:
    yield
  finally:
    if token is not None:
      try:
        _phase.reset(token)
      except Exception:
        pass


def strip_prompt_prefix(label):
  """An outcome is labelled by stack inspection and so carries the prompt
  builder's name; a decision carries the seam's. Same decision, two spellings."""
  label = label or ""
  prefix = "run_gpt_prompt_"
  return label[len(prefix):] if label.startswith(prefix) else label


class _DecisionWatch(object):
  """Whether the prompt resolved inside this decision ran out of retries."""

  def __init__(self, label):
    self.label = strip_prompt_prefix(label)
    self.gave_up = False

  def note(self, outcome_label):
    if strip_prompt_prefix(outcome_label) == self.label:
      self.gave_up = True


@contextlib.contextmanager
def decision_scope(label):
  """Watch one decision for a real fallback.

  Whether a decision fell back cannot be read off its answer: the retry loops
  hand back the fail-safe *value*, and for a 1-10 score or a sector named
  "kitchen" that value is also something the model can legitimately say. The
  loops do report it -- record_outcome carries failsafe=True -- so the seam
  listens for that report instead of comparing strings.

  Only an outcome under <label>'s own name counts. A decision's resolve() can
  sit inside a call that also runs generation prompts, and a daily schedule
  giving up is not this decision giving up.

  Usage:

    with llm_trace.decision_scope("action_sector") as watch:
      answer = resolve()
    record_decision(..., gave_up=watch.gave_up)
  """
  watch = _DecisionWatch(label)
  token = None
  try:
    token = _watch.set(watch)
  except Exception:
    pass
  try:
    yield watch
  finally:
    if token is not None:
      try:
        _watch.reset(token)
      except Exception:
        pass


def set_attempt(attempt):
  """Number the retry that the next request belongs to (1-based). A plain
  setter rather than a scope so the retry loops in gpt_structure.py keep
  their original indentation; record_outcome clears it."""
  try:
    _attempt.set(attempt)
  except Exception:
    pass


# --- prompt labelling -------------------------------------------------------
# The 34 prompt builders in run_gpt_prompt.py all sit in a function named
# run_gpt_prompt_*; the generate_* helpers in converse.py and the cognitive
# modules are the second choice. Walking the stack keeps run_gpt_prompt.py
# (2900 lines, ~50 call sites) untouched.

_SKIP_MODULES = ("llm_trace.py", "gpt_structure.py")
_PREFERRED_PREFIXES = ("run_gpt_prompt_", "generate_")


def _caller_label():
  try:
    frame = sys._getframe(1)
  except Exception:
    return "unknown"
  fallback = None
  while frame is not None:
    name = frame.f_code.co_name
    module = os.path.basename(frame.f_code.co_filename)
    if module not in _SKIP_MODULES:
      if name.startswith(_PREFERRED_PREFIXES):
        return name
      if fallback is None:
        fallback = "%s.%s" % (os.path.splitext(module)[0], name)
    frame = frame.f_back
  return fallback or "unknown"


# --- writer -----------------------------------------------------------------

_writer = None
_writer_lock = threading.Lock()
_writer_broken = False
_manifest = None


def set_manifest(fields):
  """What produced this run: seed, commit, models, backend. Written as the
  first line of every trace file this process opens, including each stage
  file the automatic runner rotates through, because a stage file is the
  unit someone reads back later and it has to identify itself."""
  global _manifest
  try:
    _manifest = dict(fields) if fields else None
  except Exception:
    _manifest = None


def _close_writer():
  global _writer
  if _writer is not None:
    try:
      _writer.close()
    except Exception:
      pass
    _writer = None


def _write_manifest(writer):
  """First line of a freshly opened file. Written straight to the handle
  rather than through _emit, which would re-enter the writer lock this runs
  under."""
  if _manifest is None:
    return
  try:
    record = dict(_manifest)
    record["ts"] = datetime.datetime.now().isoformat(timespec="milliseconds")
    record["sim"] = _sim_code
    record["step"] = _step
    writer.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
  except Exception:
    pass


def _get_writer():
  """Open the trace file lazily, so a run with no LLM calls leaves no file."""
  global _writer, _writer_broken
  if _writer is not None or _writer_broken:
    return _writer
  try:
    os.makedirs(_config["dir"], exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(_config["dir"], "%s_%s.jsonl" % (_sim_code, stamp))
    _writer = open(path, "a", buffering=1, encoding="utf-8")
    _write_manifest(_writer)
  except Exception as e:
    _writer_broken = True
    print("llm_trace: disabled, could not open trace file (%s)" % e)
  return _writer


def _emit(record):
  if not _config["enabled"]:
    return
  try:
    record["ts"] = datetime.datetime.now().isoformat(timespec="milliseconds")
    record["sim"] = _sim_code
    record["step"] = _step
    record["sim_time"] = _sim_time
    record["persona"] = _persona.get()
    record["phase"] = _phase.get()
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _writer_lock:
      writer = _get_writer()
      if writer is not None:
        writer.write(line + "\n")
  except Exception:
    pass


# --- payload helpers --------------------------------------------------------

def _head(text):
  if text is None:
    return None
  text = str(text)
  limit = _config["head_chars"]
  return text if len(text) <= limit else text[:limit] + "..."


def _sha1(text):
  if text is None:
    return None
  try:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]
  except Exception:
    return None


def _usage(response):
  """Pull token counts off a response, tolerating backends that omit usage."""
  usage = getattr(response, "usage", None)
  if usage is None:
    return None, None
  return (getattr(usage, "prompt_tokens", None),
          getattr(usage, "completion_tokens", None))


_last_call = contextvars.ContextVar("llm_trace_last_call", default=None)


def _response_facts(response):
  """What the response says about itself, beyond its text.

  Three facts, none of which the paper's code ever looked at, and all three
  of which a routed remote backend answers differently from a local one:

    finish_reason  why generation stopped. "length" means the provider cut
                   the answer for the token budget -- the model was
                   answering and did not get to finish, which is not the
                   same failure as answering something invalid (RB-D).
    provider       which provider actually served the request. OpenRouter
                   returns it as a top-level field; it is ground truth about
                   which weights ran, and the pin sent with the request is
                   only a request (RB-E).
    cost           what this call actually cost, at the provider that
                   actually served it, in USD. The router reports it; a
                   price table is a second copy of a number that moves
                   without telling us (RB-B).

  Every one is optional: Ollama reports none of them but finish_reason, and
  a backend that reports nothing reads None rather than zero. The OpenAI
  client keeps unknown response fields, so all three are read by attribute.
  """
  facts = {"finish_reason": None, "provider": None, "cost": None}
  if response is None:
    return facts
  try:
    choice = (getattr(response, "choices", None) or [None])[0]
    facts["finish_reason"] = getattr(choice, "finish_reason", None)
  except Exception:
    pass
  try:
    facts["provider"] = getattr(response, "provider", None)
  except Exception:
    pass
  try:
    cost = getattr(getattr(response, "usage", None), "cost", None)
    facts["cost"] = None if cost is None else float(cost)
  except Exception:
    pass
  return facts


def last_call():
  """The facts of the most recent call on this thread, or None.

  Exists because two readers need them after the response object is gone:
  the retry loops, which have to tell a truncation from a bad answer without
  being handed the response, and the smoke test, which reports what the
  router said. Set by record_call, so it is set whether or not the trace is
  switched on.
  """
  try:
    facts = _last_call.get()
    return dict(facts) if facts else None
  except Exception:
    return None


def reject_reason_for_validation():
  """Why a response that arrived was thrown away: "truncated" when the
  provider cut it for length, "validate_false" otherwise.

  The two share a symptom -- a validator saying no -- and have nothing else
  in common. SR-1 is the case in point: all 35 discarded social responses
  stop mid-clause because decide_to_talk sets max_tokens to 20, and every
  one of them was filed under validate_false, which reads as a model that
  cannot follow the format. This is the field that separates them, and it is
  read from the call rather than guessed from the text.
  """
  facts = last_call()
  if not facts:
    return "validate_false"
  if facts.get("finish_reason") == "length":
    return "truncated"
  return "validate_false"


# --- public recording API ---------------------------------------------------

def record_call(path, model, prompt, response=None, latency=None,
                error=None, label=None, extra=None):
  """One request sent to the backend. <path> is the code path taken:
  "GPT_request", "ChatGPT_request", "ChatGPT_single_request", "embedding"."""
  facts = _response_facts(response)
  try:
    _last_call.set(facts)
  except Exception:
    pass
  if not _config["enabled"]:
    return
  try:
    in_tok, out_tok = _usage(response)
    text = None
    if response is not None and path != "embedding":
      try:
        text = response.choices[0].message.content
      except Exception:
        text = None
    record = {
        "event": "call",
        "label": label or _caller_label(),
        "path": path,
        "model": model,
        "attempt": _attempt.get(),
        "latency_ms": None if latency is None else round(latency * 1000, 1),
        "in_tok": in_tok,
        "out_tok": out_tok,
        "ok": error is None,
        "error": None if error is None else "%s: %s" % (type(error).__name__, error),
        "prompt_sha1": _sha1(prompt),
        "prompt_chars": None if prompt is None else len(str(prompt)),
        "prompt_head": _head(prompt),
        "resp_head": _head(text),
    }
    if extra:
      record.update(extra)
    _emit(record)
  except Exception:
    pass


def record_reject(reason, response=None, label=None):
  """A response that arrived but was discarded by the retry loop.

  On a closed-set decision the whole text is kept, in "resp_full" beside the
  head, never instead of it: a reader written for the old shape keeps working
  and a trace from before this field stays legible. The test is the open
  decision_scope, because that contextvar is non-None for exactly the span of
  one closed-set decision -- which is the only place the extra bytes buy
  anything. What they buy: whether the answer the loop refused already named
  a member of the set it was offered, which a 200-character head cannot say
  for a response that opens with prose.
  """
  if not _config["enabled"]:
    return
  try:
    record = {
        "event": "reject",
        "label": label or _caller_label(),
        "attempt": _attempt.get(),
        "reason": reason,
        "resp_head": _head(response),
    }
    if response is not None and _in_decision():
      record["resp_full"] = str(response)
    _emit(record)
  except Exception:
    pass


def _in_decision():
  """Whether a closed-set decision is open around the caller."""
  try:
    return _watch.get() is not None
  except Exception:
    return False


def reject_reason_from_exception():
  """Classify the exception being handled, without re-raising it."""
  try:
    exc = sys.exc_info()[1]
    if exc is None:
      return "unknown"
    return "exception:%s" % type(exc).__name__
  except Exception:
    return "unknown"


def record_outcome(resolver, attempts, failsafe, ok, latency=None, label=None):
  """The end of a logical prompt: total attempts and whether it fell back."""
  if failsafe:
    # Tell the enclosing decision, if there is one, before anything else --
    # this must survive the trace being switched off, because decision.py
    # reads it to label the decision.
    try:
      watch = _watch.get()
      if watch is not None:
        watch.note(label or _caller_label())
    except Exception:
      pass
  if not _config["enabled"]:
    return
  try:
    _emit({
        "event": "outcome",
        "label": label or _caller_label(),
        "resolver": resolver,
        "attempts": attempts,
        "failsafe": failsafe,
        "ok": ok,
        "latency_ms": None if latency is None else round(latency * 1000, 1),
    })
    _attempt.set(None)
  except Exception:
    pass


def record_decision(label, domain, answer, backend=None, fail_safe=None,
                    gave_up=None, extra=None):
  """One closed-set choice and whether the answer stayed inside the set.

  This is the measurement the original code cannot make: its validation for
  sector/arena/game object accepts any non-empty string, so an answer outside
  the offered options is indistinguishable from a good one until something
  downstream raises a KeyError.

  <extra> carries what a typed backend knows and a prompt does not: the
  probability it put on every option, its confidence, and -- for a score --
  the float behind the integer the simulation stores.

  Two fields describe the fail-safe and they are not the same question:

    is_fail_safe  the answer EQUALS the fail-safe value. Cheap, and wrong as
                  a failure rate: event_poignancy falls back to 4, which is
                  also a legitimate score, and action_sector falls back to
                  "kitchen", which is a legitimate sector in some houses.
    gave_up       the decision ACTUALLY fell back -- a retry loop exhausted
                  (via decision_scope) or the typed backend errored. This is
                  the one to report. None means the caller did not say, which
                  is how traces written before 22 Sep 2026 read."""
  if not _config["enabled"]:
    return
  try:
    domain = list(domain) if domain else []
    inside = None
    if domain:
      norm = [str(d).strip().strip("{}").strip().lower() for d in domain]
      inside = str(answer).strip().strip("{}").strip().lower() in norm
    record = {
        "event": "decision",
        "label": label,
        "backend": backend,
        "domain_size": len(domain),
        "domain": domain if len(domain) <= 40 else domain[:40] + ["..."],
        "answer": _head(answer),
        "in_domain": inside,
        "is_fail_safe": None if fail_safe is None else str(answer) == str(fail_safe),
        "gave_up": None if gave_up is None else bool(gave_up),
    }
    if extra:
      record.update(extra)
    _emit(record)
  except Exception:
    pass


def record_backend_down(backend, consecutive, error):
  """The backend stopped answering and the run is giving up on it.

  Its own event kind, because it is not a property of any one decision: it
  says that every decision after it would have been a fallback. A trace that
  contains one is a trace whose tail is not a measurement.
  """
  if not _config["enabled"]:
    return
  try:
    _emit({
        "event": "backend_down",
        "backend": backend,
        "consecutive_errors": consecutive,
        "error": _head(error),
    })
  except Exception:
    pass


def record_guard(label, answer, replacement, domain=None):
  """A closed-set answer that the call site's domain guard replaced.

  This is deliberately not a field on the decision event. That event is
  written first and says what the model answered, with in_domain measured on
  it; this one says what the simulation went on to act on. Keeping them apart
  keeps the two rates apart: how often the answer left the set, and how often
  something had to be put in its place.
  """
  if not _config["enabled"]:
    return
  try:
    domain = list(domain) if domain else []
    _emit({
        "event": "guard",
        "label": label,
        "answer": _head(answer),
        "replacement": _head(replacement),
        "domain_size": len(domain),
        "domain": domain if len(domain) <= 40 else domain[:40] + ["..."],
    })
  except Exception:
    pass


def caller_label():
  """Exposed so a caller can resolve the label before the stack unwinds."""
  return _caller_label()
