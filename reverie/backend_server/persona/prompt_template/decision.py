"""
File: decision.py
Description: The seam where a closed-set choice or a rubric score is resolved.

Every choice in this simulation already knows its own domain. The sectors a
persona may walk to are a Python list before the prompt is built; so are the
arenas inside a sector, the objects inside an arena, and "yes"/"no". The
prompt builders flatten that list into a comma-separated string, and the
answer comes back as free text -- so the domain is lost on the way out and
never recovered on the way back.

What that costs is measurable. run_gpt_prompt_action_sector's validation
accepts any non-empty string containing "}"; on 22 Sep 2026 the model answered
"kitchen" for an apartment whose arenas are {main room, bathroom}, copying the
answer from the prompt's own few-shot example. Validation passed it, the maze
raised KeyError, and the run died. Note that run_gpt_prompt_decide_to_talk,
whose domain is two words, *does* check membership: the original authors
checked the domain where it was short enough to write out by hand.

choose() keeps the domain. It takes the option list and the function that
resolves it, records what came back and whether it was inside the domain, and
returns the answer. score() does the same for the 1-10 poignancy rubric.

Two backends resolve them, selected by one flag:

  "llm"  (default) runs exactly the prompt the paper runs and returns exactly
         what that prompt returned -- an out-of-domain answer included.
         Nothing about the simulation changes; the out-of-domain rate simply
         stops being invisible.

  "jev"  asks the same question as a typed question against TypeSafe's System
         One model. The answer cannot leave the domain by construction, and it
         arrives with a probability for every option. No prompt is touched:
         the wording lives in jev_questions.py, next to the template it
         mirrors, and the two backends are called from the same line.

The comparison the project is after is this flag, held against the same
checkpoint, read out of the same trace file.

Configuration, in openai_config.json:

  "decision-backend": "llm",
  "decision": {
      "noul-threshold": 0.5,
      "on-error": "failsafe"
  }

  noul-threshold  Above this probability a Noul reads as "yes". 0.5 is the
                  coin-flip boundary and keeps the Jev condition comparable to
                  a yes/no LLM answer; the point of raising it is the gating
                  the project's plan describes, which is a separate
                  experiment and must be registered as one.
  on-error        What to do when the API call fails after its retries:
                  "failsafe" takes the value the paper's own code falls back
                  to and marks the decision in the trace, so a run survives a
                  network blip; "raise" stops the simulation.

Environment overrides: DECISION_BACKEND, DECISION_NOUL_THRESHOLD,
DECISION_ON_ERROR.
"""
import json
import os

from persona.prompt_template import jev
from persona.prompt_template import jev_questions
from persona.prompt_template import llm_trace

_DEFAULT_BACKEND = "llm"
_DEFAULTS = {
    "noul-threshold": 0.5,
    "on-error": "failsafe",
}

_QUESTION_KEY = "q"  # the questions map needs a key; one question, one key


def _load_config():
  backend = _DEFAULT_BACKEND
  options = dict(_DEFAULTS)
  try:
    with open("../../openai_config.json", "r") as f:
      cfg = json.load(f)
    backend = cfg.get("decision-backend", _DEFAULT_BACKEND)
    options.update({k: v for k, v in (cfg.get("decision") or {}).items()
                    if v is not None})
  except Exception:
    pass

  backend = os.environ.get("DECISION_BACKEND") or backend
  options["noul-threshold"] = float(
      os.environ.get("DECISION_NOUL_THRESHOLD") or options["noul-threshold"])
  options["on-error"] = (os.environ.get("DECISION_ON_ERROR")
                         or options["on-error"])
  return backend, options


_backend, _options = _load_config()


def backend():
  """Which implementation resolves decisions in this run."""
  return _backend


def options():
  return dict(_options)


def check_ready():
  """Called once at startup. Returns a human-readable line about how
  decisions will be resolved, and raises before a run begins rather than
  after twenty minutes of it if the chosen backend cannot work."""
  if _backend == "llm":
    return "decisions: llm (the paper's prompts)"
  if _backend == "jev":
    misconfigured = jev.config_error()
    if misconfigured:
      raise RuntimeError(misconfigured)
    if not jev.configured():
      raise RuntimeError(
          "decision-backend is 'jev' but no API key was found: set $%s, or "
          "\"jev\": {\"api-key\": ...} in openai_config.json"
          % jev.key_env_name())
    return ("decisions: jev (%s, key from %s, noul threshold %.2f, "
            "on error: %s)" % (jev.model(), jev.key_source(),
                               _options["noul-threshold"],
                               _options["on-error"]))
  raise ValueError("unknown decision backend: %s" % _backend)


def normalize(value):
  """The comparison used for domain membership: the prompts and the maze
  differ in whitespace and case far more often than in meaning."""
  return str(value).strip().strip("{}").strip().lower()


def in_domain(answer, domain):
  if not domain:
    return None  # no declared domain: nothing to check against
  return normalize(answer) in [normalize(d) for d in domain]


# --- closed-set choice ------------------------------------------------------

def choose(label, domain, resolve, fail_safe=None, state=None, criteria=None):
  """Resolve one closed-set choice.

  ARGS:
    label: the decision's name, as it should appear in the trace
           (e.g. "action_sector"). jev_questions.py is keyed by it.
    domain: the closed set of valid answers, as a list.
    resolve: a zero-argument callable that produces the answer via the
             paper's prompt. Passed as a lambda so the call site keeps its
             arguments. Used by the "llm" backend only.
    fail_safe: the value this decision falls back to.
    state: what the typed backend may look at -- the same facts the prompt
           spells out, unsummarised. Required by the "jev" backend.
    criteria: {option: description} when the options need describing beyond
              their own names. Defaults to the domain, undescribed.

  RETURNS:
    The chosen answer, in the vocabulary the call site already expects. Under
    "llm", an answer outside <domain> is returned as it is, exactly as before
    this module existed -- it is recorded, not corrected. Enforcing the domain
    would change the simulation's behaviour and belongs in its own registered
    deviation. Under "jev" the answer is inside <domain> by construction.
  """
  if _backend == "llm":
    # The scope listens for the retry loop reporting that it ran out of
    # attempts. Asking the answer whether it is the fail-safe does not work:
    # action_sector falls back to "kitchen", and "kitchen" is a real sector.
    with llm_trace.decision_scope(label) as watch:
      answer = resolve()
    llm_trace.record_decision(label, domain, answer, backend=_backend,
                              fail_safe=fail_safe, gave_up=watch.gave_up)
    return answer

  if _backend != "jev":
    raise ValueError("unknown decision backend: %s" % _backend)

  spec = _require_spec(label)
  if spec["type"] == "noul":
    return _resolve_noul(label, spec, domain, state, fail_safe)
  return _resolve_choice(label, spec, domain, state, fail_safe, criteria)


def _resolve_choice(label, spec, domain, state, fail_safe, criteria):
  if criteria is None:
    criteria = dict((option, None) for option in domain)
  if not criteria:
    raise ValueError("%s: a choice needs at least one option" % label)
  if len(criteria) > 255:
    raise ValueError("%s: a Choice accepts up to 255 options, got %d"
                     % (label, len(criteria)))

  question = jev.choice(spec["instructions"], criteria)
  answers = _ask(label, state, question, fail_safe)
  if answers is _FAILED:
    return _failsafe(label, domain, fail_safe)

  answer, probabilities, confidence = jev.read_choice(answers[_QUESTION_KEY])
  llm_trace.record_decision(
      label, domain, answer, backend=_backend, fail_safe=fail_safe,
      gave_up=False,
      extra={"confidence": confidence,
             "probabilities": _top(probabilities),
             "question_type": "choice"})
  return answer


def _resolve_noul(label, spec, domain, state, fail_safe):
  """A yes/no decision. The paper's prompt reasons its way to "yes" or "no";
  a Noul answers with how likely the statement is, and the threshold turns
  that back into the yes/no the caller expects."""
  question = jev.noul(spec["instructions"])
  answers = _ask(label, state, question, fail_safe)
  if answers is _FAILED:
    return _failsafe(label, domain, fail_safe)

  probability = jev.read_noul(answers[_QUESTION_KEY])
  answer = "yes" if probability > _options["noul-threshold"] else "no"
  llm_trace.record_decision(
      label, domain, answer, backend=_backend, fail_safe=fail_safe,
      gave_up=False,
      extra={"noul": probability,
             "threshold": _options["noul-threshold"],
             "question_type": "noul"})
  return answer


# --- rubric score -----------------------------------------------------------

def score(label, resolve, fail_safe=None, state=None):
  """Resolve one score on the paper's 1-10 poignancy scale.

  <resolve> is the paper's prompt path, and under the "llm" backend its
  result is handed back untouched -- including the False that
  ChatGPT_safe_generate_response returns when it runs out of retries, which
  the call site already handles.

  Under "jev" the answer is a Score over the rubric registered for <label>:
  a float across the rubric's levels, rounded to the integer the rest of the
  simulation stores. The float is kept in the trace, because how far the
  model sat from the integer it was forced into is the thing a calibrated
  model can report and the prompt cannot.
  """
  if _backend == "llm":
    with llm_trace.decision_scope(label) as watch:
      answer = resolve()
    llm_trace.record_decision(label, None, answer, backend=_backend,
                              fail_safe=fail_safe, gave_up=watch.gave_up)
    return answer

  if _backend != "jev":
    raise ValueError("unknown decision backend: %s" % _backend)

  spec = _require_spec(label)
  levels = spec.get("criteria") or []
  if not levels:
    raise ValueError("%s: no rubric registered for this score" % label)

  question = jev.score(spec["instructions"], levels)
  answers = _ask(label, state, question, fail_safe)
  if answers is _FAILED:
    return _failsafe(label, None, fail_safe)

  raw, confidence = jev.read_score(answers[_QUESTION_KEY])
  # The rubric's level i is the scale's level i+1: the paper stores 1-10.
  answer = int(min(len(levels), max(1, round(raw) + 1)))
  llm_trace.record_decision(
      label, None, answer, backend=_backend, fail_safe=fail_safe,
      gave_up=False,
      extra={"raw_score": raw,
             "confidence": confidence,
             "levels": len(levels),
             "question_type": "score"})
  return answer


# --- shared plumbing --------------------------------------------------------

_FAILED = object()


def _require_spec(label):
  spec = jev_questions.spec(label)
  if spec is None:
    # Better to stop than to quietly leave this decision on the LLM while the
    # run is labelled as a Jev run.
    raise NotImplementedError(
        "decision '%s' has no typed question in jev_questions.py" % label)
  return spec


def _ask(label, state, question, fail_safe):
  if state is None:
    raise ValueError(
        "decision '%s' reached the jev backend without a state; the call site "
        "must pass the facts its prompt spells out" % label)
  try:
    return jev.ask(state, {_QUESTION_KEY: question}, label=label)
  except jev.JevError as e:
    if _options["on-error"] == "raise":
      raise
    print("decision: %s fell back to %r (%s)" % (label, fail_safe, e))
    return _FAILED


def _failsafe(label, domain, fail_safe):
  llm_trace.record_decision(label, domain, fail_safe, backend=_backend,
                            fail_safe=fail_safe, gave_up=True,
                            extra={"backend_error": True})
  return fail_safe


def _top(probabilities, limit=8):
  """The whole distribution is interesting up to a point; a Choice over 60
  objects would otherwise put 60 floats in every trace line."""
  if not probabilities:
    return {}
  ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
  return dict(ranked[:limit])
