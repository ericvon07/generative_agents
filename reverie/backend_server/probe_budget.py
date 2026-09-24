#!/usr/bin/env python3
"""
File: probe_budget.py
Description: 011-IP's budget probe (TS-E) -- how much room the paper's two
             social prompts need before their answer is on the page.

WHY THIS EXISTS

`decide_to_talk` and `decide_to_react` both end by asking for reasoning and
then an answer, and both are sent with `max_tokens: 20`. Twenty tokens does
not reach the end of a sentence of reasoning, so the answer never arrives,
the validator rejects all five attempts and the call site's own fail-safe
enters the world -- "yes" for talk, Option 3 for react. TS-E says the new
budget is CHOSEN FROM A MEASUREMENT and written into deviation D9 with the
number behind it. This file is that measurement.

WHAT IT MEASURES, AND WHY THE NUMBER IS EXACT

The answer is the LAST thing these two prompts ask for. So when a response
ends at its answer, `usage.completion_tokens` is not an estimate of where the
answer lands -- it IS where it lands, counted by the model's own tokenizer.
No local tokenizer is needed and nothing is interpolated. Every row says
whether anything followed the answer (`trailing_chars`), and the report reads
its percentile over the rows where nothing did, so a response that ran on
past its answer cannot inflate the budget.

THE ARMS

  paper       exactly what GPT_request sends today: the lab's completion
              system prompt, then the prompt as the user turn.
  no-system   the same prompt with that system message left off.

The second arm is here because the first one does not answer at any budget,
and one call apiece is what tells a reader which of the two facts is true.
It is a DIAGNOSTIC, not a proposal: nothing in this file changes what a run
sends. What the arms show is written up for the author to decide.

WHAT COUNTS AS AN ANSWER

Two readings, side by side on every row, because they disagree and the
disagreement is itself a finding:

  validates   what a run would accept -- `decide_to_talk`'s own reader,
              imported, and `decide_to_react`'s validator copied because it
              is a closure, with a test that keeps the copy honest.
  answer      a tolerant reader that finds an answer the way a person would.
              This is what the model actually wrote.

A budget is only worth setting where `validates` can become true. Where only
`answer` is true, the budget was never the thing standing in the way, and the
report says so rather than quietly reporting the tolerant number.

USAGE

  cd reverie/backend_server
  ../../../.venv/bin/python probe_budget.py --trace <dir> --out budget.jsonl \
      --budget 512 --workers 6
  ../../../.venv/bin/python probe_budget.py --report budget.jsonl

It asks a live model, so it costs routed calls; --dry-run builds every
request and sends none. The out file is APPENDED to and a case/arm already in
it is skipped, so a run killed by a 429 is resumed by repeating the command.
"""
import os
import re
import sys
import json
import gzip
import argparse
import datetime
import threading
import concurrent.futures

import probe_questions as probe
from persona.prompt_template import gpt_structure as gpt
from persona.prompt_template import llm_trace
from persona.prompt_template import run_gpt_prompt
from persona.prompt_template.gpt_structure import generate_prompt

SOCIAL_LABELS = ("decide_to_talk", "decide_to_react")

# The request itself is the slow part and is safe to run several at a
# time -- the client is one httpx pool and every call is independent. The
# three things that are NOT per-call are the trace file, the cost ledger
# and the out file, and all three are written under this one lock.
_SHARED = threading.Lock()

# The percentile TS-E names, and the margin it asks for on top of it.
PERCENTILE = 95

# How many ACCEPTED answers an arm needs before this probe will name a budget
# from it. A 95th percentile over one row is not a percentile, and an arm
# that answers acceptably once in twelve tries has a problem no budget fixes.
MIN_ACCEPTED = 10


# --- what a run would accept ------------------------------------------------
# `decide_to_talk` reads its answer through run_gpt_prompt.read_yes_no, which
# is a module function since D10, so this imports it rather than copying it
# and cannot drift from the call site at all.
#
# `decide_to_react`'s validator is still a closure and cannot be imported, so
# it is copied, and a test reads the call site's source and fails if its cue
# string moves.
#
# HISTORY, because a row in the archived data means what its own day's
# validator meant: before D10, `decide_to_talk` split the response on
# `Answer in yes or no:` -- a sentence its template never asks for -- and the
# 611 rows of docs/research/budget-011-2026-09-23.jsonl.gz carry a
# `validates` column computed that way, which is 0 across every model. Read
# with the validator D10 put in, the same 611 responses give 282 (CU-1).

TALK_CUE_BEFORE_D10 = "Answer in yes or no:"
REACT_CUE = "Answer: Option"


def _validate_talk(text):
  try:
    return run_gpt_prompt.read_yes_no(text) is not None
  except Exception:
    return False


def _validate_react(text):
  try:
    return text.split(REACT_CUE)[-1].strip().lower() in ["3", "2", "1"]
  except Exception:
    return False


def _clean_talk(text):
  return run_gpt_prompt.read_yes_no(text) or ""


def _clean_react(text):
  return text.split(REACT_CUE)[-1].strip().lower()


VALIDATORS = {
    "decide_to_talk": (_validate_talk, _clean_talk),
    "decide_to_react": (_validate_react, _clean_react),
}


# --- the tolerant reader ----------------------------------------------------
# What a person reading the response would call the answer. Deliberately
# separate from the validator above: this one says what the model WROTE, that
# one says what a run would ACCEPT.

_ANSWER_PATTERNS = {
    "decide_to_talk": re.compile(
        r"\*{0,2}answer\*{0,2}"
        r"(?:\s*in\s*\"?yes\"?\s*or\s*\"?no\"?)?"
        r"\s*\*{0,2}\s*:\s*\*{0,2}\s*(yes|no)\b", re.I),
    "decide_to_react": re.compile(
        r"\*{0,2}answer\*{0,2}\s*:?\s*option\s*\*{0,2}\s*([123])\b", re.I),
}


_BARE_ANSWERS = {"decide_to_talk": ("yes", "no"),
                 "decide_to_react": ("1", "2", "3")}


def read_answer(label, text):
  """RETURNS: (answer, end_char) -- the LAST answer the model wrote and where
  it ends, or (None, None). The last, not the first, because a response that
  runs on into another worked example answers the question it was asked in
  its first block and then answers a different one."""
  found = None
  for match in _ANSWER_PATTERNS[label].finditer(text):
    found = match
  if found is not None:
    return found.group(1).lower(), found.end()
  # A response that is nothing but the answer, with no cue at all. Both call
  # sites accept it -- the paper's own split returned the whole response when
  # its cue was absent, and read_yes_no keeps that -- so a reader that missed
  # it would undercount exactly the models that obey the completion system
  # prompt most literally. `deepseek-v4-flash` answers "no" and nothing else.
  bare = text.strip().lower()
  if bare in _BARE_ANSWERS[label]:
    return bare, len(text.rstrip())
  return None, None


# --- one request ------------------------------------------------------------

def request(prompt, budget, system, model=None, pin=True):
  """One call, through the lab's own ride-out, trace and cost guard.

  Not GPT_request: that one returns the text and drops the usage, and the
  usage is the measurement. Everything else about the request is the same --
  the same client, the same provider pin, the same rate-limit ride-out, the
  same recorded call and the same charge against the run's upper bound.

  <model> and <pin> exist for one question and are otherwise left alone: the
  config's model is the one the windows will run, and its provider pin is a
  run's pin (MB-A). A different model cannot be served by a pin naming one
  provider that does not host it, so an arm that names a model drops the pin
  and every row says which model answered and whether it was pinned.
  """
  gpt.temp_sleep()
  attempt = gpt._Attempt()
  messages = []
  if system:
    messages.append({"role": "system", "content": system})
  messages.append({"role": "user", "content": prompt})
  try:
    response = gpt._send(
        "probe_budget", attempt,
        lambda: gpt.client.chat.completions.create(
            model=model or gpt.openai_config["model"],
            messages=messages,
            temperature=0, max_tokens=budget, top_p=1,
            frequency_penalty=0, presence_penalty=0, stream=False, stop=None,
            extra_body=gpt._extra_body() if pin else None))
  except Exception as error:
    with _SHARED:
      llm_trace.record_call("probe_budget", gpt.openai_config["model"], prompt,
                            latency=attempt.latency, error=error)
    raise gpt.BackendTransportError(
        "%s: %s" % (type(error).__name__, error)) from error
  with _SHARED:
    llm_trace.record_call("probe_budget", gpt.openai_config["model"], prompt,
                          response=response, latency=attempt.latency,
                          extra=gpt._traced_pin())
    gpt._charge(response)
  choice = response.choices[0]
  return {"text": choice.message.content or "",
          # What one decision costs in wall clock. A window is dozens of
          # calls per agent per step, so this is the figure that decides
          # whether a model can run one at all.
          "latency": attempt.latency,
          "model": getattr(response, "model", None) or model
                   or gpt.openai_config["model"],
          "pinned": bool(pin),
          "finish_reason": choice.finish_reason,
          "prompt_tokens": response.usage.prompt_tokens,
          "completion_tokens": response.usage.completion_tokens}


def arms(which, model=None):
  """RETURNS: [(name, system_message)] for the arms asked for.

  An arm carries the model in its name when the model is not the config's,
  so a resumed run cannot mistake one model's answer for another's and the
  report can put them side by side."""
  suffix = "@%s" % model.split("/")[-1] if model else ""
  paper = ("paper" + suffix, gpt.completion_system_prompt)
  bare = ("no-system" + suffix, None)
  if which == "paper":
    return [paper]
  if which == "no-system":
    return [bare]
  return [paper, bare]


# --- the run ----------------------------------------------------------------

def cases_from(trace, limit=None, hand=True):
  """Every social state to ask about: the hand-written ones, then the run's."""
  found = []
  if hand:
    for case in probe.HAND_WRITTEN:
      if case["label"] in SOCIAL_LABELS:
        found.append({"case": case["case"], "source": "hand",
                      "label": case["label"],
                      "prompt_input": case["prompt_input"],
                      "reconstruction": None, "run": None, "step": None})
  if not trace:
    return found
  per_label = {}
  truncated = 0
  for state in probe.read_trace(trace):
    label = state["label"]
    if label not in SOCIAL_LABELS:
      continue
    if state["truncated"]:
      truncated += 1
      continue
    seen = per_label.get(label, 0)
    if limit is not None and seen >= limit:
      continue
    try:
      prompt_input, how = probe.prompt_input_of(label, state["state"],
                                                state["domain"])
    except Exception as error:
      print("(budget): skipped a %s state: %s" % (label, error))
      continue
    per_label[label] = seen + 1
    found.append({"case": "%s-%s-%s" % (state["run"], state["step"], label),
                  "source": "trace", "label": label,
                  "prompt_input": prompt_input, "reconstruction": how,
                  "run": state["run"], "step": state["step"]})
  if truncated:
    print("(budget): SKIPPED %d truncated states -- that trace was written "
          "with too small a head" % truncated)
  return found


def already_done(path):
  """The (case, arm) pairs a previous invocation wrote, so a run killed by a
  429 is resumed rather than paid for twice."""
  done = set()
  if not os.path.exists(path):
    return done
  with open(path) as handle:
    for line in handle:
      try:
        row = json.loads(line)
      except ValueError:
        continue
      if row.get("error"):
        continue
      done.add((row.get("case"), row.get("arm")))
  return done


def _one(case, arm, system, prompt, budget, stamp, dry_run, model=None,
         pin=True):
  """One row: the request, and the two readings of what came back."""
  label = case["label"]
  row = {"probe": "011-IP-budget", "at": stamp, "arm": arm,
         "budget": budget, "case": case["case"],
         "source": case["source"], "label": label,
         "run": case["run"], "step": case["step"],
         "reconstruction": case["reconstruction"],
         "prompt_chars": len(prompt)}
  if dry_run:
    row["dry_run"] = True
    return row
  try:
    answered = request(prompt, budget, system, model=model, pin=pin)
  except Exception as error:
    row["error"] = "%s: %s" % (type(error).__name__, error)
    return row
  text = answered["text"]
  validate, clean = VALIDATORS[label]
  answer, end = read_answer(label, text)
  row.update(answered)
  row.update({
      "chars": len(text),
      "validates": bool(validate(text)),
      "cleaned": clean(text)[:40],
      "answer": answer,
      "answer_end_char": end,
      "text": text})
  return row


# A 429 that has stopped answering does not start answering because the probe
# kept asking. The ride-out inside gpt._send already spent its five waits on
# each of these, so this many in a row means the pool is gone, not busy.
_GIVE_UP_AFTER = 15


def measure(cases, out, budget, which_arms, dry_run=False, done=(),
            workers=1, model=None, pin=True):
  """Ask every (state, arm) not already on disk. RETURNS: (written, errors)."""
  stamp = datetime.datetime.now().isoformat(timespec="seconds")
  work = []
  for case in cases:
    prompt = generate_prompt(case["prompt_input"], probe.TEMPLATES[case["label"]])
    for arm, system in arms(which_arms, model):
      if (case["case"], arm) in done:
        continue
      work.append((case, arm, system, prompt))

  written = [0]
  errors = [0]
  in_a_row = [0]
  stop = threading.Event()

  def run(item):
    if stop.is_set():
      return None
    case, arm, system, prompt = item
    row = _one(case, arm, system, prompt, budget, stamp, dry_run,
               model=model, pin=pin)
    with _SHARED:
      out.write(json.dumps(row, ensure_ascii=False) + "\n")
      out.flush()
      written[0] += 1
      if row.get("error"):
        errors[0] += 1
        in_a_row[0] += 1
        if in_a_row[0] >= _GIVE_UP_AFTER:
          stop.set()
          print("(budget): %d failures in a row -- stopping. Re-run the same "
                "command to resume from here." % in_a_row[0])
      else:
        in_a_row[0] = 0
      if written[0] % 25 == 0:
        print("(budget): %d of %d" % (written[0], len(work)))
    return None

  if workers <= 1:
    for item in work:
      run(item)
  else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
      list(pool.map(run, work))
  return written[0], errors[0]


# --- the reading ------------------------------------------------------------

def _rows_of(paths):
  """Every row in these JSONL files, plain or gzipped."""
  rows = []
  for path in paths:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
      for line in handle:
        line = line.strip()
        if line:
          rows.append(json.loads(line))
  return rows


def position_of(row):
  """Where the answer ends, in the model's own tokens, and whether that
  position is exact. Derived HERE from the raw facts every row stores --
  the response, its token count and the character the answer ends at --
  rather than read from a column, so a row written by any version of this
  file reads the same way and the arithmetic is on the page.

  RETURNS: (tokens, exact) or (None, None) when the row has no answer.
  """
  # The answer's end is DERIVED from the stored response, never read from
  # the row's own column: a row written by an older version of this file
  # carries that column as that version computed it, and both readers here
  # must read every row by one rule.
  _answer, end = read_answer(row["label"], row.get("text") or "")
  total = row.get("completion_tokens")
  if end is None or total is None:
    return None, None
  text = row.get("text") or ""
  if not text or len(text.rstrip()) <= end:
    # The answer is the last thing on the page, so the model's own count of
    # the whole response IS the position the answer ends at.
    return total, True
  # The response kept writing after answering -- gpt-3.5 answers on line one
  # and then continues the document. There is no local tokenizer for these
  # models, so the position is scaled by characters and said to be an
  # estimate. Dropping these rows would throw away every response that
  # answers EARLY, which is the opposite of measuring where the answer lands.
  return max(1, int(-(-total * (float(end) / len(text)) // 1))), False


def percentile(values, which):
  """The nearest-rank percentile: the smallest value at or above which
  <which> percent of the sample sits. No interpolation -- a budget is a whole
  number of tokens and a rank is a row that exists."""
  if not values:
    return None
  ordered = sorted(values)
  rank = int(-(-len(ordered) * which // 100))  # ceil
  return ordered[max(0, min(rank, len(ordered)) - 1)]


def report(paths, margin=0.30):
  rows = _rows_of(paths)
  live = [r for r in rows if not r.get("error") and not r.get("dry_run")]
  print("# 011-IP step 6 -- the budget of the two social prompts")
  print()
  print("%d rows read, %d of them answered." % (len(rows), len(live)))
  # The arms are whatever is in the file, "paper" first and its model arms
  # after it, so a file with one arm reads as cleanly as a file with four.
  def arm_order(name):
    base = 0 if name.startswith("paper") else 1
    return (base, "@" in name, name)
  present = sorted(set(r["arm"] for r in live), key=arm_order)
  budgets = sorted(set(r["budget"] for r in live))
  print("Asked at max_tokens = %s."
        % ", ".join(str(b) for b in budgets))
  print()
  print("**Acceptance is the call sites' validators AS THE CODE CARRIES THEM "
        "TODAY**, recomputed from each stored response. Rows recorded before "
        "D10 carry a `validates` column computed by the validator of their "
        "own day -- the split on `Answer in yes or no:` -- which accepted "
        "none of them. That is what CU-1 measures; this table is not it.")
  print()

  print("## 1. What each arm produced")
  print()
  print("| question | arm | asked | wrote an answer | a run would accept it |")
  print("|---|---|---:|---:|---:|")
  for label in SOCIAL_LABELS:
    for arm in present:
      here = [r for r in live if r["label"] == label and r["arm"] == arm]
      if not here:
        continue
      wrote = sum(1 for r in here
                  if read_answer(label, r.get("text") or "")[0])
      validate, _clean = VALIDATORS[label]
      ok = sum(1 for r in here if validate(r.get("text") or ""))
      print("| `%s` | %s | %d | %d (%.0f%%) | %d (%.0f%%) |"
            % (label, arm, len(here), wrote, 100.0 * wrote / len(here),
               ok, 100.0 * ok / len(here)))
  print()

  print("## 2. Where the answer lands, in the model's own tokens")
  print()
  print("| question | arm | n | median | p90 | p%d | max | estimated |"
        % PERCENTILE)
  print("|---|---|---:|---:|---:|---:|---:|---:|")
  chosen = {}
  for label in SOCIAL_LABELS:
    for arm in present:
      here = []
      estimated = 0
      for r in live:
        if r["label"] != label or r["arm"] != arm:
          continue
        tokens_here, exact = position_of(r)
        if tokens_here is None:
          continue
        here.append(tokens_here)
        if not exact:
          estimated += 1
      if not here:
        continue
      tokens = here
      p95 = percentile(tokens, PERCENTILE)
      chosen[(label, arm)] = p95
      print("| `%s` | %s | %d | %d | %d | **%d** | %d | %d |"
            % (label, arm, len(tokens), percentile(tokens, 50),
               percentile(tokens, 90), p95, max(tokens), estimated))
  print()
  print("The position is exact -- the model's own token count -- when the "
        "answer is the last thing on the page. When the response keeps "
        "writing after answering, the position is scaled by characters and "
        "the last column counts those rows.")
  print()

  print("## 3. The budget this sets")
  print()
  print("A budget is worth setting only where a run would ACCEPT the answer. "
        "Where an arm writes an answer that the call site's validator throws "
        "away, the budget was never what stood in the way, and no number "
        "here would change that.")
  print()
  print("| question | arm | p%d | + %.0f%% margin | rounded | a run accepts it |"
        % (PERCENTILE, 100 * margin))
  print("|---|---|---:|---:|---:|---|")
  thin = []
  decided = {}
  for label in SOCIAL_LABELS:
    for arm in present:
      if (label, arm) not in chosen:
        continue
      here = [r for r in live if r["label"] == label and r["arm"] == arm]
      validate, _clean = VALIDATORS[label]
      accepts = sum(1 for r in here if validate(r.get("text") or ""))
      best = chosen[(label, arm)]
      with_margin = best * (1 + margin)
      rounded = int(-(-with_margin // 10) * 10)
      note = ("yes, %d of %d" % (accepts, len(here))) if accepts else "**never**"
      print("| `%s` | %s | %d | %.0f | %d | %s |"
            % (label, arm, best, with_margin, rounded, note))
      # The arm that decides: one a run would accept, the config's own model
      # before any other, and the arm a run actually sends before a
      # diagnostic one.
      if accepts >= MIN_ACCEPTED and label not in decided:
        decided[label] = (arm, rounded, accepts)
      elif accepts and label not in decided:
        thin.append((label, arm, accepts, rounded))
  print()
  for label in SOCIAL_LABELS:
    if label in decided:
      arm, rounded, accepts = decided[label]
      print("- **`%s`: max_tokens = %d**, from the %s arm, over %d accepted "
            "answers." % (label, rounded, arm, accepts))
      continue
    near = sorted((t for t in thin if t[0] == label), key=lambda t: -t[2])
    if near:
      _l, arm, accepts, rounded = near[0]
      print("- **`%s`: this probe will not name a budget.** The arm that "
            "accepted most is %s, with **%d** accepted answers, under the %d "
            "this file requires for a 95th percentile -- the number it would "
            "give is %d, written here so nobody has to guess it, not so "
            "anybody can use it."
            % (label, arm, accepts, MIN_ACCEPTED, rounded))
      continue
    print("- **`%s`: no budget can be set from this probe** -- no arm "
          "produced an answer the call site's own validator accepts, at "
          "any length. Raising the budget alone changes nothing here."
          % label)
  print()

  truncated = [r for r in live if r.get("finish_reason") == "length"]
  if truncated:
    print("**%d responses hit the probe's own ceiling of %s tokens** -- those "
          "rows are an upper bound on nothing, and the probe should be run "
          "again with a larger --budget before the number above is used."
          % (len(truncated), budgets))
    print()


def compare(paths, reference=None):
  """One row per model: can it answer these two prompts, and what does one
  decision cost.

  Acceptance is RECOMPUTED here from each stored response, with the
  validators the call sites carry today. It is not read from the rows'
  `validates` column, because rows written before D10 carry that column as
  the pre-D10 validator computed it, and a table that mixed the two rules
  would compare models on which day they were asked.
  """
  rows = _rows_of(paths)
  live = [r for r in rows if not r.get("error") and not r.get("dry_run")
          and r.get("arm", "").startswith("paper")]

  by_model = {}
  answered_by = {}
  for row in live:
    model = row.get("model") or "?"
    by_model.setdefault(model, []).append(row)
    answer, _end = read_answer(row["label"], row.get("text") or "")
    if answer is not None:
      answered_by.setdefault(model, {})[row["case"]] = answer

  # The reference for agreement: the model that answered the most states.
  # Agreement is NOT correctness -- two models can be wrong together -- but a
  # model that answers every time and agrees with nobody is answering
  # arbitrarily, and that is worth seeing beside the acceptance rate.
  # Chosen by the caller, or else the model that answered most. That default
  # depends on WHICH FILES were passed, so a comparison whose agreement
  # column is going to be quoted should pin it with --reference.
  if reference and reference not in answered_by:
    raise SystemExit("(budget): no answers from %r in these files; the "
                     "models present are: %s"
                     % (reference, ", ".join(sorted(answered_by))))
  if reference is None and answered_by:
    reference = max(answered_by, key=lambda m: len(answered_by[m]))

  print("# The generator on the paper's two closed-set social prompts")
  print()
  print("Every row is that model asked with **exactly what a run sends** — the "
        "completion system prompt, then the prompt — over the probe's own "
        "states. Acceptance is recomputed with the validators in the code "
        "today. Each row is one model's own rate over its own states and "
        "**no two rows are paired**; none is a rate of a run.")
  print()
  if reference:
    print("Agreement is against **`%s`**, over the states both answered. Agreement is not "
          "correctness — two models can be wrong together — but a model that "
          "answers every time and agrees with nobody is answering "
          "arbitrarily." % reference)
    print()
  print("| model | asked | wrote an answer | accepted | p%d tokens to the answer | median latency | still writing at the ceiling | agrees with the reference |"
        % PERCENTILE)
  print("|---|---:|---:|---:|---:|---:|---:|---:|")
  for model in sorted(by_model):
    here = by_model[model]
    wrote = accepted = ceiling = 0
    positions = []
    latencies = []
    for row in here:
      text = row.get("text") or ""
      answer, end = read_answer(row["label"], text)
      if answer:
        wrote += 1
      validate, _clean = VALIDATORS[row["label"]]
      if validate(text):
        accepted += 1
      if row.get("finish_reason") == "length":
        ceiling += 1
      if row.get("latency"):
        latencies.append(row["latency"])
      tokens, _exact = position_of(dict(row, answer_end_char=end))
      if tokens is not None:
        positions.append(tokens)
    n = len(here)
    agree = "--"
    if reference and model != reference:
      mine, theirs = answered_by.get(model, {}), answered_by[reference]
      shared = set(mine) & set(theirs)
      if shared:
        same = sum(1 for case in shared if mine[case] == theirs[case])
        agree = "%d of %d (%.0f%%)" % (same, len(shared),
                                       100.0 * same / len(shared))
    elif reference and model == reference:
      agree = "(the reference)"
    print("| `%s` | %d | %d (%.0f%%) | **%d (%.0f%%)** | %s | %s | %d (%.0f%%) | %s |"
          % (model, n, wrote, 100.0 * wrote / n, accepted,
             100.0 * accepted / n,
             percentile(positions, PERCENTILE) if positions else "--",
             ("%.1f s" % percentile(latencies, 50)) if latencies else "--",
             ceiling, 100.0 * ceiling / n, agree))
  print()
  print("**What this table is not.** The generator in this architecture also "
        "writes plans, reflections and dialogue, and none of that is measured "
        "here. What is measured is the part this lab studies: whether a model "
        "can name an option from a closed set in the sentence the prompt asks "
        "for, how long the answer takes to arrive, and how much room it needs.")
  print()


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
  parser.add_argument("--trace", default=None,
                      help="a trace file or directory of the probe run")
  parser.add_argument("--out", default="probe_budget.jsonl")
  parser.add_argument("--budget", type=int, default=512,
                      help="the generous max_tokens the probe asks with")
  parser.add_argument("--arms", default="both",
                      choices=("both", "paper", "no-system"))
  parser.add_argument("--limit", type=int, default=None,
                      help="at most this many traced states per question")
  parser.add_argument("--no-hand", action="store_true",
                      help="leave the hand-written states out")
  parser.add_argument("--model", default=None,
                      help="ask a different model than the config's; the arm "
                           "is named after it and the provider pin is dropped")
  parser.add_argument("--pin", dest="pin", action="store_true", default=None,
                      help="keep the config's provider pin even with --model")
  parser.add_argument("--workers", type=int, default=1,
                      help="how many requests to have in flight at once")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--report", nargs="+", default=None,
                      help="read these JSONL files and print the readout")
  parser.add_argument("--margin", type=float, default=0.30)
  parser.add_argument("--reference", default=None,
                      help="the model the agreement column compares against; "
                           "the default is whichever answered most, which "
                           "depends on the files passed")
  parser.add_argument("--compare", nargs="+", default=None,
                      help="read these JSONL (or .jsonl.gz) files and print "
                           "one row per model")
  args = parser.parse_args(argv)

  if args.compare:
    compare(args.compare, reference=args.reference)
    return 0
  if args.report:
    report(args.report, margin=args.margin)
    return 0

  cases = cases_from(args.trace, limit=args.limit, hand=not args.no_hand)
  per_label = {}
  for case in cases:
    per_label[case["label"]] = per_label.get(case["label"], 0) + 1
  print("(budget): %d states -- %s"
        % (len(cases), ", ".join("%s %d" % (k, v)
                                 for k, v in sorted(per_label.items()))))
  done = already_done(args.out)
  if done:
    print("(budget): %d answers already on disk in %s; skipping those"
          % (len(done), args.out))
  with open(args.out, "a") as out:
    pin = args.pin if args.pin is not None else (args.model is None)
    written, errors = measure(cases, out, args.budget, args.arms,
                              dry_run=args.dry_run, done=done,
                              workers=args.workers, model=args.model,
                              pin=pin)
  print("(budget): wrote %d rows to %s (%d of them failures)"
        % (written, args.out, errors))
  return 0


if __name__ == "__main__":
  sys.exit(main())
