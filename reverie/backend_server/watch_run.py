#!/usr/bin/env python3
"""
File: watch_run.py
Description: watch a run while it happens, and say the moment something this
             lab already fixed comes back.

WHY THIS EXISTS

On 2026-09-23 four deviations landed in one day -- D7, D8, D9 and D10 -- and
the generator and its provider both changed (AP-2). **Not one of those six
changes has ever executed inside a simulation.** They are held by unit tests
and by a probe that asked a backend questions; a window is a different
animal, and the lab already has a gotcha for exactly this shape: GT-3, where
the routed model answered with a brace attached, the domain guard forgave it
and the maze killed the run one step later.

So this reads the trace WHILE it is written and prints one line per thing
worth interrupting for. It is meant to be pointed at a run by a watcher that
turns each line into a notification; it decides nothing and changes nothing.

WHAT IT FLAGS, AND WHICH FIX EACH ONE GUARDS

  fail-safe          a decision the backend never answered, so the call
                     site's own fallback entered the world. Read from
                     `gave_up`, never from `is_fail_safe`: the latter only
                     says the answer EQUALS the fail-safe value, so a typed
                     `kitchen` at confidence 1.0 or a genuine poignancy of 4
                     read as fail-safes (the shakedown of 2026-09-23 saw
                     seven such "fail-safes" and not one real one). On the two
                     social questions this is what D9, D10 and the new
                     generator exist to remove: before them it was ~94% of
                     decide_to_talk, and if it is still high, one of the
                     three did not land.
  gave-up            the backend answered nothing at all after its retries.
  guard              the domain guard substituted an answer -- GT-3's own
                     shape. A new model has never been asked the three
                     location prompts by this lab, and this is where a
                     different formatting habit shows up first.
  brace              a decision answer carrying punctuation the maze will
                     not forgive. GT-3 again, one step earlier than the
                     crash. An apostrophe inside a name ("Eddy Lin's
                     bedroom") is the world's own spelling and is not one.
  truncated          a response cut off at max_tokens. On the two social
                     prompts that means D9's budget of 160 is short for
                     this model.
  transport          a refused or timed-out request. The provider pin names
                     one provider and forbids fallbacks, so an outage is a
                     stopped run, not a slower one.
  rate-limit         the ride-out waiting. Bounded, but it is the failure
                     that invalidated n9probe3.
  backend-down       the typed backend gave up on the run.
  gen-fail-safe      a generator prompt (not a decision) exhausted its
                     retries and its fail-safe entered the world -- an
                     emoji, an object's state, a focal point. A new
                     generator's formatting habits show up here first: on
                     2026-09-23 gemma-4-31b-it wrapped its JSON in markdown
                     fences and pronunciatio fell back 8 times in 8.
  cost               the run's spend against the config's upper bound.

WHAT IT COUNTS, FOR 009-IP

Every `reject` is tallied by label and reason. That tally is the question
[009-IP] asks -- whether the paper's validators throw away answers a
forgiving reader would keep -- put to a generator nobody has asked it of.
OD-1 measured 241 of 241 rejections already holding a valid choice on the
location trio, on the old model. `--digest` prints the same tally as the run
goes, so the answer arrives with the shakedown instead of needing its own.

USAGE

  cd reverie/backend_server
  ../../../.venv/bin/python watch_run.py --trace ../../logs/llm_trace --follow

  # everything already written, once, with the tally
  ../../../.venv/bin/python watch_run.py --trace ../../logs/llm_trace --digest
"""
import os
import sys
import json
import glob
import time
import argparse
from collections import Counter, defaultdict

SOCIAL = ("decide_to_talk", "decide_to_react")
LOCATION = ("action_sector", "action_arena", "action_game_object")

# Punctuation the domain guard strips and the maze does not (GT-3). A single
# quote only counts at an edge: inside, it is a possessive the world spells.
_BAD_CHARS = "{}[]\""


def fell_back(row):
  """Did the decision's fail-safe actually enter the world? `gave_up` says
  so; `is_fail_safe` is only consulted on traces written before it existed."""
  if row.get("gave_up") is not None:
    return bool(row.get("gave_up"))
  return bool(row.get("is_fail_safe"))


def has_brace(answer):
  if not isinstance(answer, str):
    return False
  stripped = answer.strip()
  return (any(c in stripped for c in _BAD_CHARS)
          or stripped[:1] == "'" or stripped[-1:] == "'")


def _flag(kind, row, message):
  where = "step %s" % row.get("step")
  who = row.get("persona") or "?"
  return "[%s] %s %s | %s" % (kind.upper(), where, who, message)


def examine(row, seen):
  """RETURNS: a list of flag lines this row deserves."""
  out = []
  event = row.get("event")
  label = row.get("label") or "?"

  if event == "decision":
    if fell_back(row):
      out.append(_flag("fail-safe", row,
                       "%s took the call site's fail-safe (%r)"
                       % (label, row.get("answer"))))
    answer = row.get("answer")
    if has_brace(answer):
      out.append(_flag("brace", row,
                       "%s answered %r -- GT-3's shape, one step before the "
                       "maze refuses it" % (label, answer)))
    if row.get("in_domain") is False:
      out.append(_flag("out-of-domain", row,
                       "%s answered %r, which is not in its %s options"
                       % (label, answer, row.get("domain_size"))))

  elif event == "guard":
    out.append(_flag("guard", row,
                     "%s answered %r and the guard substituted %r -- GT-3"
                     % (label, row.get("answer"), row.get("replacement"))))

  elif event == "reject":
    reason = row.get("reason") or "?"
    if reason == "truncated" and label.replace("run_gpt_prompt_", "") in SOCIAL:
      out.append(_flag("truncated", row,
                       "%s was cut at max_tokens -- D9's 160 is short for "
                       "this model" % label))
    elif reason.startswith("transport:"):
      out.append(_flag("transport", row, "%s: %s" % (label, reason)))

  elif event == "outcome" and row.get("failsafe"):
    out.append(_flag("gen-fail-safe", row,
                     "%s fell back after %s attempts"
                     % (label, row.get("attempts"))))

  elif event == "rate_limit":
    out.append(_flag("rate-limit", row,
                     "%s waiting %ss (%s of %s)"
                     % (row.get("path"), row.get("delay"),
                        row.get("attempt"), row.get("of"))))

  elif event == "backend_down":
    out.append(_flag("backend-down", row,
                     "%s down after %s consecutive errors"
                     % (row.get("backend"), row.get("consecutive"))))

  return out


class Tally(object):
  """What the run is doing, for the digest and for 009-IP's question."""

  def __init__(self):
    self.decisions = Counter()
    self.failsafes = Counter()
    self.rejects = defaultdict(Counter)
    self.guards = Counter()
    self.outcomes = Counter()
    self.gen_failsafes = Counter()
    self.steps = set()

  def add(self, row):
    event = row.get("event")
    label = (row.get("label") or "?").replace("run_gpt_prompt_", "")
    if isinstance(row.get("step"), int):
      self.steps.add(row["step"])
    if event == "decision":
      self.decisions[label] += 1
      if fell_back(row):
        self.failsafes[label] += 1
    elif event == "reject":
      self.rejects[label][row.get("reason") or "?"] += 1
    elif event == "guard":
      self.guards[label] += 1
    elif event == "outcome":
      self.outcomes[label] += 1
      if row.get("failsafe"):
        self.gen_failsafes[label] += 1

  def digest(self):
    lines = []
    if self.steps:
      lines.append("steps %d..%d" % (min(self.steps), max(self.steps)))
    lines.append("")
    lines.append("| decision | asked | fail-safe | guard |")
    lines.append("|---|---:|---:|---:|")
    for label in sorted(self.decisions):
      n = self.decisions[label]
      fs = self.failsafes[label]
      lines.append("| %s | %d | %d (%.0f%%) | %d |"
                   % (label, n, fs, 100.0 * fs / n, self.guards[label]))
    if self.gen_failsafes:
      lines.append("")
      lines.append("Generator prompts whose fail-safe entered the world:")
      lines.append("")
      lines.append("| prompt | asked | fell back |")
      lines.append("|---|---:|---:|")
      for label in sorted(self.gen_failsafes):
        n = self.outcomes[label]
        fs = self.gen_failsafes[label]
        lines.append("| %s | %d | %d (%.0f%%) |"
                     % (label, n, fs, 100.0 * fs / n))
    if self.rejects:
      lines.append("")
      lines.append("Rejections -- the question 009-IP asks, of a generator "
                   "nobody has asked it of:")
      lines.append("")
      lines.append("| prompt | rejections | reasons |")
      lines.append("|---|---:|---|")
      for label in sorted(self.rejects):
        reasons = self.rejects[label]
        lines.append("| %s | %d | %s |"
                     % (label, sum(reasons.values()),
                        ", ".join("%s %d" % (r, n)
                                  for r, n in reasons.most_common())))
    return "\n".join(lines)


def rows_of(handle):
  for line in handle:
    line = line.strip()
    if not line:
      continue
    try:
      yield json.loads(line)
    except ValueError:
      continue


def newest(path):
  if os.path.isfile(path):
    return path
  files = glob.glob(os.path.join(path, "*.jsonl"))
  return max(files, key=os.path.getmtime) if files else None


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
  parser.add_argument("--trace", required=True,
                      help="the trace file, or the directory the run writes to")
  parser.add_argument("--follow", action="store_true",
                      help="keep reading as the run writes, and move to a new "
                           "file when a stage rolls over")
  parser.add_argument("--digest", action="store_true",
                      help="print the tally at the end (or every --every "
                           "flags while following)")
  parser.add_argument("--every", type=int, default=0,
                      help="while following, print the tally this many "
                           "seconds apart; 0 means never")
  parser.add_argument("--quiet-failsafe", action="store_true",
                      help="count fail-safes but do not print one per "
                           "occurrence -- for a run where they are expected")
  args = parser.parse_args(argv)

  tally = Tally()
  current = newest(args.trace)
  if current is None:
    print("(watch): nothing in %s yet" % args.trace)
    if not args.follow:
      return 1
  handle = open(current) if current else None
  last_digest = time.time()

  try:
    while True:
      if handle is not None:
        for row in rows_of(handle):
          tally.add(row)
          for line in examine(row, tally):
            if args.quiet_failsafe and line.startswith("[FAIL-SAFE]"):
              continue
            print(line)
            sys.stdout.flush()
      if not args.follow:
        break
      time.sleep(2)
      # A stage rollover opens a new file; follow it without losing the tally.
      latest = newest(args.trace)
      if latest and latest != current:
        if handle:
          handle.close()
        current, handle = latest, open(latest)
        print("[STAGE] now reading %s" % os.path.basename(latest))
        sys.stdout.flush()
      elif handle is None and latest:
        current, handle = latest, open(latest)
      if args.every and time.time() - last_digest >= args.every:
        print(tally.digest())
        sys.stdout.flush()
        last_digest = time.time()
  except KeyboardInterrupt:
    pass
  finally:
    if handle:
      handle.close()

  if args.digest:
    print(tally.digest())
  return 0


if __name__ == "__main__":
  sys.exit(main())
