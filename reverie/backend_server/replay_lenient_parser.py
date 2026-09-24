"""
File: replay_lenient_parser.py
Description: The ceiling of 009-IP -- how many of a run's rejections the
             lenient parser (D6) would have accepted, read off the trace
             before anything is paid for.

Usage (from reverie/backend_server, with the project venv):

  python replay_lenient_parser.py --name n9w11-jev ../../../runs/n9w11-jev/trace
  python replay_lenient_parser.py --name control ../../logs/llm_trace/n9w11-llm-*.jsonl

Each path is a trace file (.jsonl or .jsonl.gz), a directory of them, or a
glob. Every `reject` record is re-validated under the three tolerances of
LP-D and filed as ACCEPTED (D6 would have kept it), STILL REJECTED (with the
reason -- ambiguous, truncated, or a prompt D6 does not touch) or UNJUDGEABLE
(the recorded head was cut before the answer could be read). Beside them,
per prompt: how many times the prompt was asked (`outcome` records) and how
many times its fail-safe entered the world (`failsafe: true`).

What the trace lets the replay see, per path:

  * focal_pt travels the JSON path, and a `validate_false` there records the
    PARSED "output" value -- the list itself, as Python prints it. The head is
    4,000 characters on the 011-IP worktree, whole for every list seen so far;
    a head ending in "..." is unjudgeable.
  * an `exception:JSONDecodeError` records the raw text cut at the last "}",
    because the assignment that would have replaced it raised. When the head
    is whole the replay can re-cut it at the first "{" and parse; when it is
    cut, unjudgeable. (009-IP's Q under S3.1 expected null here; the code
    keeps the cut text.)
  * the location trio travels the completion path and records the raw text.

Reads only; it never touches a simulation, and it is never told an option set.
"""
import argparse
import ast
import collections
import glob
import gzip
import json
import os
import sys

# The three tolerances are lenient_parser's own pure functions, never copies.
# Step 1 of 009-IP read the ceiling with the module on disk and wired into no
# call site (fork commit named in the plan), so the counterfactual preceded
# the change in behaviour; step 3 re-read it through the wired code and the
# two tables had to agree to the row.
from persona.prompt_template.lenient_parser import (
    focal_pt_validate, json_cut, location_validate)

FOCAL = "run_gpt_prompt_focal_pt"
LOCATION = ("run_gpt_prompt_action_sector", "run_gpt_prompt_action_arena")
TRIPLES = ("run_gpt_prompt_event_triple",
           "run_gpt_prompt_act_obj_event_triple")

ACCEPTED = "accepted"
STILL = "still rejected"
UNJUDGEABLE = "unjudgeable"


def files_of(paths):
  out = []
  for p in paths:
    if os.path.isdir(p):
      out += sorted(glob.glob(os.path.join(p, "*.jsonl*")))
    elif any(ch in p for ch in "*?["):
      out += sorted(glob.glob(p))
    else:
      out.append(p)
  return out


def rows(path):
  opener = gzip.open if path.endswith(".gz") else open
  with opener(path, "rt", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except ValueError:
        continue


def head_is_cut(head):
  return isinstance(head, str) and head.endswith("...")


def judge(record):
  """(verdict, reason) for one reject record."""
  label = record.get("label")
  reason = record.get("reason") or ""
  head = record.get("resp_head")

  if reason.startswith("transport"):
    return STILL, "transport failure, not a parse"
  if reason.startswith("truncated"):
    return STILL, "truncated by the token budget (FB-E), not a parse"

  if reason.startswith("exception:JSONDecodeError"):
    if head is None or head_is_cut(head):
      return UNJUDGEABLE, "JSON path, raw text cut in the trace"
    try:
      json.loads(json_cut(head))["output"]
      return ACCEPTED, "fenced JSON, parses from the first '{'"
    except Exception:
      return STILL, "JSON path, does not parse even from the first '{'"

  if label == FOCAL:
    if head is None or head_is_cut(head):
      return UNJUDGEABLE, "focal_pt, list cut in the trace"
    try:
      value = ast.literal_eval(head)
    except Exception:
      value = head
    if focal_pt_validate(value):
      return ACCEPTED, "a list of str, refused by the text path's cleanup"
    return STILL, "focal_pt, not a list of str"

  if label in LOCATION:
    if head is None:
      return UNJUDGEABLE, "location, no text recorded"
    if location_validate(head):
      return ACCEPTED, "one option name, brace missing or doubled"
    if "," in head:
      return STILL, "ambiguous: a comma names more than one option (LP-A)"
    return STILL, "location, empty after the braces come off"

  if label in TRIPLES:
    text = str(head or "")
    parts = [p for p in text.split(")")[0].split(",")]
    if len(parts) >= 3:
      return STILL, "ambiguous: three or more parts after the subject (LP-A)"
    return STILL, "triple, not in D6"

  return STILL, "not a prompt D6 touches"


class Tally(object):

  def __init__(self):
    self.asked = collections.Counter()
    self.fell_back = collections.Counter()
    self.rejected = collections.Counter()
    self.verdicts = collections.defaultdict(collections.Counter)
    self.reasons = collections.defaultdict(collections.Counter)
    self.samples = collections.defaultdict(list)

  def add(self, record):
    event = record.get("event")
    label = record.get("label")
    if event == "outcome":
      self.asked[label] += 1
      if record.get("failsafe"):
        self.fell_back[label] += 1
    elif event == "reject":
      self.rejected[label] += 1
      verdict, reason = judge(record)
      self.verdicts[label][verdict] += 1
      self.reasons[label][(verdict, reason)] += 1
      if verdict == ACCEPTED and len(self.samples[label]) < 3:
        self.samples[label].append(str(record.get("resp_head"))[:160])

  def table(self, name):
    labels = sorted(self.rejected, key=lambda l: -self.rejected[l])
    lines = ["== %s" % name,
             "%-40s %6s %6s %6s | %8s %8s %8s" % (
                 "prompt", "asked", "reject", "fellbk",
                 "D6 acc", "still", "unjudg")]
    for label in labels:
      v = self.verdicts[label]
      lines.append("%-40s %6d %6d %6d | %8d %8d %8d" % (
          label.replace("run_gpt_prompt_", ""), self.asked[label],
          self.rejected[label], self.fell_back[label],
          v[ACCEPTED], v[STILL], v[UNJUDGEABLE]))
      for (verdict, reason), n in sorted(self.reasons[label].items(),
                                         key=lambda kv: -kv[1]):
        lines.append("%-40s %30s %5d  %s" % ("", verdict, n, reason))
    for label in labels:
      for s in self.samples[label]:
        lines.append("  e.g. %s: %s" % (label.replace("run_gpt_prompt_", ""), s))
    return "\n".join(lines)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("paths", nargs="+")
  parser.add_argument("--name", default="trace")
  args = parser.parse_args(argv)
  tally = Tally()
  files = files_of(args.paths)
  for path in files:
    for record in rows(path):
      tally.add(record)
  print("%d file(s)" % len(files))
  print(tally.table(args.name))


if __name__ == "__main__":
  main()
