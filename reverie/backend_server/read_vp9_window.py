"""read_vp9_window.py -- the operations of the two Valentine's Day party runs over the paper's
two days, read with read_valentine.py's own functions from the traces under runs/<name>/trace/.

  python read_vp9_window.py [--until 17280] [--json out.json] [run ...]

Run from reverie/backend_server. Both runs end at step 17,280, 00:00 of 15 February, the
midnight that opens a third day, and the work of that one step (the new day's planning) is
inside their totals. This prints each figure over the whole run and over the paper's two
days, the steps before 17,280.

It reads the traces in the manifest's stage order, refuses a file whose sha1 is not the
manifest's, and hands read_valentine's functions (family_line, talk, conversations,
operations) the rows of steps before --until. The cost and time of the calls behind the
decisions are not in read_valentine.py; here they follow its rules: a call belongs to the
decision its compare_backends.decision_label names, every attempt counts, the generator's
cost is the router's billed `cost`, Jev's is $0.042 per million input tokens.

Reflection cycles are counted as the paper's focal-point prompts: its reflect() asks for
focal points once per cycle (run_gpt_prompt_focal_pt), so one resolved focal-point prompt is
one cycle.

Each run is read over its own decisions and its own days; the two are never paired.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import sys

READER = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(READER, "..", "..", "runs")
sys.path.insert(0, READER)
import read_valentine as rv  # noqa: E402
from compare_backends import decision_label  # noqa: E402

JEV_PER_MTOK = 0.042
DECISIONS = rv.LOCATION + rv.SOCIAL + rv.SCORES
FAMILIES = (("location", rv.LOCATION), ("social", rv.SOCIAL),
            ("scores", rv.SCORES))


def run_rows(run):
  folder = os.path.join(RUNS, run)
  with open(os.path.join(folder, "manifest.json")) as f:
    manifest = json.load(f)
  rows = []
  for stage in sorted(manifest["stages"], key=lambda s: s["index"]):
    path = os.path.join(folder, stage["trace"])
    with open(path, "rb") as f:
      raw = f.read()
    if hashlib.sha1(raw).hexdigest() != stage["sha1"]:
      raise SystemExit("%s: sha1 differs from the manifest" % path)
    for line in gzip.decompress(raw).decode("utf-8").splitlines():
      if line.strip():
        rows.append(json.loads(line))
  return manifest, rows


def call_cost(r):
  if r.get("path") == "jev":
    return (r.get("in_tok") or 0) * JEV_PER_MTOK / 1e6
  c = r.get("cost")
  if isinstance(c, (int, float)):
    return c
  if isinstance(c, dict):
    return sum(v for v in c.values() if isinstance(v, (int, float)))
  return 0.0


def window(rows):
  decisions = [r for r in rows if r.get("event") == "decision"]
  calls = [r for r in rows if r.get("event") == "call"]
  out = {"decisions": len(decisions),
         "by_label": dict(collections.Counter(r["label"] for r in decisions))}
  for fam, labels in FAMILIES + (("all", DECISIONS),):
    f = rv.family_line(decisions, calls, labels)
    behind = [r for r in calls if decision_label(r) in labels]
    f["cost_usd"] = sum(call_cost(r) for r in behind)
    f["latency_s"] = sum(r.get("latency_ms") or 0 for r in behind) / 1000.0
    if f["decisions"]:
      f["usd_per_1000"] = 1000.0 * f["cost_usd"] / f["decisions"]
      f["s_per_decision"] = f["latency_s"] / f["decisions"]
    out[fam] = f
  obj = rv.family_line(decisions, calls, ("action_game_object",))
  out["object_out_of_set"] = [obj["out_of_domain"], obj["checked"]]
  out["talk"] = rv.talk(decisions)
  convos = rv.conversations(rows)
  out["conversations"] = len(convos)
  out["utterances"] = sum(c["lines"] for c in convos)
  ops = rv.operations([], rows)  # no files: wall clock is not read here
  out["router_cost_usd"] = ops["router_cost_usd"]
  out["jev_cost_usd"] = ops["jev_cost_usd"]
  out["total_cost_usd"] = ops["router_cost_usd"] + ops["jev_cost_usd"]
  # every router call; operations() leaves out the one
  # ChatGPT_single_request each stage makes, though its cost sum keeps it
  out["generator_calls"] = sum(1 for r in calls
                               if r.get("path") not in ("jev", "embedding"))
  out["jev_calls"] = ops["calls"].get("jev", 0)
  out["rate_limit_waits"] = ops["rate_limit_waits"]
  out["reflection_cycles"] = sum(
      1 for r in rows if r.get("event") == "outcome"
      and r.get("label") == "run_gpt_prompt_focal_pt")
  return out


def show(name, whole, cut, until):
  def line(label, a, b, fmt):
    print("  %-38s %14s %14s" % (label, fmt % a, fmt % b))
  print("=" * 70)
  print("%s   whole run   |   steps before %d" % (name, until))
  line("decisions", whole["decisions"], cut["decisions"], "%d")
  for lab in sorted(set(whole["by_label"]) | set(cut["by_label"])):
    line("  " + lab, whole["by_label"].get(lab, 0),
         cut["by_label"].get(lab, 0), "%d")
  line("calls per decision", whole["all"]["calls_per_decision"],
       cut["all"]["calls_per_decision"], "%.4f")
  line("gave up (fail-safe)", whole["all"]["gave_up"], cut["all"]["gave_up"],
       "%d")
  line("cost of the decisions' calls $", whole["all"]["cost_usd"],
       cut["all"]["cost_usd"], "%.4f")
  line("$ per 1,000 decisions", whole["all"]["usd_per_1000"],
       cut["all"]["usd_per_1000"], "%.4f")
  line("s per decision, inside the call", whole["all"]["s_per_decision"],
       cut["all"]["s_per_decision"], "%.4f")
  for fam, _ in FAMILIES:
    if whole[fam]["decisions"]:
      line("  " + fam + " s per decision", whole[fam]["s_per_decision"],
           cut[fam]["s_per_decision"], "%.4f")
  line("router cost $", whole["router_cost_usd"], cut["router_cost_usd"],
       "%.4f")
  line("Jev cost $", whole["jev_cost_usd"], cut["jev_cost_usd"], "%.4f")
  line("TOTAL cost $", whole["total_cost_usd"], cut["total_cost_usd"], "%.4f")
  line("generator calls", whole["generator_calls"], cut["generator_calls"],
       "%d")
  line("Jev calls", whole["jev_calls"], cut["jev_calls"], "%d")
  line("rate-limit waits", whole["rate_limit_waits"], cut["rate_limit_waits"],
       "%d")
  print("  %-38s %14s %14s" % ("objects out of the offered set",
                               "%d of %d" % tuple(whole["object_out_of_set"]),
                               "%d of %d" % tuple(cut["object_out_of_set"])))
  line("conversations", whole["conversations"], cut["conversations"], "%d")
  line("utterances", whole["utterances"], cut["utterances"], "%d")
  line("reflection cycles (focal points)", whole["reflection_cycles"],
       cut["reflection_cycles"], "%d")


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("runs", nargs="*", default=["vp9-llm", "vp9-jev"])
  ap.add_argument("--until", type=int, default=rv.END,
                  help="keep the steps before this one (default %d)" % rv.END)
  ap.add_argument("--json", help="write every figure to this file")
  a = ap.parse_args(argv)
  out = {}
  for run in a.runs:
    manifest, rows = run_rows(run)
    cut = [r for r in rows if r["step"] < a.until]
    out[run] = {"steps": manifest["steps"], "until": a.until,
                "whole": window(rows), "window": window(cut),
                "rows": [len(rows), len(cut)]}
    show(run, out[run]["whole"], out[run]["window"], a.until)
  if a.json:
    with open(a.json, "w") as f:
      json.dump(out, f, indent=1, sort_keys=True)


if __name__ == "__main__":
  main()
