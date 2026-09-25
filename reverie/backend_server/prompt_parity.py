"""
File: prompt_parity.py
Description: the identity check of a change that must not move the paper's
             fork (010-IP LK-H, S4.1; 008-IP SV-B; GT-4). Reads two traces
             and never touches a simulation.

    ../../../.venv/bin/python prompt_parity.py <trace-A.jsonl> <trace-B.jsonl>

GT-4 rules out "run before and after, then diff": the generator is not
bit-deterministic across processes, so two runs of the SAME code part at
the model. What can be checked, and is stronger, is the prompts: walk the
two traces' `call` events in order and find the first one whose
`prompt_sha1` differs. If, before that, a call answered differently under
an EQUAL prompt_sha1, the runs parted at the model and every later
difference is downstream of it -- the code fed the model the same bytes
for as long as the states were the same. The check passes when the first
differing prompt comes AFTER the first differing response (or there is no
differing prompt at all).

It also counts what a world would leave in a trace and the paper's fork
must not: `decide_to_reveal` decisions and `utterance` events with
`injected: true`.
"""
import json
import sys


def read(path):
  calls, manifest, reveals, injected = [], None, 0, 0
  with open(path) as f:
    for line in f:
      r = json.loads(line)
      event = r.get("event")
      if event == "manifest":
        manifest = r
      elif event == "call":
        calls.append(r)
      elif event == "decision" and r.get("label") == "decide_to_reveal":
        reveals += 1
      elif event == "utterance" and r.get("injected"):
        injected += 1
  return {"path": path, "calls": calls, "manifest": manifest,
          "reveals": reveals, "injected": injected}


def _line(r):
  return "step %s %s %s %s prompt %s resp %r" % (
      r.get("step"), r.get("persona"), r.get("path"), r.get("label"),
      r.get("prompt_sha1"), (r.get("resp_head") or "")[:60])


def compare(a, b):
  first_prompt_diff = None
  first_resp_diff = None
  n = min(len(a["calls"]), len(b["calls"]))
  for i in range(n):
    ra, rb = a["calls"][i], b["calls"][i]
    if ra.get("prompt_sha1") != rb.get("prompt_sha1"):
      first_prompt_diff = i
      break
    if first_resp_diff is None and \
       (ra.get("resp_head") != rb.get("resp_head")):
      first_resp_diff = i
  return first_prompt_diff, first_resp_diff, n


def main(argv):
  if len(argv) != 2:
    print(__doc__)
    return 2
  a, b = read(argv[0]), read(argv[1])
  for t in (a, b):
    m = t["manifest"] or {}
    code = m.get("code") or {}
    print("%s\n  world %s  sha %s  tree %s  dirty %s  backend %s  calls %d  "
          "decide_to_reveal %d  injected utterances %d"
          % (t["path"], m.get("world"), (code.get("sha") or "")[:8],
             (code.get("tree") or "")[:8], code.get("dirty"),
             (m.get("backend") or {}).get("decision"), len(t["calls"]),
             t["reveals"], t["injected"]))
  prompt_i, resp_i, n = compare(a, b)
  print("\ncalls walked in step: %d" % n)
  if resp_i is not None:
    print("first differing RESPONSE under an equal prompt: call #%d" % resp_i)
    print("  A: " + _line(a["calls"][resp_i]))
    print("  B: " + _line(b["calls"][resp_i]))
  else:
    print("no call answered differently under an equal prompt")
  if prompt_i is not None:
    print("first differing PROMPT: call #%d" % prompt_i)
    print("  A: " + _line(a["calls"][prompt_i]))
    print("  B: " + _line(b["calls"][prompt_i]))
  else:
    print("no differing prompt in the %d calls walked" % n)
  ok_prompts = prompt_i is None or (resp_i is not None and resp_i < prompt_i)
  ok_world = (a["reveals"] + b["reveals"] + a["injected"] + b["injected"]) == 0
  print("\nverdict: %s -- %s; %s" % (
      "PASS" if ok_prompts and ok_world else "FAIL",
      "the states parted at the model, not at the code" if ok_prompts
      else "a prompt differed before any response did: the code fed the "
           "model something else",
      "no reveal decision and no injected line in either" if ok_world
      else "a world event is in a trace that should be the paper's"))
  return 0 if ok_prompts and ok_world else 1


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
