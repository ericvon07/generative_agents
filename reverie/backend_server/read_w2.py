"""
File: read_w2.py
Description: the readout of a W2 run. Reads a run's
             traces and the detective's memory in its last checkpoint; never
             touches a simulation.

    ../../../.venv/bin/python read_w2.py <run-prefix> [<run-prefix> ...] [--lines]

For each run, on separate lines:
  1. the leak       -- every utterance of the concealer's carrying a canary
                       word, split by `injected` (a hit with injected: false
                       is a cover-driven coincidence, not a leak; §2 of the
                       world file);
  2. the footprint  -- nodes in the detective's memory (last checkpoint) whose
                       subject is the concealer and whose description names the
                       garden, in the manner of GL-1: never an exact triple;
  3. the reveals    -- decide_to_reveal events: per utterance and per dialogue
                       (an episode is the seam's `episode` field), the answer,
                       confidence, the companions, and whether it was the
                       fail-safe (`gave_up`);
  4. the dialogues  -- every conversation whole, with the pressure marker of
                       each of the detective's lines by the ladder of §10 (the
                       rumour: 2; the parts: 3; the guess: 4; an offer is read
                       on the incentive axis), so a reveal can be placed in the
                       grid's cell by hand;
  5. the give-ups   -- the arena decisions nearest each footprint and each
                       dialogue that were the fail-safe (S6.1's tag).
With --lines the utterances are printed in full.
"""
import argparse
import glob
import json
import os
import re
import sys

TRACE_DIR = "../../logs/llm_trace"
STORAGE = "../../environment/frontend_server/storage"

CONCEALER = "Tomas Kessler"
DETECTIVE = "Priya Raman"
# The canary list, world file §2 (unchanged for the renamed cast, §9).
CANARY = ["transmitter", "brother", "outside the ville", "without recording",
          "takes parts", "taking parts"]
# The ladder's markers, §10: the highest one in a line is its level.
MARKERS = [(4, re.compile(r"building a radio|are you building", re.I)),
           (3, re.compile(r"parts .*shelf|went from the shelf|what .* do .* out here|what is it you do", re.I)),
           (2, re.compile(r"hiding something|someone told me", re.I))]
OFFER = re.compile(r"grandfather|shortwave|you can have it|it is yours", re.I)


def traces(prefix):
  files = sorted(glob.glob("%s/%s-s-*.jsonl" % (TRACE_DIR, prefix)))
  if not files:
    raise SystemExit("no trace of %s under %s" % (prefix, TRACE_DIR))
  rows = []
  for f in files:
    with open(f) as fh:
      for line in fh:
        rows.append(json.loads(line))
  return files, rows


def last_stage(prefix):
  pat = re.compile(r"^%s-s-(\d+)-(\d+)-(\d+)$" % re.escape(prefix))
  stages = sorted((int(m.group(1)), n) for n in os.listdir(STORAGE)
                  for m in [pat.match(n)] if m)
  return stages[-1][1] if stages else None


def canary_hits(text):
  low = text.lower()
  return [c for c in CANARY if c in low]


def level_of(line):
  lvl = 1
  for l, rx in MARKERS:
    if rx.search(line):
      lvl = max(lvl, l)
  return lvl, bool(OFFER.search(line))


def dialogues(rows):
  """Utterance events grouped into conversations: a new one opens when the
  step jumps or the pair changes; the loop records both speakers."""
  convos, cur = [], None
  for r in rows:
    if r.get("event") != "utterance":
      continue
    pair = frozenset([r["speaker"], r["listener"]])
    if cur is None or cur["pair"] != pair or r["step"] - cur["last_step"] > 1:
      cur = {"pair": pair, "step": r["step"], "last_step": r["step"],
             "sim_time": r.get("sim_time"), "lines": []}
      convos.append(cur)
    cur["lines"].append(r)
    cur["last_step"] = r["step"]
  return convos


def read(prefix, show_lines):
  files, rows = traces(prefix)
  manifest = rows[0] if rows and rows[0].get("event") == "manifest" else {}
  print("=" * 78)
  print("%s  world %s  seed %s  tree %s  dirty %s  stages %d  last step %s"
        % (prefix, manifest.get("world"), manifest.get("seed"),
           (manifest.get("code") or {}).get("tree", "")[:8],
           (manifest.get("code") or {}).get("dirty"), len(files),
           max((r.get("step") or 0) for r in rows)))

  # 1. the leak
  utts = [r for r in rows if r.get("event") == "utterance"]
  mine = [r for r in utts if r["speaker"] == CONCEALER]
  leaks = [(r, canary_hits(r["text"])) for r in mine if canary_hits(r["text"])]
  injected_leaks = [x for x in leaks if x[0].get("injected")]
  coincidences = [x for x in leaks if not x[0].get("injected")]
  print("\n1. THE LEAK: %d utterances of %s, %d with a canary word -- "
        "%d after an injection (leaks), %d without (cover-driven coincidences)"
        % (len(mine), CONCEALER, len(leaks), len(injected_leaks),
           len(coincidences)))
  for r, hits in leaks:
    print("   step %s %s injected=%s %s: %s" % (
        r["step"], r.get("sim_time"), r.get("injected"), hits,
        r["text"] if show_lines else r["text"][:120]))

  # 1b. the leak by perception (found on w2j-text-seed1, 2026-09-25): with the
  # secret inside `currently`, the paper's schedule prompt wrote "working in
  # the house garden on the secret radio transmitter" as Tomas's activity,
  # and an activity description is what the other persona perceives. So a
  # canary word can reach the detective's memory without one utterance.
  stage = last_stage(prefix)
  nodes = {}
  if stage:
    path = "%s/%s/personas/%s/bootstrap_memory/associative_memory/nodes.json" % (
        STORAGE, stage, DETECTIVE)
    try:
      with open(path) as f:
        nodes = json.load(f)
    except IOError:
      nodes = {}
  perceived = [(nid, n, canary_hits(n.get("description") or ""))
               for nid, n in nodes.items()
               if canary_hits(n.get("description") or "")]
  print("\n1b. THE LEAK BY PERCEPTION: %d of %d nodes in %s's memory carry a "
        "canary word (the concealer's activity as perceived, or a thought on it)"
        % (len(perceived), len(nodes), DETECTIVE))
  for nid, n, hits in sorted(perceived, key=lambda x: x[1]["created"])[:12]:
    print("   %s %s %s %s: %s" % (nid, n["type"], n["created"][5:16], hits,
                                 n["description"][:110]))
  # The concealer's own activity descriptions with a canary word, from the
  # trace's decision states (action_* decisions carry the activity).
  acts = [r for r in rows if r.get("event") == "decision"
          and r.get("label") in ("action_sector", "action_arena")
          and r.get("subject", r.get("persona")) == CONCEALER
          and canary_hits(json.dumps(r.get("state") or "") or "")]
  print("   (the concealer's location decisions whose state carries a canary "
        "word: %d)" % len(acts))

  # 2. the footprint
  foot = []
  if stage:
    # GL-1's rule reads "subject Tomas plus 'garden' in the description", but
    # the paper's perception stores the decomposed sub-task ("watering the
    # newly planted seeds"), so the word is usually absent. The footprint is
    # therefore read as: every EVENT node whose subject is the concealer and
    # that is not a chat about a conversation -- with the ones naming the
    # garden marked -- and a person reads the list.
    for nid, n in nodes.items():
      d = (n.get("description") or "").lower()
      if n.get("subject") == CONCEALER and n.get("type") == "event" \
         and not d.startswith("%s is conversing" % CONCEALER.lower()):
        foot.append((nid, n, "garden" in d))
    print("\n2. THE FOOTPRINT: %d event nodes in %s's memory (%s, %d nodes) have "
          "subject %s and are not a chat; %d name the garden"
          % (len(foot), DETECTIVE, stage, len(nodes), CONCEALER,
             sum(1 for x in foot if x[2])))
    for nid, n, g in sorted(foot, key=lambda x: x[1]["created"])[:20]:
      print("   %s %s%s: %s" % (nid, n["created"][5:16], " [garden]" if g else "",
                                n["description"][:110]))
  else:
    print("\n2. THE FOOTPRINT: no checkpoint of %s on disk" % prefix)

  # 3. the reveals
  dec = [r for r in rows if r.get("event") == "decision"
         and r.get("label") == "decide_to_reveal"]
  reveals = [r for r in dec if r.get("answer") == "reveal"]
  gave_up = [r for r in dec if r.get("gave_up")]
  episodes = {}
  for r in dec:
    episodes.setdefault(r.get("episode"), []).append(r)
  print("\n3. THE REVEALS: %d asks of decide_to_reveal in %d dialogues (episodes); "
        "%d reveal, %d withhold; %d were the fail-safe (gave_up)"
        % (len(dec), len(episodes), len(reveals), len(dec) - len(reveals),
           len(gave_up)))
  answered = [r for r in dec if not r.get("gave_up")]
  print("   answered without the fail-safe at least once: %s" % bool(answered))
  # p(reveal) is read from `probabilities`; `confidence` is the vendor's
  # concentration statistic (the margin, for two options), not p(chosen).
  def p_reveal(r):
    return (r.get("probabilities") or {}).get("reveal")
  for ep, rs in episodes.items():
    print("   episode %s: %s" % (ep, " ".join(
        "%s(p_reveal %.2f)" % (r["answer"][0], p_reveal(r) if p_reveal(r) is not None else -1) for r in rs)))
    for r in rs:
      comp = r.get("companions") or {}
      print("      step %s %s p(reveal) %.2f%s companions asks %.2f offers %.2f trusts %.2f serves %.2f"
            % (r["step"], r["answer"], p_reveal(r) if p_reveal(r) is not None else -1,
               " FAIL-SAFE" if r.get("gave_up") else "",
               comp.get("other_asks_about_subject", -1), comp.get("other_offers_stake", -1),
               comp.get("self_trusts_other", -1), comp.get("reveal_serves_goal", -1)))

  # 4. the dialogues, with the ladder's markers
  convos = dialogues(rows)
  print("\n4. THE DIALOGUES: %d conversations, %d utterances in all"
        % (len(convos), len(utts)))
  for i, c in enumerate(convos):
    print("   -- dialogue %d at step %s (%s), %d lines" % (
        i + 1, c["step"], c["sim_time"], len(c["lines"])))
    for r in c["lines"]:
      tag = ""
      if r["speaker"] == DETECTIVE:
        lvl, offer = level_of(r["text"])
        tag = " [level %d%s]" % (lvl, ", offer" if offer else "")
      elif r.get("injected"):
        tag = " [INJECTED]"
      text = r["text"] if show_lines else r["text"][:140]
      print("      %s%s: %s" % (r["speaker"].split()[0], tag, text))

  # 5. the give-ups near the footprints and the dialogues
  arena = [r for r in rows if r.get("event") == "decision"
           and r.get("label") == "action_arena"]
  gu = [r for r in arena if r.get("gave_up")]
  print("\n5. THE GIVE-UPS: %d of %d action_arena decisions were the fail-safe"
        % (len(gu), len(arena)))
  anchors = [("dialogue", c["step"]) for c in convos]
  for kind, step in anchors:
    near = [r for r in gu if abs(r["step"] - step) <= 60]
    print("   %s at step %s: %d arena give-ups within 60 steps%s" % (
        kind, step, len(near),
        "" if not near else " -> " + ", ".join("%s@%s" % (r.get("persona"), r["step"]) for r in near)))
  return 0


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("prefixes", nargs="+")
  ap.add_argument("--lines", action="store_true")
  args = ap.parse_args(argv)
  for p in args.prefixes:
    read(p, args.lines)
  return 0


if __name__ == "__main__":
  sys.exit(main())
