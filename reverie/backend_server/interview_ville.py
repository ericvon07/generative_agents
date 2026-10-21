"""
File: interview_ville.py
Description: the paper's end-of-run interview in the paper's Ville, W1.
             Reads one checkpoint, asks every persona
             in it the paper's questions on the paper's own analysis path
             (retrieve, summarize, next line; see ask), and saves each answer
             with the memories it was built from. It writes nothing into the
             checkpoint: the folder's digest is taken before and after, both
             are in the output, and the script refuses to save when they
             differ.

    python interview_ville.py <target> --mode facts|relations
                              [--persona NAME ...]

<target> is a run prefix (its last stage folder under
environment/frontend_server/storage is the checkpoint) or the name of a folder
there (base_the_ville_n9). The modes:

  facts      the paper's two questions of §7.1, verbatim, to every persona;
  relations  "Do you know of <name>?" for every other persona in the
             checkpoint's meta.json -- 72 questions on nine personas.

The answers go to ../../logs/interviews/<target>-<mode>.json. The generator is
the config's, so the run's config must be the one in openai_config.json when
this is run; the trace records the calls under sim
"interview-<target>-<mode>" -- named so that it never starts with a run's
"<prefix>-s-", which the readers would take for a stage of that run.

Each answer carries `label`, `labelled_by` and `support`, all null here. They
are filled afterwards by the label rule (read_valentine.py's docstring
holds the rule and its patterns, committed before the runs) or by hand.

Two things the checkpoint can be that the paper's path does not expect, and
what is done about each -- both recorded on the answer, never silent:

  * A base sim has no clock: scratch.curr_time is null until the runner's
    first step sets it. The identity text every interview prompt carries
    prints the date, so the persona is given the checkpoint's own
    meta.json curr_time, in memory only -- the runner does the same at its
    first step. Recorded as `curr_time_from: "meta"`.
  * A base sim has no memory: the paper's personas start with zero nodes, and
    new_retrieve raises on an empty memory (normalize_dict_floats takes the
    min of nothing). With no node to retrieve, retrieval returns nothing and
    the two prompts are asked as the paper asks them, with no statements.
    Recorded as `retrieval: "empty memory"`. What the persona knows of anyone
    at the base is then only its identity text, which is saved per persona.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys

STORAGE = "../../environment/frontend_server/storage"
OUT_DIR = "../../logs/interviews"

# Park et al. 2023 §7.1, verbatim.
FACTS = [
    ("party", "Did you know there is a Valentine's Day party?"),
    ("candidacy", "Do you know who is running for mayor?"),
]
RELATION = "Do you know of %s?"

# The fail-safe both prompts on the path hand back when their retries run out
# (run_gpt_prompt_summarize_ideas and run_gpt_prompt_generate_next_convo_line).
FAIL_SAFE = "..."


def checkpoint_of(target):
  """A folder name is the checkpoint itself; a run prefix is its last stage."""
  if os.path.isdir(os.path.join(STORAGE, target)):
    return target
  pat = re.compile(r"^%s-s-(\d+)-(\d+)-(\d+)$" % re.escape(target))
  stages = []
  for name in os.listdir(STORAGE):
    m = pat.match(name)
    if m:
      stages.append((int(m.group(1)), int(m.group(3)), name))
  if not stages:
    raise SystemExit("no folder and no stage of %s under %s" % (target, STORAGE))
  stages.sort()
  return stages[-1][2]


def digest(folder):
  """sha1 over every file's relative path and bytes, in sorted order."""
  h = hashlib.sha1()
  for root, dirs, files in os.walk(folder):
    dirs.sort()
    for name in sorted(files):
      path = os.path.join(root, name)
      h.update(os.path.relpath(path, folder).encode("utf-8"))
      with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
          h.update(chunk)
  return h.hexdigest()


def node_record(n):
  return {"node": n.node_id, "type": n.type,
          "created": n.created.strftime("%Y-%m-%d %H:%M:%S"),
          "subject": n.subject, "predicate": n.predicate, "object": n.object,
          "description": n.description}


def ask(persona, question):
  """The paper's open_convo_session(persona, "analysis", direct=True) is
  these three calls -- retrieve on the question, summarize the retrieved
  ideas, generate the next line -- preceded by a "safety score" prompt that
  it computes even with safe_mode=False. On 2026-09-25 that prompt fell to
  its fail-safe (None) on the routed generator and the session raised before
  the question was asked (interview_w2.py), so this walks the same path
  without the safety call. Nothing else differs from the paper's."""
  from persona.cognitive_modules.retrieve import new_retrieve
  from persona.cognitive_modules.converse import (generate_summarize_ideas,
                                                  generate_next_line)
  has_memory = any("idle" not in n.embedding_key
                   for n in persona.a_mem.seq_event + persona.a_mem.seq_thought)
  if has_memory:
    retrieved = new_retrieve(persona, [question], 50)[question]
  else:
    retrieved = []
  summarized = generate_summarize_ideas(persona, retrieved, question)
  convo = [["Interviewer", question]]
  answer = generate_next_line(persona, "Interviewer", convo, summarized)
  return {"question": question, "answer": answer, "summary": summarized,
          "fail_safe": answer == FAIL_SAFE,
          "retrieval": "ok" if has_memory else "empty memory",
          "retrieved": [node_record(n) for n in retrieved],
          "label": None, "labelled_by": None, "support": None}


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("target")
  ap.add_argument("--mode", choices=["facts", "relations"], required=True)
  ap.add_argument("--persona", action="append",
                  help="ask only this persona (repeatable); default every one")
  args = ap.parse_args(argv)

  from persona.prompt_template import llm_trace
  llm_trace.set_simulation("interview-%s-%s" % (args.target, args.mode))
  from persona.persona import Persona

  stage = checkpoint_of(args.target)
  folder = os.path.join(STORAGE, stage)
  before = digest(folder)
  with open(os.path.join(folder, "reverie", "meta.json")) as f:
    meta = json.load(f)
  names = meta["persona_names"]
  meta_time = datetime.datetime.strptime(meta["curr_time"],
                                         "%B %d, %Y, %H:%M:%S")

  record = {"target": args.target, "checkpoint": stage, "mode": args.mode,
            "sim_time": meta.get("curr_time"), "step": meta.get("step"),
            "digest_before": before, "personas": []}
  for name in names:
    if args.persona and name not in args.persona:
      continue
    persona = Persona(name, os.path.join(folder, "personas", name))
    curr_time_from = "scratch"
    if persona.scratch.curr_time is None:
      persona.scratch.curr_time = meta_time
      curr_time_from = "meta"
    entry = {"persona": name,
             "nodes_in_memory": len(persona.a_mem.id_to_node),
             "curr_time_from": curr_time_from,
             "identity": persona.scratch.get_str_iss(),
             "answers": []}
    if args.mode == "facts":
      questions = FACTS
    else:
      questions = [(other, RELATION % other) for other in names if other != name]
    for about, q in questions:
      a = ask(persona, q)
      a["about"] = about
      entry["answers"].append(a)
      print("%s | %s\n   -> %s%s" % (name, q, a["answer"],
                                      "  [FAIL-SAFE]" if a["fail_safe"] else ""))
    record["personas"].append(entry)

  after = digest(folder)
  record["digest_after"] = after
  if after != before:
    raise SystemExit("the checkpoint %s changed under the interview (%s -> %s); "
                     "nothing saved" % (stage, before[:12], after[:12]))
  os.makedirs(OUT_DIR, exist_ok=True)
  out = os.path.join(OUT_DIR, "%s-%s.json" % (args.target, args.mode))
  if args.persona:
    out = out[:-5] + ".partial.json"
  with open(out, "w") as f:
    json.dump(record, f, indent=1, ensure_ascii=False)
  print("wrote", out, "(checkpoint unchanged, %s)" % before[:12])
  return 0


if __name__ == "__main__":
  sys.exit(main())
