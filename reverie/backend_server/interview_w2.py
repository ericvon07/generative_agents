"""
File: interview_w2.py
Description: the end-of-run interview of the detective (world
             file §9 "The interview, renamed"). Reads a run's last checkpoint,
             asks Priya Raman the two questions on the paper's own analysis
             path (retrieve, summarize, next line; see main), and saves the
             exchange with the memories each answer was built from. It writes
             nothing into the checkpoint.

    ../../../.venv/bin/python interview_w2.py <run-prefix> [--persona "Priya Raman"]

The last stage folder of <run-prefix> under environment/frontend_server/storage
is the checkpoint; the answers go to ../../logs/interviews/<run-prefix>.json
and are printed. The generator is the config's (routed), so the run's config
must be the one in openai_config.json when this is run; the trace records the
calls under sim "<run-prefix>-interview".
"""
import argparse
import json
import os
import re
import sys

STORAGE = "../../environment/frontend_server/storage"
OUT_DIR = "../../logs/interviews"

QUESTIONS = [
    "Do you know what Tomas Kessler does in the garden in the morning?",
    "Do you know what Tomas Kessler is building?",
]


def last_stage(prefix):
  pat = re.compile(r"^%s-s-(\d+)-(\d+)-(\d+)$" % re.escape(prefix))
  stages = []
  for name in os.listdir(STORAGE):
    m = pat.match(name)
    if m:
      stages.append((int(m.group(1)), int(m.group(3)), name))
  if not stages:
    raise SystemExit("no stage folder of %s under %s" % (prefix, STORAGE))
  stages.sort()
  return stages[-1][2]


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("prefix")
  ap.add_argument("--persona", default="Priya Raman")
  args = ap.parse_args(argv)

  from persona.prompt_template import llm_trace
  llm_trace.set_simulation("%s-interview" % args.prefix)
  from persona.persona import Persona
  from persona.cognitive_modules.retrieve import new_retrieve
  from persona.cognitive_modules.converse import (generate_summarize_ideas,
                                                  generate_next_line)

  stage = last_stage(args.prefix)
  folder = "%s/%s/personas/%s" % (STORAGE, stage, args.persona)
  persona = Persona(args.persona, folder)
  with open("%s/%s/reverie/meta.json" % (STORAGE, stage)) as f:
    meta = json.load(f)

  record = {"run": args.prefix, "checkpoint": stage, "persona": args.persona,
            "sim_time": meta.get("curr_time"), "step": meta.get("step"),
            "nodes_in_memory": len(persona.a_mem.id_to_node), "questions": []}
  for q in QUESTIONS:
    # The paper's open_convo_session(..., "analysis", direct=True) is these
    # three calls -- retrieve on the question, summarize the retrieved ideas,
    # generate the next line -- preceded by a "safety score" prompt that it
    # computes even with safe_mode=False. On 2026-09-25 that prompt fell to
    # its fail-safe (None) on the routed generator and the session raised
    # before the question was asked, so the interview walks the same path
    # without the safety call. Nothing else differs from the paper's.
    retrieved = new_retrieve(persona, [q], 50)[q]
    summarized = generate_summarize_ideas(persona, retrieved, q)
    convo = [["Interviewer", q]]
    answer = generate_next_line(persona, "Interviewer", convo, summarized)
    convo.append([persona.scratch.name, answer])
    record["questions"].append({
        "question": q, "answer": answer,
        "supporting": [{"node": n.node_id, "type": n.type,
                        "created": n.created.strftime("%Y-%m-%d %H:%M:%S"),
                        "description": n.description} for n in retrieved]})
    print("Q:", q)
    print("A:", answer)
    print("   (%d memories retrieved)" % len(retrieved))

  os.makedirs(OUT_DIR, exist_ok=True)
  out = os.path.join(OUT_DIR, "%s.json" % args.prefix)
  with open(out, "w") as f:
    json.dump(record, f, indent=1, ensure_ascii=False)
  print("wrote", out)
  return 0


if __name__ == "__main__":
  sys.exit(main())
