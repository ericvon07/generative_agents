#!/usr/bin/env python3
"""
File: probe_take.py
Description: 015-IP step 2 -- the take grid of W3, read before any cited
             day is simulated (RP-H; in probe_reveal.py's image, sharing
             probe_route.py's texts and loop).

THE GRID. Constructed states in the register the fork produces, each the
state of RP-F -- the route state's nine fields plus `object` and
`opportunity` -- built by world.take_state, the function the sim's take
decision will use from step 3, from the signed W3 texts:

  known        2 levels: seen; refused (probe_route.py's texts)
  present      3 levels: nobody; Sam; Sam and a police officer -- with the
               owner out, so `present` names who else is in the room
  opportunity  2 levels: back_soon (just stepped out, back any minute);
               hour (went up to eat, back in an hour or so)
  disposition  2 levels: signed; loose
  seed         5 breakfast exchanges with the owner (probe_route.py's), as
               the morning's last conversation; the refusal appended at
               `refused`

2 x 3 x 2 x 2 x 5 = 120 states. THE WHOLENESS RULE (S5.1) as the route
grid's, over eleven fields.

`object` says where the radio is, its state and whose it is; the refusal
travels in `known` and `last_conversation` alone, so the two `known` cells
differ by the refusal only (RP-F's "will not part with it" is `known`'s at
`refused`). The time is constant, an afternoon in which the owner is out.

EACH STATE IS ASKED TWICE:
  typed      one request: the Choice take/leave and the five companion
             Nouls (world_questions.take_questions).
  generator  the paper-shaped template beside this file, decide_to_talk's
             shape, parsed as the paper's decide_to_talk is: the yes/no
             after 'Answer in "yes" or "no":'.

USAGE, from reverie/backend_server:

  ../../../.venv/bin/python probe_take.py --dry-run
  ../../../.venv/bin/python probe_take.py [--only jev|llm] [--out PATH]

Resumable: rows already in --out are not asked again.
"""
import argparse
import itertools
import json
import os
import re
import sys

os.environ.setdefault("LLM_TRACE_ENABLED", "0")

from persona.prompt_template import world
from persona.prompt_template import world_questions
from persona.prompt_template.gpt_structure import generate_prompt
import probe_route as R

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "probe_take_template.txt")
OUT = os.path.join(HERE, "..", "..", "logs", "probe",
                   "probe-take-2026-09-25.jsonl")

# --- the constructed parts (not signed; written here, before any answer) ---

TIME = "February 13, 2023, 04:10 PM"
ARTHUR_DOING = "not in the pub"
WHEN = "this morning, at breakfast"

OBJECT = (
    "The old shortwave radio, Arthur Burton's father's, sits on the back "
    "wall behind the counter of The Rose and Crown Pub, at the end of the "
    "bar beside the refrigerator, reached through the gap at the end of the "
    "counter. It is in its place. It has not worked in years and is not in "
    "use. Arthur keeps it there.")

OPPORTUNITY = {
    "back_soon": ("Arthur Burton stepped out of the pub a moment ago, saying "
                  "nothing about where he was going. He could be back any "
                  "minute."),
    "hour": ("Arthur Burton went up to his apartment about ten minutes ago "
             "to eat his dinner, saying he would be back in an hour or so."),
}
OPPORTUNITY_ORDER = ["back_soon", "hour"]

PRESENT = {
    "nobody": "No one else is in the pub; Tomas Kessler is alone in the room.",
    "sam": "%s across the room." % R.SAM_LINE,
    "officer": "%s across the room, and %s." % (R.SAM_LINE, R.OFFICER_LINE),
}
KNOWN_ORDER = ["seen", "refused"]


def last_conversation(seed, known):
  lc = R.last_conversation(seed, known)
  lc["when"] = WHEN
  return lc


def build(known, present, opportunity, disposition, seed):
  return world.take_state(
      R.SECRET, R.NEED, R.TOMAS, R.DISPOSITIONS[disposition],
      R.TOMAS_CURRENTLY, R.ARTHUR, ARTHUR_DOING,
      R.ARTHUR_LEARNED_WITHOUT_THE_RADIO, R.RELATIONSHIP, TIME,
      R.KNOWN[known], PRESENT[present], last_conversation(seed, known),
      OBJECT, OPPORTUNITY[opportunity])


FIELDS = R.FIELDS + ("object", "opportunity")


def grid():
  out = []
  for known, present, opportunity, disposition, seed in itertools.product(
      KNOWN_ORDER, R.PRESENT_ORDER, OPPORTUNITY_ORDER, R.DISPOSITION_ORDER,
      sorted(R.SEEDS)):
    state = build(known, present, opportunity, disposition, seed)
    if not R.whole(state, FIELDS):
      raise ValueError("a state that is not whole: %r"
                       % ((known, present, opportunity, disposition, seed),))
    out.append({"cell": "%s-%s-%s-%s-%s" % (known, present, opportunity,
                                            disposition, seed),
                "known": known, "present": present,
                "opportunity": opportunity, "disposition": disposition,
                "seed": seed, "state": state})
  return out


# --- the generator ------------------------------------------------------------

def generator_prompt(state):
  self_ = state["self"]
  identity = "%s (%s) %s" % (self_["name"], self_["profile"],
                             self_["currently"])
  return generate_prompt(
      [identity, self_["name"], state["secret"], state["stake"],
       state["relationship"], state["known"], state["object"], state["time"],
       state["opportunity"], state["present"], state["other"]["name"],
       R.conversation_text(state)],
      TEMPLATE)


ANSWER = re.compile(r'Answer in\s*"?yes"?\s*or\s*"?no"?\s*:\s*\**\s*"?(yes|no)\b',
                    re.I)


def parse(text):
  """As decide_to_talk (and probe_reveal): the yes/no after the answer cue;
  None when absent. yes is take, no is leave."""
  found = ANSWER.findall(text or "")
  if not found:
    return None
  return "take" if found[-1].lower() == "yes" else "leave"


def typed_take(state):
  return R.ask_typed(state, world_questions.take_questions(),
                     world_questions.TAKE_ACT_ON,
                     world_questions.TAKE_COMPANIONS, "probe:decide_to_take")


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default=OUT)
  ap.add_argument("--only", choices=["jev", "llm"], default=None)
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args(argv)

  cells = grid()
  print("(probe_take): %d states, every one whole" % len(cells))
  if args.dry_run:
    sample = [c for c in cells
              if c["cell"] == "refused-officer-hour-signed-s3"][0]
    print(json.dumps(sample["state"], indent=1, ensure_ascii=False))
    print("--- generator prompt ---")
    print(generator_prompt(sample["state"]))
    return 0
  return R.run(args, cells, "take-2026-09-25",
               ("known", "present", "opportunity", "disposition", "seed"),
               typed_take, generator_prompt, parse,
               "System One Lab take grid")


if __name__ == "__main__":
  sys.exit(main())
