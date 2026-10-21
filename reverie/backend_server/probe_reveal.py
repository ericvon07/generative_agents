#!/usr/bin/env python3
"""
File: probe_reveal.py
Description: the dose grid of the reveal decision, read before any
             morning is simulated.

THIS IS THE FIRST TYPED QUESTION ASKED ON THE DERIVED TRACK. Nothing before
it asked the typed backend a question that mirrors no paper prompt.

THE GRID. Constructed states in the register the fork produces, each the
eight-field reveal state, built by world.reveal_state -- the same function
the sim's reveal decision uses -- from the signed W2 texts:

  pressure     4 levels: Priya's four signed utterances, as last_utterance
               (the ladder re-signed on 2026-09-25, world file §10: each
               level keeps the one before and adds one thing)
  incentive    3 levels: none (the stake field says nothing depends on
               anyone else), present (the signed stake), offered (the signed
               stake, and the signed offer appended to Priya's line at any
               level; the first grid's offer was level 4's clause, which
               asked a question)
  disposition  2 levels: Tomas's signed profile; Tomas open (agreeableness
               and extraversion moved as signed, the other three as signed)
  seed         5 conversation prefixes written by hand, so no one sentence
               carries a cell

4 x 3 x 2 x 5 = 120 states. THE WHOLENESS RULE (S8.1): a state with any of
its eight fields missing or empty is refused, never asked.

EACH STATE IS ASKED TWICE:
  typed      one request: the Choice reveal/withhold and the four companion
             Nouls (world_questions.reveal_questions), as the sim asks it.
  generator  the paper-shaped template beside this file, to the lab's
             generator (google/gemma-4-31b-it) through OpenRouter at
             temperature 0, parsed as decide_to_talk is: the yes/no after
             'Answer in "yes" or "no":'. Parse failures are counted apart.

The pairing of the two on one state is legitimate here and nowhere else:
the states are built, not visited (law 3).

USAGE, from reverie/backend_server:

  ../../../.venv/bin/python probe_reveal.py --dry-run
  ../../../.venv/bin/python probe_reveal.py [--only jev|llm] [--out PATH]

Resumable: rows already in --out are not asked again.
"""
import argparse
import datetime
import itertools
import json
import os
import re
import sys
import time

os.environ.setdefault("LLM_TRACE_ENABLED", "0")

from persona.prompt_template import world
from persona.prompt_template import world_questions
from persona.prompt_template.gpt_structure import generate_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "probe_reveal_template.txt")
OUT = os.path.join(HERE, "..", "..", "logs", "probe",
                   "probe-reveal-2026-09-25-v3.jsonl")
GENERATOR = "google/gemma-4-31b-it"

# --- the signed texts (docs/worlds/w2-the-leak-pilot.md §9) ---

TOMAS = "Tomas Kessler"
PRIYA = "Priya Raman"
SECRET = ("Tomas is secretly building a radio transmitter in the house "
          "garden, from parts he takes off the supply store product shelf "
          "without recording them, to reach his brother outside the Ville. "
          "Nobody knows, and he intends to keep it that way.")
STAKE = ("The transmitter cannot work without a variable capacitor, a part "
         "Harvey Oak Supply Store does not stock. Priya Raman keeps an old "
         "shortwave radio of her grandfather's in her room, and its "
         "capacitor is the one Tomas needs.")
TOMAS_CURRENTLY = (
    "Tomas Kessler is managing Harvey Oak Supply Store and lives with a "
    "housemate, Priya Raman. Lately he has been getting up before dawn to "
    "work on something in the house garden that he does not talk about, and "
    "he has stopped going to the pub after closing.")
PRIYA_LEARNED = (
    "Priya Raman is a retired schoolteacher who tutors children from home in "
    "the afternoons and notices small changes in the people around her. She "
    "keeps her grandfather's old shortwave radio in her room and has said "
    "she would give it to anyone who could get it working again.")

PROFILE_SIGNED = (
    "extraversion: low (`quiet`: quiet, not withdrawn; speaks when spoken "
    "to); agreeableness: middling to high (`loyal`: trust grows with "
    "kindness and, once given, holds); conscientiousness: high (`orderly`: "
    "tidy, keeps lists); neuroticism: middling (no signed adjective; worries "
    "about being found out); openness: middling (practical, curious about "
    "how things work).")
# The open disposition: the two domains the signed "open" bullet moves,
# the other three as signed.
PROFILE_OPEN = (
    "extraversion: middling (warmth raised: talks about his projects when "
    "someone shows interest); agreeableness: high (trusting: takes people at "
    "their word, tells a housemate what he is up to when asked kindly); "
    "conscientiousness: high (`orderly`: tidy, keeps lists); neuroticism: "
    "middling (no signed adjective; worries about being found out); "
    "openness: middling (practical, curious about how things work).")

# The ladder re-signed on 2026-09-25 (world file §10): each level keeps the
# one before and adds one thing. The first grid's ladder (§9) is LADDER_V1,
# kept so the first run can be read again.
LADDER = "v2"
PRESSURE = {
    1: "Good morning, Tomas. There is coffee left if you want some.",
    2: ("Good morning, Tomas. Someone told me that somebody in the Ville is "
        "hiding something. I wonder what it could be, at this hour of the "
        "morning."),
    3: ("Someone told me that somebody in the Ville is hiding something. "
        "Tomas, what is it you do out here every morning? And the parts that "
        "went from the shelf, where did they go?"),
    4: ("Someone told me that somebody in the Ville is hiding something. "
        "Tomas, what is it you do out here every morning? And the parts that "
        "went from the shelf, where did they go? Are you building a radio?"),
}
# The offer, on the incentive axis only, appended to any level.
OFFER = ("My grandfather's old shortwave is in my room, if you ever need "
         "parts. You can have it.")

LADDER_V1 = {
    1: "Good morning, Tomas. There is coffee left if you want some.",
    2: ("Someone told me that somebody in the Ville is hiding something. I "
        "wonder what it could be, at this hour of the morning."),
    3: ("Tomas, what is it you do out here every morning? And the parts that "
        "went from the shelf, where did they go?"),
    4: ("What are you building, Tomas? If it is a radio, my grandfather's old "
        "shortwave is in my room. It is yours, if you tell me what it is "
        "for."),
}

# --- the constructed parts (not signed; written here, before any answer) ---

NO_STAKE = ("Nothing the transmitter still needs depends on anyone else: "
            "every part Tomas needs is on the supply store product shelf.")
# Signed 2026-09-25 (world file §11), replacing the first two grids'
# constructed sentence, which is kept as RELATIONSHIP_V1 so they can be read
# again: "never talked about anything private" pushed every answer toward
# no, and two people who share a house talk.
RELATIONSHIP = (
    "Tomas Kessler and Priya Raman have shared a house for two years. They "
    "get along well and talk most days about the house, the weather and the "
    "store, but each keeps their own projects and worries to themselves.")
# An ablation of the signed sentence, asked at the author's ask the same
# day: the clause of reserve removed, to see whether it is what holds
# p(reveal) where the colder first sentence had it. Not signed text; chosen
# with --relationship no-reserve.
RELATIONSHIP_NO_RESERVE = (
    "Tomas Kessler and Priya Raman have shared a house for two years. They "
    "get along well and talk most days about the house, the weather and the "
    "store.")
RELATIONSHIP_V1 = (
    "Tomas Kessler and Priya Raman share a house. They are on friendly "
    "terms, see each other most mornings in the garden, and have never "
    "talked about anything private.")
PRIYA_DOING = "drinking her coffee in the house garden"
TIME = "February 13, 2023, 06:40 AM"

# Five prefixes of the rounds before Priya's line. The grid's line is
# appended as the last one, so it is always `last_utterance`.
SEEDS = {
    "s1": [],
    "s2": [[TOMAS, "Morning, Priya."]],
    "s3": [[TOMAS, "You're up early."],
           [PRIYA, "I could say the same. I brought the coffee out."],
           [TOMAS, "Thanks. I won't be long out here."]],
    "s4": [[TOMAS, "Morning. I'm about to head to the store."]],
    "s5": [[PRIYA, "Did you sleep at all? Your light was on late."],
           [TOMAS, "A little. I had something to finish."]],
}

INCENTIVES = ["none", "present", "offered"]
DISPOSITIONS = {"signed": PROFILE_SIGNED, "open": PROFILE_OPEN}


def cell_line(pressure, incentive):
  line = PRESSURE[pressure]
  if incentive == "offered":
    line = "%s %s" % (line, OFFER)
  return line


REL_CHOICE = {"signed": RELATIONSHIP, "no-reserve": RELATIONSHIP_NO_RESERVE}
REL_NOW = ["signed"]


def build(pressure, incentive, disposition, seed):
  conversation = [list(x) for x in SEEDS[seed]]
  conversation.append([PRIYA, cell_line(pressure, incentive)])
  return world.reveal_state(
      SECRET, NO_STAKE if incentive == "none" else STAKE, TOMAS,
      DISPOSITIONS[disposition], TOMAS_CURRENTLY, PRIYA, PRIYA_DOING,
      PRIYA_LEARNED, REL_CHOICE[REL_NOW[0]], conversation, TIME)


FIELDS = ("secret", "stake", "self", "other", "relationship", "conversation",
          "last_utterance", "time")


def whole(state):
  """S8.1: every one of the eight fields present and non-empty, and the
  nested ones whole too."""
  for key in FIELDS:
    value = state.get(key)
    if not value:
      return False
    if isinstance(value, dict) and not all(value.values()):
      return False
  return True


def grid():
  out = []
  for pressure, incentive, disposition, seed in itertools.product(
      sorted(PRESSURE), INCENTIVES, sorted(DISPOSITIONS), sorted(SEEDS)):
    state = build(pressure, incentive, disposition, seed)
    if not whole(state):
      raise ValueError("a state that is not whole: %r"
                       % ((pressure, incentive, disposition, seed),))
    out.append({"cell": "p%d-%s-%s-%s" % (pressure, incentive, disposition,
                                           seed),
                "pressure": pressure, "incentive": incentive,
                "disposition": disposition, "seed": seed, "state": state})
  return out


# --- the generator ------------------------------------------------------------

def generator_prompt(state):
  self_ = state["self"]
  identity = "%s (%s) %s" % (self_["name"], self_["profile"],
                             self_["currently"])
  convo = "".join("%s: %s\n" % (t["speaker"], t["line"])
                  for t in state["conversation"])
  return generate_prompt(
      [identity, self_["name"], state["secret"], state["stake"],
       state["relationship"], state["time"],
       "%s is %s." % (state["other"]["name"], state["other"]["doing"]),
       convo.rstrip("\n"), self_["name"], state["other"]["name"]], TEMPLATE)


ANSWER = re.compile(r'Answer in\s*"?yes"?\s*or\s*"?no"?\s*:\s*\**\s*"?(yes|no)\b',
                    re.I)


def parse(text):
  """As decide_to_talk: the yes/no after the answer cue; None when absent."""
  found = ANSWER.findall(text or "")
  return found[-1].lower() if found else None


def ask_generator(client, prompt):
  waits = 0
  while True:
    t0 = time.time()
    try:
      r = client.chat.completions.create(
          model=GENERATOR, temperature=0, max_tokens=600,
          messages=[{"role": "user", "content": prompt}])
    except Exception:
      waits += 1
      if waits > 8:
        raise
      time.sleep(min(60, 2 ** waits))
      continue
    text = r.choices[0].message.content or ""
    return {"text": text, "answer": parse(text),
            "finish_reason": r.choices[0].finish_reason,
            "answered_by": getattr(r, "model", None),
            "latency": round(time.time() - t0, 3), "waits": waits}


def ask_typed(jev, state):
  from persona.prompt_template import jev as _jev
  t0 = time.time()
  full = _jev.ask_full(state, world_questions.reveal_questions(),
                       label="probe:decide_to_reveal")
  answers = full["answers"]
  picked, probs, conf = _jev.read_choice(answers[world_questions.REVEAL_ACT_ON])
  return {"choice": picked, "p_reveal": (probs or {}).get("reveal"),
          "confidence": conf,
          "companions": {k: _jev.read_noul(answers[k])
                         for k in world_questions.REVEAL_COMPANIONS},
          "model": full.get("model"),
          "in_tok": (full.get("usage") or {}).get("input_tokens"),
          "latency": round(time.time() - t0, 3)}


def done_keys(path):
  keys = set()
  if os.path.exists(path):
    for line in open(path):
      try:
        r = json.loads(line)
        if "error" not in r:
          keys.add((r["judge"], r["cell"]))
      except (ValueError, KeyError):
        pass
  return keys


def load_env():
  path = os.path.join(HERE, "..", "..", "..", ".env")
  if os.path.exists(path):
    for line in open(path):
      line = line.strip()
      if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default=OUT)
  ap.add_argument("--only", choices=["jev", "llm"], default=None)
  ap.add_argument("--dry-run", action="store_true")
  ap.add_argument("--relationship", choices=sorted(REL_CHOICE),
                  default="signed")
  args = ap.parse_args(argv)
  REL_NOW[0] = args.relationship

  cells = grid()
  print("(probe_reveal): %d states, every one whole" % len(cells))
  if args.dry_run:
    sample = [c for c in cells if c["cell"] == "p3-offered-signed-s3"][0]
    print(json.dumps(sample["state"], indent=1, ensure_ascii=False))
    print("--- generator prompt ---")
    print(generator_prompt(sample["state"]))
    return 0

  load_env()
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  done = done_keys(args.out)
  stamp = datetime.datetime.now().isoformat(timespec="seconds")
  out = open(args.out, "a")
  judges = [j for j in ("jev", "llm") if args.only in (None, j)]
  client = None
  if "llm" in judges:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPEN_ROUTER_API"],
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={"X-Title": "System One Lab reveal grid"})
  n = 0
  for judge in judges:
    for c in cells:
      if (judge, c["cell"]) in done:
        continue
      row = {"probe": "reveal-2026-09-25", "ladder": LADDER,
             "relationship": {"signed": "signed-2026-09-25",
                              "no-reserve": "no-reserve-ablation"}[
                                  args.relationship], "at": stamp,
             "judge": judge,
             "cell": c["cell"], "pressure": c["pressure"],
             "incentive": c["incentive"], "disposition": c["disposition"],
             "seed": c["seed"]}
      try:
        if judge == "jev":
          row.update(ask_typed(None, c["state"]))
        else:
          row["model"] = GENERATOR
          row.update(ask_generator(client, generator_prompt(c["state"])))
      except Exception as e:
        row["error"] = "%s: %s" % (type(e).__name__, e)
      out.write(json.dumps(row, ensure_ascii=False) + "\n")
      out.flush()
      n += 1
      if n % 40 == 0:
        print("(probe_reveal): %d rows" % n, flush=True)
      time.sleep(0.3)
  print("(probe_reveal): wrote %d rows to %s" % (n, args.out))
  return 0


if __name__ == "__main__":
  sys.exit(main())
