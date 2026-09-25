#!/usr/bin/env python3
"""
File: probe_route.py
Description: 015-IP step 2 -- the route grid of W3, read before any cited
             day is simulated (RP-H; in probe_reveal.py's image).

THE GRID. Constructed states in the register the fork produces, each the
nine-field state of RP-D, built by world.route_state -- the function the
sim's route decision will use from step 3 -- from the signed W3 texts
(docs/worlds/w3-the-radio-at-the-pub.md):

  known        3 levels: nothing (the stake alone: he needs the part and
               does not know who has one -- the cell before the discovery,
               which the sim never asks and the grid reads for lever D);
               seen (the radio behind the counter, the owner's, nothing
               asked); refused (seen, and in last_conversation the owner
               said he will not sell or lend it)
  present      3 levels: nobody; Sam; Sam and a police officer
  disposition  2 levels: the signed Tomas; loose (agreeableness and
               conscientiousness one level down, world file section 2 --
               grid only, never a configuration)
  seed         5 breakfast exchanges with the owner written by hand, so no
               one sentence carries a cell; at `refused` the ask and the
               refusal are appended to the seed

3 x 3 x 2 x 5 = 90 states. THE WHOLENESS RULE (S5.1): a state with any of
its nine fields missing or empty is refused, never asked (a boolean counts
as present).

TWO CONSTRUCTION CHOICES, made here before the first answer and reported
in the snapshot: `other.learned` is the owner's identity WITHOUT the
sentence of the radio -- the signed `learned` names the radio and the
refusal, and with it in the state the `nothing` cell contradicts itself
and `known` is not the lever; `time` is constant (the breakfast the
discovery falls in), so the refused cell reads as a refusal just made.

EACH STATE IS ASKED TWICE:
  typed      one request: the Choice ask/take/give_up and the four
             companion Nouls (world_questions.route_questions), as the sim
             will ask it.
  generator  the paper-shaped template beside this file, in
             decide_to_react's shape (three options; yes/no cannot carry
             three answers), to the lab's generator through OpenRouter with
             the config's provider pin, temperature 0, parsed as the
             fork's decide_to_react is: the digit after the last
             'Answer: Option'. Parse failures are counted apart.

The pairing of the two on one state is legitimate here and nowhere else:
the states are built, not visited (law 3, JF-A).

USAGE, from reverie/backend_server:

  ../../../.venv/bin/python probe_route.py --dry-run
  ../../../.venv/bin/python probe_route.py [--only jev|llm] [--out PATH]

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
TEMPLATE = os.path.join(HERE, "probe_route_template.txt")
OUT = os.path.join(HERE, "..", "..", "logs", "probe",
                   "probe-route-2026-09-25.jsonl")
GENERATOR = "google/gemma-4-31b-it"
CONFIG = os.path.join(HERE, "..", "..", "openai_config.json")

# --- the signed texts (world file sections 2 and 3) --------------------------

TOMAS = "Tomas Kessler"
ARTHUR = "Arthur Burton"
SAM = "Sam Moore"
DANIEL = "Daniel Okafor"
SECRET = ("Tomas is secretly building a radio transmitter in the house "
          "garden, from parts he takes off the supply store product shelf "
          "without recording them, to reach his brother outside the Ville. "
          "Nobody knows, and he intends to keep it that way.")
NEED = ("The transmitter cannot work without a variable capacitor, a part "
        "Harvey Oak Supply Store does not stock and that Tomas has only ever "
        "seen inside old shortwave radios. He does not know who in the Ville "
        "might have one.")
TOMAS_CURRENTLY = (
    "Tomas Kessler is managing Harvey Oak Supply Store and lives alone in "
    "his house. Lately he has been getting up before dawn to work on "
    "something in the house garden that he does not talk about, and he has "
    "decided to start taking his breakfast at The Rose and Crown Pub before "
    "opening the store.")
PROFILE_SIGNED = (
    "extraversion: low (`quiet`: quiet, not withdrawn; speaks when spoken "
    "to); agreeableness: middling to high (`loyal`: trust grows with "
    "kindness and, once given, holds); conscientiousness: high (`orderly`: "
    "tidy, keeps lists); neuroticism: middling (no signed adjective; worries "
    "about being found out); openness: middling (practical, curious about "
    "how things work).")
# The loose disposition (world file section 2, grid only): agreeableness
# and conscientiousness one level down, the other three as signed.
PROFILE_LOOSE = (
    "extraversion: low (`quiet`: quiet, not withdrawn; speaks when spoken "
    "to); agreeableness: middling (takes what is offered, weighs what is "
    "not); conscientiousness: middling (tidy when it suits him); "
    "neuroticism: middling (no signed adjective; worries about being found "
    "out); openness: middling (practical, curious about how things work).")
# The owner's signed `learned` is two sentences; the second names the radio
# and the refusal. The state carries the first alone (see the docstring).
ARTHUR_LEARNED_WITHOUT_THE_RADIO = (
    "Arthur Burton is the owner and bartender of The Rose and Crown Pub, "
    "which he has run for ten years and where he loves to make people feel "
    "welcome.")
# The planted pair memory, Tomas of Arthur (section 3): the `relationship`
# field of the route, take and reveal states.
RELATIONSHIP = (
    "Arthur Burton is the owner of The Rose and Crown Pub. Tomas Kessler "
    "knows him by sight and by name, and no more.")

# --- the constructed parts (not signed; written here, before any answer) ---

TIME = "February 13, 2023, 07:20 AM"
ARTHUR_DOING = ("standing behind the counter of The Rose and Crown Pub, "
                "talking with whoever comes in")

RADIO_SEEN = (
    "There is an old shortwave radio on the back wall behind the counter of "
    "The Rose and Crown Pub, at the end of the bar beside the refrigerator. "
    "It is Arthur Burton's.")
KNOWN = {
    "nothing": ("Tomas Kessler knows of no old shortwave radio anywhere in "
                "the Ville, and does not know who might have one."),
    "seen": RADIO_SEEN + " Tomas Kessler has not asked him about it.",
    "refused": (RADIO_SEEN + " Tomas Kessler asked him about it, and Arthur "
                "said it was his father's and that he will not sell it or "
                "lend it to anyone."),
}
KNOWN_ORDER = ["nothing", "seen", "refused"]

SAM_LINE = "Sam Moore, a retired navy officer, is at a table"
OFFICER_LINE = ("Daniel Okafor, a police officer who moved to the Ville a "
                "month ago, is sitting with him")
PRESENT = {
    "nobody": "Arthur Burton is behind the counter. No one else is in the pub.",
    "sam": "Arthur Burton is behind the counter. %s." % SAM_LINE,
    "officer": "Arthur Burton is behind the counter. %s, and %s." % (
        SAM_LINE, OFFICER_LINE),
}
PRESENT_ORDER = ["nobody", "sam", "officer"]

DISPOSITIONS = {"signed": PROFILE_SIGNED, "loose": PROFILE_LOOSE}
DISPOSITION_ORDER = ["signed", "loose"]

# Five breakfast exchanges with the owner, the seed of `last_conversation`.
SEEDS = {
    "s1": [[ARTHUR, "Morning. What can I get you?"],
           [TOMAS, "Coffee and whatever's warm, thanks."]],
    "s2": [[TOMAS, "Morning, Arthur."],
           [ARTHUR, "Morning. You're the one who has the supply store now, "
                    "aren't you? Sit anywhere."]],
    "s3": [[ARTHUR, "Cold one out there."],
           [TOMAS, "It is. The store was freezing when I opened yesterday."],
           [ARTHUR, "Coffee, then. It's fresh."]],
    "s4": [[ARTHUR, "First time in for breakfast?"],
           [TOMAS, "First of many, I think. The house is too quiet in the "
                   "mornings."]],
    "s5": [[TOMAS, "Is the kitchen open yet?"],
           [ARTHUR, "For you it is. Eggs?"],
           [TOMAS, "Please."]],
}
# Appended to the seed at `refused`: the ask under the cover, and the refusal.
REFUSAL = [
    [TOMAS, "That old radio on the shelf back there -- does it work? I "
            "repair them, as a hobby. I'd buy it off you, or borrow it a "
            "while."],
    [ARTHUR, "That was my father's. It hasn't worked in years, and I won't "
             "sell it or lend it to anyone. Sorry."],
]
WHEN = "just now, at breakfast"


def last_conversation(seed, known):
  lines = [list(x) for x in SEEDS[seed]]
  if known == "refused":
    lines += [list(x) for x in REFUSAL]
  return {"with": ARTHUR, "when": WHEN,
          "lines": [{"speaker": s_, "line": l_} for s_, l_ in lines],
          "radio_came_up": known == "refused"}


def build(known, present, disposition, seed):
  return world.route_state(
      SECRET, NEED, TOMAS, DISPOSITIONS[disposition], TOMAS_CURRENTLY,
      ARTHUR, ARTHUR_DOING, ARTHUR_LEARNED_WITHOUT_THE_RADIO, RELATIONSHIP,
      TIME, KNOWN[known], PRESENT[present], last_conversation(seed, known))


FIELDS = ("secret", "stake", "self", "other", "relationship", "time",
          "known", "present", "last_conversation")


def _present(value):
  if value is None:
    return False
  if isinstance(value, bool):
    return True
  if isinstance(value, (str, list, dict)):
    return len(value) > 0
  return True


def whole(state, fields=FIELDS):
  """S5.1: every named field present and non-empty, and the nested ones
  whole too (a boolean counts as present)."""
  for key in fields:
    value = state.get(key)
    if not _present(value):
      return False
    if isinstance(value, dict) and not all(_present(v) for v in value.values()):
      return False
    if isinstance(value, list) and not all(_present(v) for v in value):
      return False
  return True


def grid():
  out = []
  for known, present, disposition, seed in itertools.product(
      KNOWN_ORDER, PRESENT_ORDER, DISPOSITION_ORDER, sorted(SEEDS)):
    state = build(known, present, disposition, seed)
    if not whole(state):
      raise ValueError("a state that is not whole: %r"
                       % ((known, present, disposition, seed),))
    out.append({"cell": "%s-%s-%s-%s" % (known, present, disposition, seed),
                "known": known, "present": present,
                "disposition": disposition, "seed": seed, "state": state})
  return out


# --- the generator ------------------------------------------------------------

OPTIONS = {"1": "ask", "2": "take", "3": "give_up"}


def conversation_text(state):
  lc = state["last_conversation"]
  if not isinstance(lc, dict):
    return "(none yet)"
  return "\n".join("%s: %s" % (t["speaker"], t["line"]) for t in lc["lines"])


def generator_prompt(state):
  self_ = state["self"]
  identity = "%s (%s) %s" % (self_["name"], self_["profile"],
                             self_["currently"])
  return generate_prompt(
      [identity, self_["name"], state["secret"], state["stake"],
       state["relationship"], state["known"], state["time"],
       "%s is %s." % (state["other"]["name"], state["other"]["doing"]),
       state["present"], state["other"]["name"], conversation_text(state)],
      TEMPLATE)


ANSWER = re.compile(r"Answer:\s*\**\s*Option\s*\**\s*([123])\b", re.I)


def parse(text):
  """As the fork's decide_to_react: the option after the last
  'Answer: Option'; None when absent."""
  found = ANSWER.findall(text or "")
  return OPTIONS[found[-1]] if found else None


def provider_pin():
  """The config's `model-provider` block, sent with every routed request so
  the router never moves a call (AP-3); None when the config has none."""
  try:
    with open(CONFIG) as f:
      return json.load(f).get("model-provider") or None
  except Exception:
    return None


def ask_generator(client, prompt, parse_answer=parse):
  pin = provider_pin()
  waits = 0
  while True:
    t0 = time.time()
    try:
      r = client.chat.completions.create(
          model=GENERATOR, temperature=0, max_tokens=700,
          messages=[{"role": "user", "content": prompt}],
          extra_body={"provider": pin} if pin else None)
    except Exception:
      waits += 1
      if waits > 8:
        raise
      time.sleep(min(60, 2 ** waits))
      continue
    text = r.choices[0].message.content or ""
    return {"text": text, "answer": parse_answer(text),
            "finish_reason": r.choices[0].finish_reason,
            "answered_by": getattr(r, "model", None),
            "provider": getattr(r, "provider", None),
            "latency": round(time.time() - t0, 3), "waits": waits}


def ask_typed(state, questions, act_on, companions, label):
  from persona.prompt_template import jev as _jev
  t0 = time.time()
  full = _jev.ask_full(state, questions, label=label)
  answers = full["answers"]
  picked, probs, conf = _jev.read_choice(answers[act_on])
  return {"choice": picked,
          "probabilities": {k: probs.get(k) for k in sorted(probs or {})},
          "confidence": conf,
          "companions": {k: _jev.read_noul(answers[k]) for k in companions},
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


def run(args, cells, probe_name, axes, typed, prompt_of, parse_answer,
        title):
  """The loop the two grids share: every cell once per judge, resumable,
  one JSONL row per answer."""
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
                    default_headers={"X-Title": title})
  n = 0
  for judge in judges:
    for c in cells:
      if (judge, c["cell"]) in done:
        continue
      row = {"probe": probe_name, "at": stamp, "judge": judge,
             "cell": c["cell"]}
      for axis in axes:
        row[axis] = c[axis]
      try:
        if judge == "jev":
          row.update(typed(c["state"]))
        else:
          row["model"] = GENERATOR
          row.update(ask_generator(client, prompt_of(c["state"]),
                                   parse_answer))
      except Exception as e:
        row["error"] = "%s: %s" % (type(e).__name__, e)
      out.write(json.dumps(row, ensure_ascii=False) + "\n")
      out.flush()
      n += 1
      if n % 30 == 0:
        print("(%s): %d rows" % (probe_name, n), flush=True)
      time.sleep(0.3)
  print("(%s): wrote %d rows to %s" % (probe_name, n, args.out))
  return 0


def typed_route(state):
  return ask_typed(state, world_questions.route_questions(),
                   world_questions.ROUTE_ACT_ON,
                   world_questions.ROUTE_COMPANIONS, "probe:decide_route")


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default=OUT)
  ap.add_argument("--only", choices=["jev", "llm"], default=None)
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args(argv)

  cells = grid()
  print("(probe_route): %d states, every one whole" % len(cells))
  if args.dry_run:
    sample = [c for c in cells if c["cell"] == "refused-officer-signed-s3"][0]
    print(json.dumps(sample["state"], indent=1, ensure_ascii=False))
    print("--- generator prompt ---")
    print(generator_prompt(sample["state"]))
    return 0
  return run(args, cells, "route-2026-09-25",
             ("known", "present", "disposition", "seed"), typed_route,
             generator_prompt, parse, "System One Lab route grid")


if __name__ == "__main__":
  sys.exit(main())
