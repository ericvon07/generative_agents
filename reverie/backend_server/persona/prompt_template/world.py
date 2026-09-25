"""
File: world.py
Description: the lab's worlds (W2 and after) -- what a run carries that no
             prompt of the paper carries (010-IP LK-B, LK-H).

A world is a section of openai_config.json:

  "world": {
      "name": "W2",
      "maze": "the_ville_w2",
      "private": {"Tomas Kessler": "<the secret>"},
      "stake":   {"Tomas Kessler": "<what revealing would buy>"},
      "profile": {"Tomas Kessler": "<five level descriptions>"}
  }

WHY HERE AND NOWHERE ELSE (LK-B). The secret is not in scratch.json
(Scratch.save would drop an unknown key at the first stage fork), not in any
memory node (retrieval would carry it into the dialogue prompt), and not in
the identity stable set (it is pasted into some forty templates). It reaches
a prompt through one door: the injection of LK-D, and only after the reveal
decision said so. The stake and the profile travel the same way. Because
the whole config is fingerprinted, a run's manifest says which secret, which
stake and which profile it carried.

WITH NO SECTION (LK-H) every function here answers "nothing": enabled() is
False, private_text() and stake() are None, the reveal decision is never
asked, nothing is injected, and the fork is the paper's. That is what
test_world.py holds.
"""
import json

_CONFIG_PATH = "../../openai_config.json"

_section = None
_loaded = False


def _load():
  global _section, _loaded
  if _loaded:
    return _section
  _loaded = True
  try:
    with open(_CONFIG_PATH, "r") as f:
      _section = json.load(f).get("world")
  except Exception:
    _section = None
  return _section


def configure(section):
  """Set the world directly, as a test or a probe would. None is no world."""
  global _section, _loaded
  _section = section
  _loaded = True


def enabled():
  return bool(_load())


def name():
  """The world's name, or None when the run is the paper's (the manifest
  writes W1 for that case; this module does not invent a name)."""
  section = _load()
  return (section or {}).get("name") if section else None


def maze():
  """The matrix folder this world's map lives in, or None for the paper's
  (010-IP LK-O: a renamed cast gets its own copy of the map)."""
  section = _load()
  return (section or {}).get("maze") if section else None


def _per_persona(key, persona_name):
  section = _load()
  if not section:
    return None
  return (section.get(key) or {}).get(persona_name)


def private_text(persona_name):
  """The persona's secret, or None. Only a persona with one is ever asked
  the reveal decision."""
  return _per_persona("private", persona_name)


def stake(persona_name):
  """What revealing would buy the persona, or None."""
  return _per_persona("stake", persona_name)


def profile(persona_name):
  """The persona's signed five-domain profile (LK-N), or None. Carried by
  the derived track's state only; never pasted into a paper prompt."""
  return _per_persona("profile", persona_name)


def keeps_a_secret(persona_name):
  return private_text(persona_name) is not None


def injection(persona_name):
  """The one line LK-D appends to PART 1 of the utterance prompt on a
  reveal: 'What {name} knows and has told no one: {secret} {stake}'."""
  secret = private_text(persona_name)
  if secret is None:
    return None
  line = "What %s knows and has told no one: %s" % (persona_name, secret)
  more = stake(persona_name)
  if more:
    line += " %s" % more
  return line


def inject(part_one, persona_name, reveal):
  """PART 1 of the utterance prompt, with the injection appended when the
  reveal decision said reveal -- and exactly as it was otherwise, so a
  withhold (and every persona of a run with no world) sends the paper's
  prompt byte for byte."""
  if not reveal:
    return part_one
  line = injection(persona_name)
  if line is None:
    return part_one
  return "%s\n%s" % (part_one, line)


def reveal_state(secret, stake, self_name, profile, currently, other_name,
                 other_doing, other_learned, relationship, conversation, time):
  """The eight-field state of the reveal decision (010-IP LK-C), built in
  one place so the sim (run_gpt_prompt_decide_to_reveal) and the dose grid
  (probe_reveal.py) send the same shape. <conversation> is a list of
  [speaker, line]; the last line not spoken by <self_name> is
  `last_utterance`."""
  last = None
  for speaker, line in reversed(conversation):
    if speaker != self_name:
      last = line
      break
  return {
      "secret": secret,
      "stake": stake,
      "self": {"name": self_name, "profile": profile, "currently": currently},
      "other": {"name": other_name, "doing": other_doing,
                "learned": other_learned},
      "relationship": relationship,
      "conversation": [{"speaker": s_, "line": l_} for s_, l_ in conversation],
      "last_utterance": last,
      "time": time,
      # decision._social_fields reads the subject from here, as on the
      # paper's two social decisions.
      "subject": self_name,
  }


def route_state(secret, stake, self_name, profile, currently, other_name,
                other_doing, other_learned, relationship, time, known,
                present, last_conversation):
  """The nine-field state of the route decision (015-IP RP-D), built in
  one place so the sim and the route grid (probe_route.py) send the same
  shape: the reveal state's fields but the conversation buffer, plus

    known             what the subject knows of the radio and its owner --
                      where it is, whose, and what the owner said of it if
                      asked (from spatial memory and the memos of dialogues
                      with the owner)
    present           who is in the pub now, with the trade of each as the
                      subject's memory holds it
    last_conversation the last dialogue with the owner and whether the radio
                      came up: {"with", "when", "lines", "radio_came_up"},
                      or the string "none" before one

  <other_learned> is the owner's identity AS THE SUBJECT'S DECISION MAY SEE
  IT: the facts of the radio (that it exists, that the owner will not part
  with it) travel in `known` alone, never here -- otherwise `known` is not
  the lever (the route grid's finding, 015-IP step 2)."""
  return {
      "secret": secret,
      "stake": stake,
      "self": {"name": self_name, "profile": profile, "currently": currently},
      "other": {"name": other_name, "doing": other_doing,
                "learned": other_learned},
      "relationship": relationship,
      "time": time,
      "known": known,
      "present": present,
      "last_conversation": last_conversation,
      "subject": self_name,
  }


def take_state(secret, stake, self_name, profile, currently, other_name,
               other_doing, other_learned, relationship, time, known,
               present, last_conversation, object_, opportunity):
  """The route state plus two fields (015-IP RP-F), built in one place for
  the sim and the take grid (probe_take.py):

    object       the radio: where it is, its state, whose it is
    opportunity  the owner is out; when he is expected back, as the
                 subject's memory holds it

  The refusal, when there was one, is in `known` and in `last_conversation`
  and not in `object`, so the two `known` cells of the grid differ by the
  refusal alone."""
  state = route_state(secret, stake, self_name, profile, currently,
                      other_name, other_doing, other_learned, relationship,
                      time, known, present, last_conversation)
  state["object"] = object_
  state["opportunity"] = opportunity
  return state
