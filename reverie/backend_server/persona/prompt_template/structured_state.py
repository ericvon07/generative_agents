"""
File: structured_state.py
Description: the fourth form -- the SAME facts the paper's sentences carry,
             laid out as named JSON fields, the way TypeSafe's own guidance
             says a state should be built.

WHY THIS EXISTS

D7 moved four typed questions from a dictionary of keys to the paper's own
last block, one string, and the probe of 2026-09-23 measured that the string
won. But the dictionary it beat was a poor structure: keys named after the
prompt's slots, five of eight questions carrying keys their instruction never
named, and the paper's rules nowhere. TypeSafe's documentation says the
opposite of what D7 landed -- "use an object for most requests so each part of
the state has a descriptive name", keep judgments out of the state, reference
fields from the instruction with backticked paths. So D7 compared a bad
structure with sentences, and never a good structure with sentences. This
module is the good structure.

THE CONTRACT (law 2)

Every field is taken from the prompt_input the call site already builds, and
nothing else. `to_prompt_input` inverts `build` exactly, so for every state
the probe asks:

    composed_state(to_prompt_input(build(pi)))  ==  composed_state(pi)

character for character. That equality is the proof that this form carries
the same information as the paper's sentences -- no fact dropped, none added.
Where the paper's sentence is only form (the verb rewritten to "was" in the
events, "is already" vs "is on the way to"), the field keeps the words the
sentence used, so the inversion can put them back.

What moves OUT of the state: the paper's rules ("Stay in the current area if
the activity can be done there", "NEVER go into other people's rooms unless
necessary"). They are judgment, not facts, so they go into the instruction,
verbatim -- TypeSafe: "Don't embed questions within state".

What is deliberately NOT done here: naming whose memory `context` is. The
paper's prompt says "Context:" and nothing more; the code knows it is what the
subject retrieved about the other. Saying so is a fact the prompt never
stated, and belongs to the second question (does a richer state decide
better?), not to this one (does a better SHAPE decide better?).
"""

_EVENT_SEP = ". "


# --- small readers ----------------------------------------------------------

def _items(text):
  """'a. b. ' -> ['a', 'b']. The builder appends f'{desc}. ' per node."""
  text = str(text)
  if not text:
    return []
  if not text.endswith(_EVENT_SEP):
    raise ValueError("a context half that does not end in '. ': %r" % text)
  return text[:-len(_EVENT_SEP)].split(_EVENT_SEP)


def _join(items):
  return "".join("%s%s" % (i, _EVENT_SEP) for i in items)


def _context(text):
  """The builder writes events, one '\\n', then thoughts."""
  text = str(text)
  if "\n" not in text:
    raise ValueError("a context with no events/thoughts separator: %r" % text)
  events, thoughts = text.split("\n", 1)
  return {"recent_events": _items(events), "thoughts": _items(thoughts)}


def _context_text(context):
  return "%s\n%s" % (_join(context["recent_events"]),
                     _join(context["thoughts"]))


def _places(text):
  text = str(text)
  return [p for p in text.split(", ")] if text else []


def _places_text(places):
  return ", ".join(places)


# The three ways the social builders say what someone is doing. The label is
# what the model reads; the phrase is what the sentence said.
_PROGRESS = (("already doing it", " is already "),
             ("on the way to do it", " is on the way to "),
             ("doing it now", " is "))


def _doing(sentence, name, activity=None, with_place=False):
  """'Klaus is already reading at desk in library' -> fields.

  decide_to_talk: the activity is whatever follows the phrase.
  decide_to_react: the activity is known (prompt_input 5/7), and what follows
  it is ' at <arena> in <sector>' or a bare ' at '.
  """
  sentence = str(sentence)
  for progress, phrase in _PROGRESS:
    head = "%s%s" % (name, phrase)
    if not sentence.startswith(head):
      continue
    rest = sentence[len(head):]
    if not with_place:
      return {"name": name, "activity": rest, "progress": progress}
    lead = "%s at " % activity
    if not rest.startswith(lead):
      continue
    place = rest[len(lead):]
    return {"name": name, "activity": activity, "progress": progress,
            "place": place or None}
  raise ValueError("cannot read %r as what %s is doing" % (sentence, name))


def _doing_text(who, with_place=False):
  phrase = dict(_PROGRESS)[who["progress"]]
  text = "%s%s%s" % (who["name"], phrase, who["activity"])
  if with_place:
    text += " at %s" % (who["place"] or "")
  return text


# --- the four states --------------------------------------------------------

def build(label, prompt_input, domain=None):
  """The structured state for <label>, from the call site's prompt_input."""
  pi = [str(i) for i in prompt_input]

  if label == "action_sector":
    state = {"person": pi[0],
             "home": {"area": pi[1], "places_inside": _places(pi[2])},
             # null when the tile has no sector -- on a path between places;
             # the paper renders that as "is currently in {} that has ."
             "currently_in": ({"area": pi[4], "places_inside": _places(pi[5])}
                              if pi[4] or pi[5] else None),
             "activity": pi[9],
             "task": pi[10]}
    if pi[6]:
      if not pi[6].startswith("\n"):
        raise ValueError("daily plan requirement without its newline")
      state["daily_plan_requirement"] = pi[6][1:]
    return state

  if label == "action_arena":
    return {"person": pi[0],
            "going_to": {"area": pi[1], "places_inside": _places(pi[2])},
            "activity": pi[4],
            "task": pi[5]}

  if label == "decide_to_talk":
    last = None
    if pi[4] or pi[5]:
      last = {"when": pi[4], "about": pi[5]}
    return {"now": pi[1],
            "subject": _doing(pi[6], pi[2]),
            "other": _doing(pi[7], pi[3]),
            "last_conversation": last,
            "context": _context(pi[0])}

  if label == "decide_to_react":
    return {"now": pi[1],
            "subject": _doing(pi[2], pi[4], pi[5], with_place=True),
            "other": _doing(pi[3], pi[6], pi[7], with_place=True),
            "context": _context(pi[0])}

  raise ValueError("no structured state for %r" % label)


def to_prompt_input(label, state, domain=None):
  """The exact inverse of build. <domain> is needed by action_sector only:
  its area options are the Choice's options, not part of the state."""
  s = state
  if label == "action_sector":
    dpr = s.get("daily_plan_requirement")
    here = s["currently_in"] or {"area": "", "places_inside": []}
    return [s["person"], s["home"]["area"],
            _places_text(s["home"]["places_inside"]),
            s["person"], here["area"], _places_text(here["places_inside"]),
            ("\n%s" % dpr) if dpr is not None else "",
            ", ".join(domain or []),
            s["person"], s["activity"], s["task"], s["person"]]

  if label == "action_arena":
    area = s["going_to"]["area"]
    places = _places_text(s["going_to"]["places_inside"])
    return [s["person"], area, places, s["person"], s["activity"], s["task"],
            s["person"], area, places]

  if label == "decide_to_talk":
    last = s["last_conversation"] or {"when": "", "about": ""}
    return [_context_text(s["context"]), s["now"],
            s["subject"]["name"], s["other"]["name"],
            last["when"], last["about"],
            _doing_text(s["subject"]), _doing_text(s["other"]),
            s["subject"]["name"], s["other"]["name"]]

  if label == "decide_to_react":
    sub, oth = s["subject"], s["other"]
    return [_context_text(s["context"]), s["now"],
            _doing_text(sub, with_place=True),
            _doing_text(oth, with_place=True),
            sub["name"], sub["activity"], oth["name"], oth["activity"],
            sub["activity"]]

  raise ValueError("no structured state for %r" % label)


# --- the instructions -------------------------------------------------------
# Each one names every field the state carries, and carries the paper's rules
# verbatim. Criteria are NOT changed: the comparison with the composed form
# must move one thing, the shape of the state (and the instruction that has
# to point into it).

INSTRUCTIONS = {
    "action_sector": (
        "`person` is `activity`. For `task`, which area should `person` go "
        "to? `home` is where they live and `currently_in` is where they are "
        "now (null when between places), each with the places inside it; `daily_plan_requirement`, when "
        "present, is a standing part of their day. Stay in the current area "
        "if the activity can be done there. Only go out if the activity "
        "needs to take place in another place."),
    "action_arena": (
        "`person` is `activity`. For `task`, which of the places inside "
        "`going_to.area` (`going_to.places_inside`) should `person` go to? "
        "Stay in the current area if the activity can be done there. NEVER "
        "go into other people's rooms unless necessary."),
    "decide_to_talk": (
        "Right now, at `now`, would `subject.name` initiate a conversation "
        "with `other.name`? `subject` and `other` say what each of them is "
        "doing and whether they are already at it or on the way to it; "
        "`last_conversation` is when the two last chatted and about what "
        "(null when none is remembered); `context` holds recent events and thoughts."),
    "decide_to_react": (
        "Right now, at `now`, what should `subject.name` do about "
        "`other.name`? `subject` and `other` say what each of them is doing, "
        "whether they are already at it or on the way to it, and where; "
        "`context` holds recent events and thoughts. Choose one of the "
        "options."),
}


def instructions(label):
  return INSTRUCTIONS.get(label)


# === THE FIFTH FORM: named fields whose values are the paper's sentences =====
#
# The author's objection to the structured form (2026-09-24): it chopped a
# sentence like "Klaus Mueller is on the way to catching up with friends at
# Hobbs Cafe" into name / activity / progress / place, and the relation the
# decision turns on lived in the sentence. So this form keeps the SAME named
# fields and puts the paper's whole sentence in each, verbatim -- braces,
# blanks and all. Rules go to the instruction and options to the criteria,
# exactly as in the structured form, so between "structured" and "sentences"
# the one thing that moves is atoms versus sentences; and between "composed"
# and "sentences" the one thing that moves (besides where the rules sit) is
# whether the sentences come in a container with names.
#
# Same contract: sentences_to_prompt_input inverts build_sentences exactly.


def _raw(half):
  """A context half as a sentence string: 'a. b.' (the builder's 'a. b. ')."""
  return half[:-1] if half.endswith(" ") else half


def _unraw(value):
  return "%s " % value if value else ""


def _sentence_context(text):
  events, thoughts = str(text).split("\n", 1) if "\n" in str(text) else (None, None)
  if events is None:
    raise ValueError("a context with no events/thoughts separator: %r" % text)
  for half in (events, thoughts):
    if half and not half.endswith(". "):
      raise ValueError("a context half that does not end in '. ': %r" % half)
  return {"recent_events": _raw(events), "thoughts": _raw(thoughts)}


def _sentence_context_text(context):
  return "%s\n%s" % (_unraw(context["recent_events"]),
                     _unraw(context["thoughts"]))


def _cut(text, prefix, suffix=""):
  if not text.startswith(prefix) or not text.endswith(suffix):
    raise ValueError("%r is not %r...%r" % (text, prefix, suffix))
  return text[len(prefix):len(text) - len(suffix)]


def build_sentences(label, prompt_input, domain=None):
  pi = [str(i) for i in prompt_input]

  if label == "action_sector":
    state = {"person": pi[0],
             "home": "%s lives in {%s} that has %s." % (pi[0], pi[1], pi[2]),
             "currently_in": "%s is currently in {%s} that has %s."
                             % (pi[3], pi[4], pi[5]),
             "activity": "%s is %s." % (pi[8], pi[9]),
             "task": pi[10]}
    if pi[6]:
      if not pi[6].startswith("\n"):
        raise ValueError("daily plan requirement without its newline")
      state["daily_plan_requirement"] = pi[6][1:]
    return state

  if label == "action_arena":
    return {"person": pi[0],
            "going_to": "%s is going to %s that has the following areas: {%s}"
                        % (pi[0], pi[1], pi[2]),
            "activity": "%s is %s." % (pi[3], pi[4]),
            "task": pi[5]}

  if label == "decide_to_talk":
    return {"now": "Right now, it is %s." % pi[1],
            "subject": {"name": pi[2], "doing": pi[6]},
            "other": {"name": pi[3], "doing": pi[7]},
            "last_conversation": "%s and %s last chatted at %s about %s."
                                 % (pi[2], pi[3], pi[4], pi[5]),
            "context": _sentence_context(pi[0])}

  if label == "decide_to_react":
    return {"now": "Right now, it is %s." % pi[1],
            "subject": {"name": pi[4], "doing": pi[2]},
            "other": {"name": pi[6], "doing": pi[3]},
            "context": _sentence_context(pi[0])}

  raise ValueError("no sentences state for %r" % label)


def _react_activity(sentence, name):
  for _, phrase in _PROGRESS[:2]:
    head = "%s%s" % (name, phrase)
    if sentence.startswith(head):
      return sentence[len(head):].rsplit(" at ", 1)[0]
  raise ValueError("cannot read an activity out of %r" % sentence)


def sentences_to_prompt_input(label, state, domain=None):
  s = state
  if label == "action_sector":
    p = s["person"]
    home_area, home_places = _cut(s["home"], "%s lives in {" % p,
                                  ".").split("} that has ", 1)
    here_area, here_places = _cut(s["currently_in"],
                                  "%s is currently in {" % p,
                                  ".").split("} that has ", 1)
    dpr = s.get("daily_plan_requirement")
    return [p, home_area, home_places, p, here_area, here_places,
            ("\n%s" % dpr) if dpr is not None else "", ", ".join(domain or []),
            p, _cut(s["activity"], "%s is " % p, "."), s["task"], p]

  if label == "action_arena":
    p = s["person"]
    area, places = _cut(s["going_to"], "%s is going to " % p,
                        "}").split(" that has the following areas: {", 1)
    return [p, area, places, p, _cut(s["activity"], "%s is " % p, "."),
            s["task"], p, area, places]

  if label == "decide_to_talk":
    a, b = s["subject"]["name"], s["other"]["name"]
    when, about = _cut(s["last_conversation"],
                       "%s and %s last chatted at " % (a, b),
                       ".").split(" about ", 1)
    return [_sentence_context_text(s["context"]),
            _cut(s["now"], "Right now, it is ", "."), a, b, when, about,
            s["subject"]["doing"], s["other"]["doing"], a, b]

  if label == "decide_to_react":
    sub, oth = s["subject"], s["other"]
    sub_act = _react_activity(sub["doing"], sub["name"])
    return [_sentence_context_text(s["context"]),
            _cut(s["now"], "Right now, it is ", "."),
            sub["doing"], oth["doing"], sub["name"], sub_act, oth["name"],
            _react_activity(oth["doing"], oth["name"]), sub_act]

  raise ValueError("no sentences state for %r" % label)


SENTENCE_INSTRUCTIONS = {
    "action_sector": (
        "`activity` says what `person` is doing. For `task`, which area "
        "should `person` go to? `home` says where they live and "
        "`currently_in` where they are now, each with the places inside it; "
        "`daily_plan_requirement`, when present, is a standing part of their "
        "day. Stay in the current area if the activity can be done there. "
        "Only go out if the activity needs to take place in another place."),
    "action_arena": (
        "`activity` says what `person` is doing. For `task`, which of the "
        "areas named in `going_to` should `person` go to? Stay in the "
        "current area if the activity can be done there. NEVER go into "
        "other people's rooms unless necessary."),
    "decide_to_talk": (
        "`now` is the time. Would `subject.name` initiate a conversation "
        "with `other.name`? `subject.doing` and `other.doing` say what each "
        "of them is doing; `last_conversation` says when the two last "
        "chatted and about what; `context` holds recent events and "
        "thoughts."),
    "decide_to_react": (
        "`now` is the time. What should `subject.name` do? `subject.doing` "
        "and `other.doing` say what each of them is doing and where; "
        "`context` holds recent events and thoughts. Choose one of the "
        "options."),
}


def sentence_instructions(label):
  return SENTENCE_INSTRUCTIONS.get(label)


# === THE QUESTION MOVED OUT: the composed state without its question =========
#
# Named fields lost to the composed state whether their values were atoms or
# whole sentences. One of the differences left is WHERE THE QUESTION SITS: the
# composed state ends with the paper's own question and its instruction only
# says "answer the question the text above asks", while every fielded form put
# the question in the instruction, as TypeSafe's guidance asks. This form is
# the composed state, character for character, minus the question sentence,
# which becomes the instruction verbatim. Social pair only: there the question
# is a sentence of its own; on the location pair it is fused with the task.

_QUESTION_CUTS = {"decide_to_talk": "\n\nQuestion: ",
                  "decide_to_react": "\nMy question: "}


def question_out(label, composed):
  """(state, instructions) -- the composed text split at its question."""
  cut = _QUESTION_CUTS.get(label)
  if cut is None or cut not in composed:
    raise ValueError("no question to move out of a %s state" % label)
  state, rest = composed.split(cut, 1)
  if label == "decide_to_talk":
    return state, rest.strip()
  # decide_to_react: the question is the first line; the options stay put,
  # as they do in the composed state.
  question, options = rest.split("\n", 1)
  return "%s\n%s" % (state, options), question.strip()
