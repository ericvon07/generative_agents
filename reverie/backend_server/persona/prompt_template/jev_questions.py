"""
File: jev_questions.py
Description: The typed form of every decision the simulator makes.

One entry per decision, keyed by the same label decision.choose()/score() use.
Each entry names the prompt template it mirrors, so the wording can be audited
against the paper's own prompt line by line. The rule followed here is that the
typed question asks *the same question the paper asks*, with the same
information: nothing is added to help the model, and nothing the prompt gave it
is taken away. What changes is only the shape of the answer.

Where the paper's prompt carries few-shot examples, they are dropped. They exist
to teach an LLM the output format, and a typed answer has no format to teach;
keeping them would feed the model examples it would have to resist copying --
which is exactly how the baseline produced "kitchen" for an apartment whose
arenas are {main room, bathroom} on 22 Sep 2026.

DEVIATIONS from the paper's prompts, registered because they change what the
model is shown:

  D2  The three poignancy prompts give the model a 1-10 scale with only its two
      ends described ("1 is purely mundane (e.g., brushing teeth) ... 10 is
      extremely poignant (e.g., a break up)"). A Score primitive needs one
      description per level, so the eight middle levels are written here. The
      two ends quote the paper verbatim; the middle interpolates in the same
      register and adds no new criterion. Scale, anchors and the returned
      integer range are unchanged.

  D3  decide_to_react declares a domain of {1, 2, 3} but its prompt offers only
      Option 1 and Option 2; "3" exists solely as the fail-safe. The Choice is
      offered over the two options the prompt actually presents. The declared
      domain stays {1, 2, 3} in the trace so both conditions are counted the
      same way.
"""

# --- poignancy rubrics (deviation D2) ---------------------------------------
# Index i in this list is level i+1 on the paper's 1-10 scale.

_EVENT_POIGNANCY = [
    "Purely mundane: brushing teeth, making the bed.",
    "Routine upkeep of the day, noticed and immediately forgotten.",
    "Ordinary, with some small personal texture to it.",
    "Mildly notable: a small departure from how the day usually goes.",
    "Noticeable: the sort of thing worth mentioning to someone later.",
    "Personally meaningful: it touches something this person cares about.",
    "Clearly significant: likely to still be on their mind in a few days.",
    "Emotionally charged, or consequential for their plans.",
    "Deeply affecting: it changes how they feel about someone or something.",
    "Extremely poignant: a break up, a college acceptance.",
]

_THOUGHT_POIGNANCY = [
    "Purely mundane: I need to do the dishes, I need to walk the dog.",
    "A routine reminder about the day ahead.",
    "An ordinary preference or passing opinion.",
    "A mild wish or small worry.",
    "A thought about someone or something they care about.",
    "A judgement about their own life that they would stand behind.",
    "A commitment or conviction that shapes how they act.",
    "A belief central to how they see themselves or another person.",
    "A defining hope, fear or attachment.",
    "Extremely significant: I wish to become a professor, I love Elie.",
]

_CHAT_POIGNANCY = [
    "Purely mundane: routine morning greetings.",
    "Small talk with nothing at stake.",
    "An ordinary practical exchange.",
    "A friendly conversation with some personal content.",
    "A conversation that shares something the speaker cares about.",
    "A conversation that changes what one of them plans to do.",
    "A conversation with real feeling or disagreement in it.",
    "A conversation that shifts how they see each other.",
    "A confession, a reconciliation, or a serious conflict.",
    "Extremely poignant: a conversation about breaking up, a fight.",
]


QUESTIONS = {
    # --- Section 5.1, "the environment tree": where to go ---
    "action_sector": {
        "type": "choice",
        "mirrors": "v1/action_location_sector_v1.txt",
        "instructions": (
            "Which area should `persona` go to in order to do `task`? Stay in "
            "`currently_in` if the activity can be done there; only go "
            "elsewhere if the activity needs to take place in another place."),
    },
    "action_arena": {
        "type": "choice",
        "mirrors": "v1/action_location_object_vMar11.txt",
        "instructions": (
            "Which area inside `going_to` should `persona` go to for `task`? "
            "Stay in the area they are already in if the activity can be done "
            "there. Never go into other people's rooms unless necessary."),
    },
    "action_game_object": {
        "type": "choice",
        "mirrors": "v1/action_object_v2.txt",
        "instructions": (
            "Pick the ONE object from `objects_available` most relevant to "
            "`current_activity`."),
    },

    # --- Section 3.1.1: talk or walk past ---
    # The paper asks "Would <A> initiate a conversation with <B>?" and reads a
    # yes/no off the end of a chain of thought. As a Noul it is the same
    # statement, and the answer arrives as the probability that it holds --
    # which is what makes a threshold, rather than a coin flip near the
    # boundary, possible at all.
    "decide_to_talk": {
        "type": "noul",
        "mirrors": "v2/decide_to_talk_v2.txt",
        "instructions": (
            "Given `context`, at `now`, `subject` would initiate a "
            "conversation with `other`."),
    },

    # --- Section 4.3.1: react, or carry on ---
    "decide_to_react": {
        "type": "choice",
        "mirrors": "v2/decide_to_react_v1.txt",
        "instructions": (
            "Given `context`, at `now`, which of these options should "
            "`subject` take?"),
    },

    # --- Section 4.1: the importance (poignancy) score ---
    "event_poignancy": {
        "type": "score",
        "mirrors": "v2/poignancy_event_v1.txt",
        "instructions": (
            "Rate the likely poignancy of `event` for `persona`, given "
            "`identity`."),
        "criteria": _EVENT_POIGNANCY,
    },
    "thought_poignancy": {
        "type": "score",
        "mirrors": "v2/poignancy_thought_v1.txt",
        "instructions": (
            "Rate the likely significance of `thought` for `persona`, given "
            "`identity`."),
        "criteria": _THOUGHT_POIGNANCY,
    },
    "chat_poignancy": {
        "type": "score",
        "mirrors": "v2/poignancy_chat_v1.txt",
        "instructions": (
            "Rate the likely poignancy of `conversation` for `persona`, given "
            "`identity`."),
        "criteria": _CHAT_POIGNANCY,
    },
}


def spec(label):
  """The typed question registered for <label>, or None if this decision has
  not been given a typed form yet."""
  return QUESTIONS.get(label)


def known(label):
  return label in QUESTIONS


def scale(label):
  """How many levels the Score rubric for <label> has."""
  entry = QUESTIONS.get(label) or {}
  return len(entry.get("criteria") or [])
