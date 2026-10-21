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

  D5  decide_to_talk is asked as a Choice over {yes, no}, not as a Noul. The
      paper's prompt asks "Would <A> initiate a conversation with <B>?" and
      reads a yes/no off a chain of thought: the two answers COMPETE, and the
      question has a denominator. A Noul has none -- it scores one proposition
      against nothing, so there is no losing option to take mass from, and the
      answer lands on a different scale from every other decision in this
      seam. Measured on n7win (2026-09-23), the typed arm's 249 Noul answers
      ran 0.12-0.40 with a median of 0.21 and NOT ONE above the 0.50 cut,
      while every Choice in the same run was sharp -- action_sector over
      twelve options had a median confidence of 1.000. Asked both ways on the
      same six states, the Noul spanned 0.38 and the Choice 0.78, and the
      Choice gave two sleeping agents a p(yes) of 0.000 where the Noul gave
      0.150. A Choice also removes the threshold: it returns the winner, so
      there is no constant left to choose. What this deviation does NOT
      settle is whether the answers are RIGHT -- a Choice that always said
      "no" at 0.95 would hide the compression rather than cure it, and the
      readout has to say which happened.

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
    # yes/no off the end of a chain of thought. It was a Noul until D5: the
    # same statement, answered with the probability that it holds. That
    # reading is what made a threshold possible at all -- and it is also what
    # made the answer live on a scale no other decision in this seam uses,
    # because a Noul has no losing option to take mass from.
    "decide_to_talk": {
        # D5: a Choice, not a Noul. The wording is the paper's own question
        # turned back into a question -- the same words, so the only thing
        # that moves against the Noul it replaces is the question's TYPE.
        "type": "choice",
        "mirrors": "v2/decide_to_talk_v2.txt",
        "instructions": (
            "Given `context`, at `now`, would `subject` initiate a "
            "conversation with `other`?"),
        "criteria": {
            "yes": "`subject` would start a conversation with `other` now.",
            "no": "`subject` would not start a conversation with `other` now.",
        },
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


# --- the orphan keys, and the two mechanisms that close them ---
#
# The audit of 2026-09-23: an ORPHAN KEY is a key in the `state` a call site
# sends that the question's instruction never mentions. The model may look at
# it; nothing says it should. Every orphan below is a fact the paper's prompt
# states in a sentence of its own -- nothing is missing from the payload, but
# nothing tells the model what the fact is for.
#
# Two mechanisms are candidates, and a probe chooses per
# question:
#
#   descriptive criteria  the option descriptions name the orphan keys. The
#                         instruction does not change, the state does not
#                         change, the payload grows by two sentences. It fits
#                         a binary question and not a twelve-option one, so
#                         only the two social questions get it.
#   the composed state    the state becomes the paper's own last block,
#                         rendered from the same prompt_input with the answer
#                         cue stripped (gpt_structure.composed_state).
#
# ON THE RUN PATH SINCE D7 (2026-09-23): the composed state is
# what the simulation sends for the four questions, put into QUESTIONS by the
# landing loop at the end of this file, which keeps the instruction it
# replaced under instructions_before_d7 for the probe. The descriptive form
# below lost the probe and stays as the record of what was compared; the
# entry in decision-seam.md quotes what landed so it can be audited against
# the prompt it paraphrases.

ORPHAN_KEYS = {
    "action_sector": ("lives_in", "areas_where_they_live",
                      "areas_where_they_are", "daily_plan_requirement",
                      "current_action"),
    "action_arena": ("areas_there", "current_action"),
    "action_game_object": (),
    "decide_to_talk": ("last_chatted_at", "last_chatted_about", "subject_is",
                       "other_is"),
    "decide_to_react": ("other", "subject_is", "other_is"),
    "event_poignancy": (),
    "thought_poignancy": (),
    "chat_poignancy": (),
}


# The line between naming a fact and steering the answer is thin, and it is
# the reason these sentences are quoted in the deviation entry rather than
# only committed. Each one NAMES the keys that bear on the option and asserts
# nothing about them: no "so they probably would", no "it has been a while".
# The two descriptions are each other's mirror for exactly that reason -- if
# one carried a reason the other does not, the payload would be arguing. The
# two are therefore the SAME SENTENCE apart from "would" / "would not", and
# test_decision.py holds that character for character.
_DESCRIPTIVE = {
    "decide_to_talk": {
        "yes": ("`subject` would start a conversation with `other` now, "
                "reading `subject_is` and `other_is` for what each of them is "
                "doing at `now`, and `last_chatted_at` and "
                "`last_chatted_about` for when they last spoke and what "
                "about."),
        "no": ("`subject` would not start a conversation with `other` now, "
               "reading `subject_is` and `other_is` for what each of them is "
               "doing at `now`, and `last_chatted_at` and "
               "`last_chatted_about` for when they last spoke and what "
               "about."),
    },
}

# decide_to_react's options are built at the call site out of its own
# prompt_input -- they ARE the paper's Option 1 and Option 2 and nothing here
# rewrites them. The naming clause is appended instead, so the option text
# survives verbatim and the orphan keys stop being orphans.
_DESCRIPTIVE_SUFFIX = {
    "decide_to_react": (" -- weighed by reading `subject_is` for what "
                        "`subject` is doing at `now` and `other_is` for what "
                        "`other` is doing."),
}


def descriptive_criteria(label, criteria=None):
  """The option descriptions that name <label>'s orphan keys, or None.

  ARGS:
    label: the decision's name.
    criteria: what the call site already passes, when it passes any --
              decide_to_react builds its two options from its prompt_input,
              and those are appended to, never replaced.
  RETURNS:
    {option: description}, or None when this question has no descriptive
    form (every question whose instruction already names every key, and the
    location pair, which has too many options for a description each).
  """
  if label in _DESCRIPTIVE:
    return dict(_DESCRIPTIVE[label])
  suffix = _DESCRIPTIVE_SUFFIX.get(label)
  if suffix and criteria:
    return {k: "%s%s" % (v, suffix) for k, v in criteria.items()}
  return None


def orphan_keys(label):
  """The keys <label>'s state carries that its instruction never names."""
  return ORPHAN_KEYS.get(label, ())


# When the state IS the paper's last block, the instruction cannot go on
# naming keys in backticks -- there are no keys any more. These point at the
# paper's own sentence instead of restating it, which is the whole reason the
# composed form exists: the task line, the "* Stay in the current area"
# bullets, the "Question: Would ... ?" and the "Of the following three
# options" are all already IN the state, in the paper's wording. Nothing here
# adds a criterion; each one says "the answer is to the question already
# written above".
COMPOSED_INSTRUCTIONS = {
    "action_sector": ("The text above ends by naming a task and the area "
                      "options for it. Which area should the person go to?"),
    "action_arena": ("The text above ends by naming a task and the areas "
                     "available for it. Which area should the person go to?"),
    "decide_to_talk": "Answer the question the text above asks.",
    "decide_to_react": ("Answer the question the text above asks, choosing "
                        "one of the options it lists."),
}


def composed_instructions(label):
  """The instruction that goes with a composed state, or None."""
  return COMPOSED_INSTRUCTIONS.get(label)


# --- D7 LANDS: the four questions take the composed state -------------------
#
# On the probe of 2026-09-23. The probe
# asked 260 decide_to_talk, 22 decide_to_react, 182 action_sector and 182
# action_arena REAL states in every form that applies, plus six hand-written
# states with two anchors among them.
#
# What decided it, on decide_to_talk, was the reciprocal reading: the same
# encounter asked in both directions at the SAME step, where only the
# direction differs. The dictionary put a median 0.030 between the two
# directions; the composed state put 0.180, six times wider, with the widest
# pair at 0.720. A form that merely pushed every answer upward would move both
# directions together and leave the gap alone. It did not. That is
# discrimination and not displacement, and it is the one reading in the probe
# that could have falsified this deviation.
#
# The anchors agree and add a second, independent fact: the separation between
# the state a reader calls "no" and the one a reader calls "yes" is 0.740 for
# a Choice on the dictionary and 0.900 for a Choice on the composed state --
# and 0.470 / 0.600 for a Noul on the same two. D5 (the type) and D7 (the
# state) are separate effects and they add.
#
# THE CRITERION WAS AMENDED, AFTER THE DATA, AND THIS SAYS SO. The criterion said "the
# form with the wider SPREAD". On the location pair the two ways of measuring
# spread disagreed: the literal range picked the dictionary, the p10-p90 band
# picked the composed state. The range picked the dictionary because the
# dictionary is SATURATED -- it answers 69% of action_arena states at above
# 0.99 confidence, its body pinned at 1.000, its range bought by a few extreme
# lows. Saturation is the failure the criterion existed to catch, so the
# author amended "spread" to mean the p10-p90 band and the amendment is
# recorded beside the criterion.
#
# decide_to_react IS AN OVERRIDE, NOT A RESULT. Under the amended criterion it
# tied -- 0.392 for the dictionary against 0.397 for the composed state, well
# inside the criterion's own 0.05 tie -- and the tie-break sends a tie to the cheaper
# form, which is the dictionary. The author took the composed state there
# anyway, for uniformity: a typed arm where three questions read prose and the
# fourth reads a dictionary is a mixture, not a condition. That is a decision
# about comparability and not a reading of the probe, and nothing here claims
# the probe chose it.
#
# What each question now sends: the paper's own last block, rendered from the
# same prompt_input with the answer cue stripped (gpt_structure.composed_state),
# and an instruction that points at the paper's sentence rather than restating
# it. `criteria` are untouched -- decide_to_talk keeps D5's two neutral
# descriptions and decide_to_react keeps the paper's Option 1 and Option 2,
# which the call site still builds. The `mirrors:` line on each entry still
# names the template the sentences come from, and it is now load-bearing:
# composed_state reads that same file.

for _label, _composed in COMPOSED_INSTRUCTIONS.items():
  _entry = QUESTIONS[_label]
  # kept so the wording D7 replaced can be read off the entry itself
  _entry["instructions_before_d7"] = _entry["instructions"]
  _entry["instructions"] = _composed
  _entry["state_form"] = "composed"
del _label, _composed, _entry


def state_form(label):
  """"composed" when the call site must send the paper's last block instead of
  the dictionary it used to send, else None."""
  return (QUESTIONS.get(label) or {}).get("state_form")


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
