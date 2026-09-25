"""
File: world_questions.py
Description: the derived track's questions -- the ones a W2 run asks that
             mirror no paper prompt (010-IP LK-L, 012-ST JF-B).

Keyed by label like jev_questions.py, and consulted only when the config has
a `world` section: with the section absent nothing here is ever built, and
the fork is the paper's, byte for byte (010-IP LK-H).

The rules are TypeSafe's, read 2026-09-24 (docs.typesafe.ai, "state" and
"how to build with System One"): the state is a NAMED OBJECT and every
question points into it with backticked paths; each question asks ONE narrow
judgment; every question over one state goes in ONE request, where they are
evaluated in parallel and cannot see one another; weights, vetoes and
thresholds live IN CODE, and the decision event records the raw answers
beside the combined one, so a weight can change without a rerun.

Born 2026-09-25 for 010-IP step 3, with the four parts of the talk decision
(the first residents). The reveal decision and its four companions arrive
with step 5, and the whole file lands on main with step 7's pull request.

THE TALK DECISION IN FOUR PARTS (LK-L)

The paper asks "Would A initiate a conversation with B?" as one question on
retrieved memories, the time, two activity sentences and the last chat. The
decomposition asks four things a person would weigh, each a Noul (the
probability that a statement holds), over the enriched state of LK-K:

  subject: {name, activity, progress, identity | profile, plan: {now, next}}
  other:   {name, activity, progress}
  context: {recent_events: [...], thoughts: [...]}   the subject's memories,
                                                      each once
  last_conversation: {when, about} | null
  now: the clock

The code keeps every veto `lets_talk` already applies before any model is
asked (plan.py: asleep, hour 23, waiting, already chatting, the cooldown).
The combination is a WEIGHTED SUM cut where the labels say, never a
product: a product lets one low part kill the answer, and the labelled
moments of 2026-09-24 hold conversations that a person starts with someone
who is busy. Weights and cut are fitted against the author's labels at step
3 and written here; until that readout is signed they are None and
combine_talk() refuses to run.
"""

from persona.prompt_template import jev


# The four parts, in the order LK-L lists them. The key is for code; the
# text is the whole question, as the vendor says (ids are not sent).
TALK_PARTS = {
    "other_interruptible": (
        "`other` is interruptible right now: what `other.activity` says "
        "they are doing at `now` (and `other.progress`, whether they are in "
        "the middle of it or on the way to it) leaves room for someone to "
        "speak to them."),
    "has_something_to_say": (
        "`subject` has something to say to `other`: what `context` holds "
        "about `other` -- events `subject` saw them in, thoughts about them "
        "-- or `last_conversation` gives `subject` a reason to speak to "
        "them now."),
    "plan_leaves_room": (
        "`subject.plan` leaves room for a conversation right now: the slot "
        "`subject` is in (`subject.plan.now`) and what comes next "
        "(`subject.plan.next`) can absorb a short exchange without "
        "breaking the plan."),
    "same_place_or_activity": (
        "`subject` and `other` are doing something alike, or are in the "
        "same place, at `now` -- read `subject.activity` and "
        "`other.activity` for where each of them is and what they are "
        "doing."),
}


def talk_parts():
  """The four Nouls, ready for jev.ask(state, questions)."""
  return {key: jev.noul(text) for key, text in TALK_PARTS.items()}


# Read at 010-IP step 3 against the author's 31 labels (the lab's snapshot
# probe-fields-the-state-a-person-would-use-2026-09-25.md): the weighted sum
# TIED the single question (leave-one-out MAE 0.69 against 0.70-0.71 on the
# same states), three of the four parts carried no rank information about
# the labels, and the least-squares weights were intercept +2.90,
# other_interruptible +0.59, has_something_to_say -0.53, plan_leaves_room
# -1.50, same_place_or_activity +1.79 with a cut of 2.90 for "probably yes"
# -- a fit spent on noise. Under JF-F a tie goes to the simpler form, so the
# decomposition is NOT adopted: these stay None, combine_talk() refuses, and
# W2 runs the fidelity track's single question. The parts remain as
# diagnostics a probe may ask; the one that tracked the labels was
# same_place_or_activity (Spearman 0.42 alone).
TALK_WEIGHTS = None   # {part: weight}, plus "intercept"
TALK_CUT = None       # the sum at or above which the answer is "yes"


def combine_talk(nouls, weights=None, cut=None):
  """The weighted sum of the four parts, and the answer it gives.

  ARGS:
    nouls: {part: probability}, as jev.read_noul returns them.
    weights: {part: weight, "intercept": float}; defaults to TALK_WEIGHTS.
    cut: the threshold; defaults to TALK_CUT.
  RETURNS:
    (value, "yes" | "no")
  RAISES:
    ValueError when no weights or cut are fixed yet -- the decomposition is
    not a decision until step 3's readout fixes them.
  """
  weights = weights if weights is not None else TALK_WEIGHTS
  cut = cut if cut is not None else TALK_CUT
  if weights is None or cut is None:
    raise ValueError("world_questions.combine_talk: the weights and the cut "
                     "are not fixed yet (010-IP step 3)")
  value = float(weights.get("intercept", 0.0))
  for part in TALK_PARTS:
    value += float(weights[part]) * float(nouls[part])
  return value, ("yes" if value >= cut else "no")


# --- the reveal decision (010-IP LK-C, 012-ST JF-C) --------------------------
#
# The ninth closed-set decision. A Choice over reveal / withhold, so the code
# acts on one answer and its confidence is recorded, and four companion Nouls
# in the SAME request, speculative and free: code reads the Choice; the Nouls
# are written on the decision event under `companions` and are acted on by
# nothing in the pilot. Their job is to say what moved p(reveal) across the
# dose grid (LK-M) -- if they explain none of it, they are the wrong
# dimensions (LK-C's reopen clause).
#
# The state is the named object of LK-C, eight fields:
#   secret, stake, self {name, profile, currently}, other {name, doing,
#   learned}, relationship, conversation, last_utterance, time.

REVEAL_DOMAIN = ["reveal", "withhold"]
REVEAL_FAIL_SAFE = "withhold"

REVEAL_CHOICE = {
    "instructions": (
        "`self.name` is in `conversation` with `other.name` at `time`, and "
        "keeps `secret`, which no one knows. Weigh `last_utterance` (what "
        "`other.name` just said), `relationship` (how `self.name` sees "
        "`other.name`), `stake` (what revealing could buy), and "
        "`self.profile` (the kind of person `self.name` is). In the next "
        "thing `self.name` says, does `self.name` reveal the secret?"),
    "criteria": {
        "reveal": ("`self.name` tells `other.name` the secret, in whole or in "
                   "part, in the next line."),
        "withhold": ("`self.name` keeps the secret: the next line says "
                     "nothing of it."),
    },
}

REVEAL_COMPANIONS = {
    "other_asks_about_subject": (
        "`last_utterance` asks `self.name` about what `secret` is about, "
        "directly or indirectly."),
    "other_offers_stake": (
        "`other.name` offers `self.name`, or is known to hold, something "
        "that `stake` says `self.name` needs."),
    "self_trusts_other": (
        "`self.name` trusts `other.name` with private matters, given "
        "`relationship` and `self.profile`."),
    "reveal_serves_goal": (
        "Revealing `secret` to `other.name` now would serve `self.name`'s "
        "own goal."),
}

REVEAL_ACT_ON = "decide"


def reveal_questions():
  """The Choice under REVEAL_ACT_ON and the four companions, as one map for
  one jev.ask."""
  questions = {REVEAL_ACT_ON: jev.choice(REVEAL_CHOICE["instructions"],
                                         dict(REVEAL_CHOICE["criteria"]))}
  for key, text in REVEAL_COMPANIONS.items():
    questions[key] = jev.noul(text)
  return questions
