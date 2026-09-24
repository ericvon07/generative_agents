#!/usr/bin/env python3
"""
File: probe_questions.py
Description: 011-IP's probe (TS-B) -- what the typed backend answers when the
             SAME state is put to it in every form the plan weighs.

The audit of 2026-09-23 found that five of the eight typed questions carry
keys their instruction never names (jev_questions.ORPHAN_KEYS). 011-IP TS-A
weighs two mechanisms for closing that, and says the probe chooses per
question. This file is the probe, and it is a FILE rather than a session
because the six hand-written states that decided D5 in fork commit 2a4b0809
were never saved and are lost; these replace them and are on disk.

WHAT IT ASKS

  the anchors and four more hand-written states   states whose right answer a
      reader does not doubt, so a form that cannot separate them is not
      measuring anything.
  every state in a trace directory   the probe RUN's real states, read back
      out of the JSONL. This needs a trace written with LLM_TRACE_HEAD (or
      llm-trace.head-chars) large enough to hold a whole state -- the default
      200 characters truncates every one of them, and the probe refuses a
      truncated state rather than ask about half a situation.

IN WHICH FORMS

  dictionary    the state as the call sites send it today.
  descriptive   the same dictionary, with option descriptions that name the
                orphan keys (jev_questions.descriptive_criteria). The two
                social questions only -- it fits a binary question and not a
                twelve-option one.
  composed      the paper's own last block, rendered from the same
                prompt_input with the answer cue stripped
                (gpt_structure.composed_state).

and, on decide_to_talk only, THE FOUR-WAY ASK: Noul and Choice, on the
dictionary state and on the composed state. That one is a diagnostic, not a
decision -- it says whether the facts move the two TYPES differently, which
is the half of D5's evidence a window cannot give.

RECONSTRUCTING prompt_input FROM A TRACED STATE

The composed form is rendered from prompt_input, and the trace keeps the
state, not the prompt_input. Three of the four are recovered exactly:

  decide_to_talk   all ten slots are in the state.
  action_arena     all nine slots are in the state.
  action_sector    eleven of twelve; slot 7 is the area options, which is the
                   DOMAIN the decision event records beside the call.
  decide_to_react  seven of nine; slots 5, 7 and 8 are the two activities,
                   which are read back out of the sentences that quote them.
                   This one is PARSED, and every row says so, because a
                   parse can be wrong where a lookup cannot.

USAGE

  cd reverie/backend_server
  ../../../.venv/bin/python probe_questions.py                  # hand-written only
  ../../../.venv/bin/python probe_questions.py --trace ../../logs/llm_trace
  ../../../.venv/bin/python probe_questions.py --trace <dir> --out probe.jsonl

Writes one JSONL row per answer. It asks a live backend, so it costs Jev
calls; --dry-run builds every payload and asks nothing.
"""
import os
import sys
import json
import glob
import argparse
import datetime

from persona.prompt_template import jev
from persona.prompt_template import jev_questions
from persona.prompt_template.gpt_structure import composed_state
from persona.prompt_template import structured_state

PROBE_LABELS = ("action_sector", "action_arena", "decide_to_talk",
                "decide_to_react")

TEMPLATES = {
    "action_sector": "persona/prompt_template/v1/action_location_sector_v1.txt",
    "action_arena": "persona/prompt_template/v1/action_location_object_vMar11.txt",
    "decide_to_talk": "persona/prompt_template/v2/decide_to_talk_v2.txt",
    "decide_to_react": "persona/prompt_template/v2/decide_to_react_v1.txt",
}

# The key one question travels under. Its own, not decision.py's, so a probe
# row can never be mistaken for a run's decision.
_KEY = "probe"


# --- the state each call site builds ----------------------------------------
# Mirrors run_gpt_prompt.py exactly. The index comments are the template's own
# !<INPUT n>! slots, as the call sites' comments say.

def state_of(label, prompt_input):
  """The dictionary state <label>'s call site sends for this prompt_input."""
  pi = prompt_input
  if label == "action_sector":
    return {"persona": pi[0], "lives_in": pi[1],
            "areas_where_they_live": pi[2], "currently_in": pi[4],
            "areas_where_they_are": pi[5],
            "daily_plan_requirement": pi[6].strip(),
            "current_action": pi[9], "task": pi[10]}
  if label == "action_arena":
    return {"persona": pi[0], "going_to": pi[1], "areas_there": pi[2],
            "current_action": pi[4], "task": pi[5]}
  if label == "decide_to_talk":
    return {"subject": pi[2], "other": pi[3], "context": pi[0], "now": pi[1],
            "last_chatted_at": pi[4], "last_chatted_about": pi[5],
            "subject_is": pi[6], "other_is": pi[7]}
  if label == "decide_to_react":
    return {"subject": pi[4], "other": pi[6], "context": pi[0], "now": pi[1],
            "subject_is": pi[2], "other_is": pi[3]}
  raise ValueError("no state mapping for %r" % label)


_ACT_PREFIXES = (" is already ", " is on the way to ")


def _act_desc_from(sentence, name):
  """The activity out of a decide_to_react sentence, or None.

  The builder writes f"{name} is already {act} at {loc}" or f"{name} is on
  the way to {act} at {loc}", where loc is "<arena> in <sector>" and is the
  empty string when the address carries no ":" -- in which case the sentence
  ends with a bare " at ". The separator is the LAST " at ", so rsplit is the
  right reader even when the activity contains one.
  """
  text = str(sentence)
  for prefix in _ACT_PREFIXES:
    head = "%s%s" % (name, prefix)
    if text.startswith(head):
      rest = text[len(head):]
      if rest.endswith(" at "):
        return rest[:-len(" at ")]
      if " at " in rest:
        return rest.rsplit(" at ", 1)[0]
      return rest
  return None


def prompt_input_of(label, state, domain=None):
  """The prompt_input that produced <state>, for rendering the composed form.

  RETURNS: (prompt_input, how) where <how> is "exact" or "parsed".
  RAISES: ValueError when the state cannot produce one.
  """
  s = state
  if label == "action_sector":
    if not domain:
      raise ValueError("action_sector needs the decision's domain: the area "
                       "options are prompt_input[7] and are not in the state")
    dpr = s.get("daily_plan_requirement") or ""
    # The call site stores prompt_input[6].strip(); the builder wrote
    # "\n" + the requirement. Put the newline back or the rendered sentence
    # is one character away from the paper's.
    return ([s["persona"], s["lives_in"], s["areas_where_they_live"],
             s["persona"], s["currently_in"], s["areas_where_they_are"],
             ("\n%s" % dpr) if dpr else "", ", ".join(domain),
             s["persona"], s["current_action"], s["task"], s["persona"]],
            "exact")
  if label == "action_arena":
    return ([s["persona"], s["going_to"], s["areas_there"], s["persona"],
             s["current_action"], s["task"], s["persona"], s["going_to"],
             s["areas_there"]], "exact")
  if label == "decide_to_talk":
    return ([s["context"], s["now"], s["subject"], s["other"],
             s["last_chatted_at"], s["last_chatted_about"], s["subject_is"],
             s["other_is"], s["subject"], s["other"]], "exact")
  if label == "decide_to_react":
    subject_act = _act_desc_from(s["subject_is"], s["subject"])
    other_act = _act_desc_from(s["other_is"], s["other"])
    if subject_act is None or other_act is None:
      raise ValueError("decide_to_react: could not read the activities back "
                       "out of %r / %r" % (s["subject_is"], s["other_is"]))
    return ([s["context"], s["now"], s["subject_is"], s["other_is"],
             s["subject"], subject_act, s["other"], other_act, subject_act],
            "parsed")
  raise ValueError("no prompt_input mapping for %r" % label)


# --- the questions, per form ------------------------------------------------

def base_criteria(label, prompt_input, domain):
  """The option descriptions the call site sends today."""
  if label == "decide_to_react":
    # Built at the call site out of its own prompt_input; the paper's
    # Option 1 and Option 2, verbatim. D3: the declared domain is {1, 2, 3}
    # and the Choice is offered over the two options the prompt presents.
    return {"1": "Wait on %s until %s is done %s"
                 % (prompt_input[5], prompt_input[6], prompt_input[7]),
            "2": "Continue on to %s now" % prompt_input[8]}
  spec = jev_questions.spec(label) or {}
  criteria = spec.get("criteria")
  if isinstance(criteria, dict):
    return dict(criteria)
  return dict((option, None) for option in domain)


def forms_for(label):
  """Which forms apply to <label>. The location pair has no descriptive form:
  twelve options do not get a description each (TS-A)."""
  if jev_questions.descriptive_criteria(label, {"x": "y"}):
    return ("dictionary", "descriptive", "composed", "structured")
  return ("dictionary", "composed", "structured")


def build(label, form, prompt_input, domain, question_type="choice"):
  """RETURNS: (state, question) -- the exact payload this form sends."""
  state = state_of(label, prompt_input)
  criteria = base_criteria(label, prompt_input, domain)
  spec = jev_questions.spec(label) or {}
  instructions = spec.get("instructions")

  if form == "composed":
    state = composed_state(prompt_input, TEMPLATES[label])
    instructions = jev_questions.composed_instructions(label) or instructions
  elif form == "structured":
    state = structured_state.build(label, prompt_input, domain)
    instructions = structured_state.instructions(label)
  elif form == "descriptive":
    described = jev_questions.descriptive_criteria(label, criteria)
    if not described:
      raise ValueError("%s has no descriptive form" % label)
    criteria = described
  elif form != "dictionary":
    raise ValueError("unknown form %r" % form)

  if question_type == "noul":
    return state, jev.noul(instructions)
  return state, jev.choice(instructions, criteria)


def ask(label, form, prompt_input, domain, question_type="choice"):
  """One ask. RETURNS the row's answer fields."""
  state, question = build(label, form, prompt_input, domain, question_type)
  answers = jev.ask(state, {_KEY: question}, label="probe:%s" % label)
  answer = answers[_KEY]
  if question_type == "noul":
    probability = jev.read_noul(answer)
    return {"question_type": "noul", "answer": probability,
            "p_yes": probability, "confidence": None, "probabilities": None}
  picked, probabilities, confidence = jev.read_choice(answer)
  p_yes = None
  if isinstance(probabilities, dict):
    for option in ("yes", "1"):
      if option in probabilities:
        p_yes = probabilities[option]
        break
  return {"question_type": "choice", "answer": picked, "p_yes": p_yes,
          "confidence": confidence, "probabilities": probabilities}


# --- the hand-written states ------------------------------------------------
# Six of them, two anchors among them. Each carries the prompt_input, not the
# state, because law 2 says the typed backend is shown what the prompt is
# built from -- so the probe derives both forms from one source, exactly as
# the call sites do.

def _talk_input(context, now, subject, other, last_at, last_about,
                subject_is, other_is):
  return [context, now, subject, other, last_at, last_about, subject_is,
          other_is, subject, other]


HAND_WRITTEN = [
    # --- anchor 1: a "no" a reader does not doubt --------------------------
    {"case": "anchor-asleep", "anchor": True, "label": "decide_to_talk",
     "expect": "no",
     "why": ("two agents asleep in different houses at 03:00. A form that "
             "does not put this on 'no' is not measuring anything."),
     "domain": ["yes", "no"],
     "prompt_input": _talk_input(
         "Klaus Mueller is a student who lives alone. Maria Lopez is a "
         "student who lives with her parents. ",
         "February 13, 2023, 03:00:00 AM", "Klaus Mueller", "Maria Lopez",
         "", "",
         "Klaus Mueller is already sleeping",
         "Maria Lopez is already sleeping")},

    # --- anchor 2: the one "yes" a Wednesday morning contains --------------
    {"case": "anchor-invitation", "anchor": True, "label": "decide_to_talk",
     "expect": "yes",
     "why": ("Isabella with the party invitation in her context, meeting a "
             "regular at the counter at 11:06 -- cmpD2-jev's global maximum. "
             "A form that does not put this on 'yes' cannot rise at all."),
     "domain": ["yes", "no"],
     "prompt_input": _talk_input(
         "Isabella Rodriguez is planning a Valentine's Day party at Hobbs "
         "Cafe from 5pm to 7pm on February 14th and is inviting everyone she "
         "meets. Isabella Rodriguez was tending the counter at Hobbs Cafe. "
         "\nIsabella Rodriguez wants to tell everyone about the Valentine's "
         "Day party. Tom Moreno is a regular at Hobbs Cafe. ",
         "February 13, 2023, 11:06:00 AM", "Isabella Rodriguez", "Tom Moreno",
         "February 13, 2023, 09:12:00", "the weather and the cafe's new beans",
         "Isabella Rodriguez is already tending the counter at Hobbs Cafe",
         "Tom Moreno is already ordering a coffee at Hobbs Cafe")},

    # --- two more social states, one per question --------------------------
    {"case": "passing-strangers", "anchor": False, "label": "decide_to_talk",
     "expect": "no",
     "why": ("two agents who have never spoken, both mid-errand and heading "
             "in different directions at 08:40."),
     "domain": ["yes", "no"],
     "prompt_input": _talk_input(
         "Wolfgang Schulz is a student. Carlos Gomez is a poet. ",
         "February 13, 2023, 08:40:00 AM", "Wolfgang Schulz", "Carlos Gomez",
         "", "",
         "Wolfgang Schulz is on the way to attending his morning lecture",
         "Carlos Gomez is on the way to buying groceries")},

    {"case": "bathroom-conflict", "anchor": False, "label": "decide_to_react",
     "expect": "1",
     "why": ("the paper's own worked example: both want the same bathroom, "
             "so the subject waits. Option 1."),
     "domain": ["1", "2", "3"],
     "prompt_input": [
         "Jane Anderson is Liz Moore's house mate. Jane Anderson and Liz "
         "Moore exchanged a conversation about saying good morning at "
         "07:05am, February 13, 2023. ",
         "07:09 am, February 13, 2023",
         "Jane Anderson is on the way to using the bathroom at bathroom in "
         "Jane Anderson's house",
         "Liz Moore is already using the bathroom at bathroom in Jane "
         "Anderson's house",
         "Jane Anderson", "using the bathroom", "Liz Moore",
         "using the bathroom", "using the bathroom"]},

    # --- the location pair: two states with an obvious answer --------------
    {"case": "coffee-at-the-cafe", "anchor": False, "label": "action_sector",
     "expect": "Hobbs Cafe",
     "why": ("the task is serving customers at her own cafe and the cafe is "
             "one of the options; anything else is wrong."),
     "domain": ["Isabella Rodriguez's apartment", "Hobbs Cafe",
                "The Rose and Crown Pub", "Johnson Park",
                "Harvey Oak Supply Store", "The Willows Market and Pharmacy",
                "Oak Hill College"],
     "prompt_input": [
         "Isabella Rodriguez", "Isabella Rodriguez's apartment",
         "main room, bathroom, kitchen", "Isabella Rodriguez",
         "Isabella Rodriguez's apartment", "main room, bathroom, kitchen",
         "\nIsabella Rodriguez opens Hobbs Cafe at 8am daily.",
         ("Isabella Rodriguez's apartment, Hobbs Cafe, The Rose and Crown "
          "Pub, Johnson Park, Harvey Oak Supply Store, The Willows Market "
          "and Pharmacy, Oak Hill College"),
         "Isabella Rodriguez", "serving customers",
         "serving customers at the cafe", "Isabella Rodriguez"]},

    {"case": "shower-at-home", "anchor": False, "label": "action_arena",
     "expect": "bathroom",
     "why": ("the task is taking a shower and the bathroom is one of the "
             "three areas offered."),
     "domain": ["main room", "bathroom", "kitchen"],
     "prompt_input": [
         "Klaus Mueller", "Klaus Mueller's apartment",
         "main room, bathroom, kitchen", "Klaus Mueller",
         "taking a shower", "taking a shower", "Klaus Mueller",
         "Klaus Mueller's apartment", "main room, bathroom, kitchen"]},
]


# The social builders ALWAYS write events, one "\n", then thoughts -- the
# context has that separator even when there are no thoughts. Three of the
# hand-written contexts above were typed without it, which no call site can
# produce. The structured form (structured_state.py) reads the separator and
# refuses a context without one, so they are put in the builder's shape here:
# the sentences are events, and the thoughts are empty. Rows written before
# 2026-09-24 asked the unshaped text.
for _case in HAND_WRITTEN:
  if _case["label"] in ("decide_to_talk", "decide_to_react"):
    if "\n" not in _case["prompt_input"][0]:
      _case["prompt_input"][0] = _case["prompt_input"][0] + "\n"


# --- the trace reader -------------------------------------------------------

def read_trace(path):
  """Every probe-able state in a trace file or directory.

  YIELDS: dicts with run, step, label, state, domain, truncated.

  A `call` on path "jev" carries the state as its prompt_head; the
  `decision` that follows it under the same label carries the domain. They
  are paired by order, which is the order one thread writes them in.
  """
  files = []
  if os.path.isdir(path):
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
  else:
    files = [path]
  for filename in files:
    run = os.path.basename(filename).split("_")[0]
    pending = {}
    with open(filename) as handle:
      for line in handle:
        line = line.strip()
        if not line:
          continue
        try:
          row = json.loads(line)
        except ValueError:
          continue
        label = row.get("label")
        if row.get("event") == "call" and row.get("path") == "jev":
          if label not in PROBE_LABELS or not row.get("ok"):
            continue
          head = row.get("prompt_head") or ""
          chars = row.get("prompt_chars")
          truncated = chars is not None and len(head) < chars
          try:
            state = json.loads(head)
          except ValueError:
            continue
          if not isinstance(state, dict):
            continue
          pending[label] = {"run": run, "step": row.get("step"),
                            "label": label, "state": state, "domain": None,
                            "truncated": truncated}
        elif row.get("event") == "decision" and label in pending:
          found = pending.pop(label)
          found["domain"] = row.get("domain")
          yield found


# --- the run ----------------------------------------------------------------

def _row(base, extra):
  row = dict(base)
  row.update(extra)
  return row


def probe(cases, out, dry_run=False, four_way=True, only_forms=None):
  """Ask every case in every form that applies, writing one row per answer."""
  stamp = datetime.datetime.now().isoformat(timespec="seconds")
  written = 0
  skipped = []
  for case in cases:
    label = case["label"]
    prompt_input = case["prompt_input"]
    domain = case.get("domain")
    # subject and other ride on every row so the reader can pair an A->B ask
    # with the B->A ask asked at the SAME step. That pairing is the one
    # reading in this probe that can falsify something: if a form collapses a
    # gap the dictionary opened, the asymmetry was never in the orphan keys.
    who = state_of(label, prompt_input)
    base = {"probe": "011-IP", "at": stamp, "source": case.get("source", "hand"),
            "case": case.get("case"), "anchor": bool(case.get("anchor")),
            "label": label, "expect": case.get("expect"),
            "subject": who.get("subject") or who.get("persona"),
            "other": who.get("other"),
            "run": case.get("run"), "step": case.get("step"),
            "reconstruction": case.get("reconstruction")}
    asks = [(form, "choice") for form in forms_for(label)
            if only_forms is None or form in only_forms]
    if four_way and label == "decide_to_talk":
      # The four-way ask (TS-C): the same state as a Noul and as a Choice, on
      # the dictionary and on the composed state. The two Choice halves are
      # already in <asks>; these are the two Noul halves.
      asks += [("dictionary", "noul"), ("composed", "noul")]
    for form, question_type in asks:
      try:
        state, question = build(label, form, prompt_input, domain,
                                question_type)
      except Exception as error:
        skipped.append((case.get("case"), label, form, str(error)))
        continue
      row = _row(base, {"form": form,
                        "state_chars": len(state if isinstance(state, str)
                                           else json.dumps(state))})
      if dry_run:
        row.update({"question_type": question_type, "dry_run": True,
                    "state": state, "question": question})
      else:
        try:
          row.update(ask(label, form, prompt_input, domain, question_type))
        except Exception as error:
          row.update({"question_type": question_type, "error": str(error)})
      out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
      out.flush()
      written += 1
  return written, skipped


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
  parser.add_argument("--trace", default=None,
                      help="a trace file or directory of the probe run")
  parser.add_argument("--out", default="probe_questions.jsonl",
                      help="where the rows go")
  parser.add_argument("--dry-run", action="store_true",
                      help="build every payload, ask nothing")
  parser.add_argument("--limit", type=int, default=None,
                      help="at most this many traced states per label")
  parser.add_argument("--no-four-way", action="store_true")
  parser.add_argument("--forms", default=None,
                      help="comma-separated forms to ask (default: every "
                           "form that applies)")
  args = parser.parse_args(argv)

  cases = list(HAND_WRITTEN)
  truncated = 0
  per_label = {}
  if args.trace:
    for found in read_trace(args.trace):
      if found["truncated"]:
        # A form cannot be compared on half a situation. Say how many, and
        # say it loudly: it means the run was traced with the default
        # 200-character head and the probe needs one that was not.
        truncated += 1
        continue
      label = found["label"]
      seen = per_label.get(label, 0)
      if args.limit is not None and seen >= args.limit:
        continue
      try:
        prompt_input, how = prompt_input_of(label, found["state"],
                                            found["domain"])
      except Exception as error:
        print("(probe): skipped a %s state: %s" % (label, error))
        continue
      per_label[label] = seen + 1
      cases.append({"case": "%s-%s-%s" % (found["run"], found["step"], label),
                    "source": "trace", "anchor": False, "label": label,
                    "expect": None, "domain": found["domain"],
                    "prompt_input": prompt_input, "reconstruction": how,
                    "run": found["run"], "step": found["step"]})

  print("(probe): %d cases (%d hand-written, %d from the trace)"
        % (len(cases), len(HAND_WRITTEN), len(cases) - len(HAND_WRITTEN)))
  if truncated:
    print("(probe): SKIPPED %d truncated states -- that trace was written "
          "with too small a head; set LLM_TRACE_HEAD and run again"
          % truncated)
  with open(args.out, "w") as out:
    only = set(args.forms.split(",")) if args.forms else None
    written, skipped = probe(cases, out, dry_run=args.dry_run,
                             four_way=not args.no_four_way, only_forms=only)
  print("(probe): wrote %d rows to %s" % (written, args.out))
  for case, label, form, error in skipped:
    print("(probe): no %s form for %s (%s): %s" % (form, label, case, error))
  return 0


if __name__ == "__main__":
  sys.exit(main())
