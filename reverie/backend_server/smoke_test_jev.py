"""
Smoke test for the Jev decision backend configured in ../../openai_config.json.

Run from reverie/backend_server (same cwd the simulation uses):

    export TYPESAFE_API_KEY=...
    python smoke_test_jev.py

It asks every question registered in jev_questions.py, on states shaped like
the ones the simulation actually builds, and reports per-question latency,
input tokens and cost. Nothing here touches the simulation: the point is to
find a bad key, a renamed field or an unaffordable state before a measurement
run does, and to have the numbers the plan's cost projection needs.

The sector case is the one from 22 Sep 2026 that killed a baseline run: with
the paper's prompt, llama3.1:8b answered "kitchen" for an apartment whose
arenas are {main room, bathroom}, copying it out of the prompt's own few-shot
example. A Choice cannot return it.

    python smoke_test_jev.py --batch

also sends all eight questions against one state in a single request, which is
the measurement behind the plan's batching idea: if adding questions barely
moves the latency, a whole agent's decisions for a time step cost one call.
"""
import argparse
import sys

from persona.prompt_template import jev
from persona.prompt_template import jev_questions
# D7: four questions no longer take a dictionary, so this builds their state
# the way the run path does -- the paper's last block, rendered from a
# prompt_input. probe_questions holds the templates and the shape of each
# prompt_input, and having one source for both is the point.
from persona.prompt_template.gpt_structure import composed_state
import probe_questions


# States shaped like the ones run_gpt_prompt.py builds. The fourth slot is the
# criteria the call site passes; None means the options describe themselves,
# which is true everywhere except decide_to_react, whose options are the
# numbers "1" and "2" and mean nothing without the sentences the template puts
# next to them.
#
# SINCE D7 the dictionary below is not what four of these questions receive.
# action_sector, action_arena, decide_to_talk and decide_to_react take the
# paper's own last block instead, and `state_for` turns the dictionary into
# one through probe_questions' prompt_input for the same case -- which is the
# mechanism the run path uses, so this smoke test exercises it rather than a
# shape nothing sends any more. The dictionaries stay because they are what
# the composed state is built FROM, and because a reader comparing the two
# forms wants both on the page.
CASES = [
    ("action_sector",
     {"persona": "Isabella Rodriguez",
      "lives_in": "Isabella Rodriguez's apartment",
      "areas_where_they_live": "main room, bathroom",
      "currently_in": "Isabella Rodriguez's apartment",
      "areas_where_they_are": "main room, bathroom",
      "daily_plan_requirement": "Isabella Rodriguez opens Hobbs Cafe at 8am.",
      "current_action": "waking up and completing her morning routine",
      "task": "brushing her teeth"},
     ["Isabella Rodriguez's apartment", "The Rose and Crown Pub", "Hobbs Cafe",
      "Oak Hill College", "Johnson Park", "Harvey Oak Supply Store",
      "The Willows Market and Pharmacy"],
     None),

    # The 22 Sep 2026 failure, reduced to its two options.
    ("action_arena",
     {"persona": "Isabella Rodriguez",
      "going_to": "Isabella Rodriguez's apartment",
      "areas_there": "main room, bathroom",
      "current_action": "waking up and completing her morning routine",
      "task": "brushing her teeth"},
     ["main room", "bathroom"],
     None),

    ("action_game_object",
     {"current_activity": "brushing her teeth",
      "objects_available": "shower, bathroom sink, toilet"},
     ["shower", "bathroom sink", "toilet"],
     None),

    ("decide_to_talk",
     {"subject": "Isabella Rodriguez",
      "other": "Klaus Mueller",
      "context": "Klaus Mueller is a student at Oak Hill College. Isabella "
                 "Rodriguez is the owner of Hobbs Cafe and knows Klaus as a "
                 "regular customer.",
      "now": "February 13, 2023, 08:15:00 AM",
      "last_chatted_at": "February 12, 2023, 16:20:00",
      "last_chatted_about": "the upcoming Valentine's Day party at Hobbs Cafe",
      "subject_is": "Isabella Rodriguez is on the way to opening Hobbs Cafe",
      "other_is": "Klaus Mueller is on the way to reading at the library"},
     ["yes", "no"],
     None),

    ("decide_to_react",
     {"subject": "Maria Lopez",
      "other": "Klaus Mueller",
      "context": "Maria Lopez and Klaus Mueller are both students at Oak Hill "
                 "College and often study in the same room.",
      "now": "February 13, 2023, 14:02:00 PM",
      "subject_is": "Maria Lopez is on the way to using the bathroom at "
                    "bathroom in Oak Hill College",
      "other_is": "Klaus Mueller is already using the bathroom at bathroom in "
                  "Oak Hill College"},
     ["1", "2"],
     {"1": "Wait on using the bathroom until Klaus Mueller is done using the "
           "bathroom",
      "2": "Continue on to using the bathroom now"}),

    ("event_poignancy",
     {"persona": "Isabella Rodriguez",
      "identity": "Name: Isabella Rodriguez\nAge: 34\nInnate traits: friendly, "
                  "outgoing, hospitable\nLearned traits: Isabella Rodriguez "
                  "runs Hobbs Cafe and loves making people feel at home.",
      "event": "Isabella Rodriguez is brushing her teeth"},
     None, None),

    ("thought_poignancy",
     {"persona": "Klaus Mueller",
      "identity": "Name: Klaus Mueller\nAge: 20\nInnate traits: kind, "
                  "inquisitive, passionate\nLearned traits: Klaus Mueller is "
                  "writing a research paper on gentrification.",
      "thought": "Klaus Mueller wants to finish his research paper before the "
                 "end of the semester"},
     None, None),

    ("chat_poignancy",
     {"persona": "Isabella Rodriguez",
      "identity": "Name: Isabella Rodriguez\nAge: 34\nInnate traits: friendly, "
                  "outgoing, hospitable",
      "conversation": "Isabella Rodriguez: Good morning, Klaus!\n"
                      "Klaus Mueller: Morning, Isabella. The usual, please."},
     None, None),
]


# One prompt_input per D7 question, in the order that question's
# create_prompt_input builds it, carrying the same facts as the dictionary
# above it.
_PROMPT_INPUTS = {
    "action_sector": [
        "Isabella Rodriguez", "Isabella Rodriguez's apartment",
        "main room, bathroom", "Isabella Rodriguez",
        "Isabella Rodriguez's apartment", "main room, bathroom",
        "\nIsabella Rodriguez opens Hobbs Cafe at 8am.",
        ("Isabella Rodriguez's apartment, The Rose and Crown Pub, Hobbs Cafe, "
         "Oak Hill College, Johnson Park, Harvey Oak Supply Store, "
         "The Willows Market and Pharmacy"),
        "Isabella Rodriguez", "waking up and completing her morning routine",
        "brushing her teeth", "Isabella Rodriguez"],
    "action_arena": [
        "Isabella Rodriguez", "Isabella Rodriguez's apartment",
        "main room, bathroom", "Isabella Rodriguez",
        "waking up and completing her morning routine", "brushing her teeth",
        "Isabella Rodriguez", "Isabella Rodriguez's apartment",
        "main room, bathroom"],
    "decide_to_talk": [
        ("Klaus Mueller is a student at Oak Hill College. Isabella Rodriguez "
         "is the owner of Hobbs Cafe and knows Klaus as a regular customer."),
        "February 13, 2023, 08:15:00 AM", "Isabella Rodriguez",
        "Klaus Mueller", "February 12, 2023, 16:20:00",
        "the upcoming Valentine's Day party at Hobbs Cafe",
        "Isabella Rodriguez is on the way to opening Hobbs Cafe",
        "Klaus Mueller is on the way to reading at the library",
        "Isabella Rodriguez", "Klaus Mueller"],
    "decide_to_react": [
        ("Maria Lopez and Klaus Mueller are both students at Oak Hill College "
         "and often study in the same room."),
        "February 13, 2023, 14:02:00 PM",
        ("Maria Lopez is on the way to using the bathroom at bathroom in Oak "
         "Hill College"),
        ("Klaus Mueller is already using the bathroom at bathroom in Oak Hill "
         "College"),
        "Maria Lopez", "using the bathroom", "Klaus Mueller",
        "using the bathroom", "using the bathroom"],
}


def state_for(label, dictionary):
  """What the run path would actually send for <label> today."""
  if jev_questions.state_form(label) != "composed":
    return dictionary
  return composed_state(_PROMPT_INPUTS[label], probe_questions.TEMPLATES[label])


def build(label, domain, criteria=None):
  """The question the simulation would send for <label>."""
  spec = jev_questions.spec(label)
  if spec["type"] == "choice":
    if criteria is None:
      criteria = dict((option, None) for option in domain)
    return jev.choice(spec["instructions"], criteria)
  if spec["type"] == "score":
    return jev.score(spec["instructions"], spec["criteria"])
  return jev.noul(spec["instructions"])


def describe(label, answer):
  spec = jev_questions.spec(label)
  if spec["type"] == "choice":
    choice, probabilities, confidence = jev.read_choice(answer)
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    spread = ", ".join("%s %.3f" % (k, v) for k, v in ranked[:4])
    return "%-24s confidence %.3f   [%s]" % (choice, confidence or 0.0, spread)
  if spec["type"] == "score":
    raw, confidence = jev.read_score(answer)
    levels = len(spec["criteria"])
    scaled = int(min(levels, max(1, round(raw) + 1)))
    return ("%-24s raw %.3f over 0-%d   confidence %.3f"
            % ("%d/10" % scaled, raw, levels - 1, confidence or 0.0))
  probability = jev.read_noul(answer)
  return "%-24s p(true) %.3f" % ("yes" if probability > 0.5 else "no",
                                 probability)


def cost_of(input_tokens):
  return input_tokens / 1e6 * jev.costs()["input"]


def one_at_a_time():
  """How the simulation calls it today: one decision, one request."""
  total_tokens = 0
  total_seconds = 0.0
  failures = 0

  for label, dictionary, domain, criteria in CASES:
    question = build(label, domain, criteria)
    state = state_for(label, dictionary)
    try:
      response = jev.ask_full(state, {"q": question}, label="smoke:%s" % label)
    except jev.JevError as e:
      print("%-20s FAILED  %s" % (label, e))
      failures += 1
      continue
    tokens = (response["usage"] or {}).get("input_tokens") or 0
    total_tokens += tokens
    total_seconds += response["latency"]
    print("%-20s %6.0f ms %6d tok  %s"
          % (label, response["latency"] * 1000, tokens,
             describe(label, response["answers"]["q"])))

  print()
  print("%d/%d questions answered, %.0f ms and %d input tokens in total "
        "($%.6f)" % (len(CASES) - failures, len(CASES), total_seconds * 1000,
                     total_tokens, cost_of(total_tokens)))
  return failures


def all_in_one():
  """The questions that still share one state, batched, to price a time step.

  D7 took this apart. A batched request is one state and many questions, and
  four of the eight now carry a state of their own -- the paper's own last
  block, different prose per question. They cannot be merged into a single
  dictionary, and merging their text would ask each question about three
  other situations. So the batch is what is left: action_game_object and the
  three poignancy scores. The figure it prints is no longer a whole time
  step's price and the header says so.
  """
  state = {}
  questions = {}
  left_out = []
  for label, case_state, domain, criteria in CASES:
    if jev_questions.state_form(label) == "composed":
      left_out.append(label)
      continue
    state.update(case_state)
    questions[label] = build(label, domain, criteria)
  if left_out:
    print()
    print("--- not batchable since D7 (each carries its own prose): %s ---"
          % ", ".join(left_out))

  try:
    response = jev.ask_full(state, questions, label="smoke:batch")
  except jev.JevError as e:
    print("batch FAILED  %s" % e)
    return 1

  tokens = (response["usage"] or {}).get("input_tokens") or 0
  print()
  print("--- the %d dictionary questions against one state, in one request "
        "---" % len(questions))
  for label in questions:
    print("%-20s %s" % (label, describe(label, response["answers"][label])))
  print()
  print("%.0f ms and %d input tokens for the whole call ($%.6f)"
        % (response["latency"] * 1000, tokens, cost_of(tokens)))
  return 0


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--batch", action="store_true",
                      help="also send every question in a single request")
  args = parser.parse_args()

  if not jev.configured():
    print("No API key: set %s in the environment." % jev.key_env_name())
    return 2

  print("model=%s  base_url=%s  $%.3f/M input tokens"
        % (jev.model(), jev._config["base-url"], jev.costs()["input"]))
  print()

  failures = one_at_a_time()
  if args.batch:
    failures += all_in_one()
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
