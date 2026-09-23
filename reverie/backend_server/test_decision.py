#!/usr/bin/env python3
"""
File: test_decision.py
Description: 011-IP's tests -- the composed state, the descriptive criteria,
             and D8's repair of the target's sentence.

Asks no backend and opens no socket: every case here is about what the fork
BUILDS before anything is asked. Run it on its own, like the fork's other
tests, from reverie/backend_server:

    ../../../.venv/bin/python test_decision.py
"""
import json
import datetime
import unittest

from persona.prompt_template import jev_questions
from persona.prompt_template.gpt_structure import (generate_prompt,
                                                   composed_state,
                                                   strip_answer_cue)
from persona.prompt_template import run_gpt_prompt
import probe_questions


# --- fixtures ---------------------------------------------------------------
# One known prompt_input per question, in the order that question's
# create_prompt_input builds it. probe_questions.state_of mirrors the call
# sites, so these drive both halves of every case.

TALK_INPUT = probe_questions._talk_input(
    "Isabella is a cafe owner. ", "February 13, 2023, 11:06:00 AM",
    "Isabella Rodriguez", "Tom Moreno", "February 13, 2023, 09:00:00",
    "the Valentine's Day party",
    "Isabella Rodriguez is already tending the counter",
    "Tom Moreno is already ordering coffee")

REACT_INPUT = [
    "Jane is Liz's house mate. ", "07:09 am, October 25, 2022",
    "Jane Anderson is already using the bathroom at bathroom in Jane's house",
    "Liz Moore is on the way to using the bathroom at bathroom in Jane's house",
    "Jane Anderson", "using the bathroom", "Liz Moore", "using the bathroom",
    "using the bathroom"]

SECTOR_INPUT = [
    "Isabella Rodriguez", "Hobbs Cafe", "cafe", "Isabella Rodriguez",
    "Hobbs Cafe", "cafe", "\nIsabella opens Hobbs Cafe at 8am daily.",
    "Hobbs Cafe, The Rose and Crown Pub, Johnson Park", "Isabella Rodriguez",
    "preparing the cafe", "brewing coffee", "Isabella Rodriguez"]

ARENA_INPUT = [
    "Isabella Rodriguez", "Hobbs Cafe", "cafe", "Isabella Rodriguez",
    "preparing the cafe", "brewing coffee", "Isabella Rodriguez",
    "Hobbs Cafe", "cafe"]

INPUTS = {"action_sector": SECTOR_INPUT, "action_arena": ARENA_INPUT,
          "decide_to_talk": TALK_INPUT, "decide_to_react": REACT_INPUT}

DOMAINS = {"action_sector": ["Hobbs Cafe", "The Rose and Crown Pub",
                             "Johnson Park"],
           "action_arena": ["cafe"],
           "decide_to_talk": ["yes", "no"],
           "decide_to_react": ["1", "2", "3"]}


class ComposedState(unittest.TestCase):
  """S2.1 and S2.2 -- the paper's own sentences, and nowhere to write."""

  def test_every_composed_state_is_a_substring_of_the_papers_prompt(self):
    # The claim that makes D7 auditable by string comparison rather than by
    # reading: the typed backend is shown the control arm's own words.
    for label, prompt_input in INPUTS.items():
      template = probe_questions.TEMPLATES[label]
      full = generate_prompt(prompt_input, template)
      state = composed_state(prompt_input, template)
      self.assertIn(state, full, "%s: the composed state is not a substring "
                                 "of the prompt built from the same "
                                 "prompt_input" % label)

  def test_the_few_shot_examples_are_gone(self):
    # They teach a format a typed answer does not have, and they are where
    # "kitchen" came from (jev_questions' own reason).
    state = composed_state(SECTOR_INPUT, probe_questions.TEMPLATES["action_sector"])
    self.assertNotIn("Sam Kim", state)
    self.assertNotIn("Jane Anderson", state)
    react = composed_state(REACT_INPUT, probe_questions.TEMPLATES["decide_to_react"])
    self.assertNotIn("Sam is Sarah's friend", react)

  def test_the_answer_cue_is_gone_on_every_question(self):
    # Nothing left that tells an LLM where to write.
    for label, prompt_input in INPUTS.items():
      state = composed_state(prompt_input, probe_questions.TEMPLATES[label])
      for cue in ("Answer: {", "Reasoning:", 'Answer in "yes" or "no":'):
        self.assertFalse(state.endswith(cue),
                         "%s still ends with %r" % (label, cue))
      self.assertFalse(state.endswith("{"),
                       "%s still ends with an opening brace" % label)

  def test_the_expected_sentences_are_there(self):
    sector = composed_state(SECTOR_INPUT, probe_questions.TEMPLATES["action_sector"])
    self.assertIn("Isabella Rodriguez lives in {Hobbs Cafe} that has cafe.",
                  sector)
    self.assertIn("Area options: {Hobbs Cafe, The Rose and Crown Pub, "
                  "Johnson Park}.", sector)
    self.assertTrue(sector.rstrip().endswith(
        "For brewing coffee, Isabella Rodriguez should go to the following "
        "area:"), sector[-120:])

    arena = composed_state(ARENA_INPUT, probe_questions.TEMPLATES["action_arena"])
    self.assertIn("Isabella Rodriguez is going to Hobbs Cafe that has the "
                  "following areas: {cafe}", arena)
    self.assertTrue(arena.rstrip().endswith("(MUST pick one of {cafe}):"),
                    arena[-120:])

    talk = composed_state(TALK_INPUT, probe_questions.TEMPLATES["decide_to_talk"])
    self.assertIn("Isabella Rodriguez and Tom Moreno last chatted at "
                  "February 13, 2023, 09:00:00 about the Valentine's Day "
                  "party.", talk)
    self.assertTrue(talk.rstrip().endswith(
        "Question: Would Isabella Rodriguez initiate a conversation with "
        "Tom Moreno?"), talk[-120:])

    react = composed_state(REACT_INPUT, probe_questions.TEMPLATES["decide_to_react"])
    self.assertIn("Option 1: Wait on using the bathroom until Liz Moore is "
                  "done using the bathroom", react)
    self.assertTrue(react.rstrip().endswith(
        "Option 2: Continue on to using the bathroom now"), react[-120:])

  def test_the_daily_plan_requirement_keeps_its_newline(self):
    # The call site stores prompt_input[6].strip(); the probe puts the
    # newline back when it rebuilds one from a traced state, or the rendered
    # sentence is one character away from the paper's.
    state = probe_questions.state_of("action_sector", SECTOR_INPUT)
    rebuilt, how = probe_questions.prompt_input_of(
        "action_sector", state, DOMAINS["action_sector"])
    self.assertEqual(how, "exact")
    self.assertEqual(rebuilt[6], "\nIsabella opens Hobbs Cafe at 8am daily.")

  def test_strip_answer_cue_takes_the_longer_cue_first(self):
    self.assertEqual(strip_answer_cue("... area in Hobbs Cafe:\nAnswer: {"),
                     "... area in Hobbs Cafe:")
    self.assertEqual(strip_answer_cue("... following area: {"),
                     "... following area:")
    self.assertEqual(strip_answer_cue("Question: Would A talk to B? \n\n"
                                      "Reasoning: Let's think step by step."),
                     "Question: Would A talk to B?")


class Reconstruction(unittest.TestCase):
  """The probe rebuilds a prompt_input from a traced state. It is the only
  way the composed form reaches a REAL state, so it round-trips here."""

  def test_every_question_round_trips(self):
    for label, prompt_input in INPUTS.items():
      state = probe_questions.state_of(label, prompt_input)
      rebuilt, how = probe_questions.prompt_input_of(label, state,
                                                     DOMAINS[label])
      again = probe_questions.state_of(label, rebuilt)
      self.assertEqual(state, again, "%s did not round-trip" % label)
      self.assertIn(how, ("exact", "parsed"))

  def test_the_composed_state_survives_the_round_trip(self):
    for label, prompt_input in INPUTS.items():
      template = probe_questions.TEMPLATES[label]
      state = probe_questions.state_of(label, prompt_input)
      rebuilt, _ = probe_questions.prompt_input_of(label, state,
                                                   DOMAINS[label])
      self.assertEqual(composed_state(prompt_input, template),
                       composed_state(rebuilt, template),
                       "%s: the rebuilt prompt_input renders differently"
                       % label)

  def test_an_empty_location_is_read_back(self):
    # decide_to_react writes a bare " at " when the address has no ":".
    self.assertEqual(
        probe_questions._act_desc_from("Jane is already reading at ", "Jane"),
        "reading")
    # ... and the separator is the LAST " at ", so an activity may contain one.
    self.assertEqual(
        probe_questions._act_desc_from(
            "Jane is on the way to looking at the menu at cafe in Hobbs Cafe",
            "Jane"),
        "looking at the menu")

  def test_action_sector_refuses_to_guess_the_area_options(self):
    state = probe_questions.state_of("action_sector", SECTOR_INPUT)
    with self.assertRaises(ValueError):
      probe_questions.prompt_input_of("action_sector", state, None)


class DescriptiveCriteria(unittest.TestCase):
  """S2.2 -- the option descriptions name every orphan key."""

  def test_decide_to_talk_names_every_orphan(self):
    criteria = jev_questions.descriptive_criteria("decide_to_talk")
    self.assertEqual(set(criteria), {"yes", "no"})
    for option, description in criteria.items():
      for key in jev_questions.orphan_keys("decide_to_talk"):
        self.assertIn("`%s`" % key, description,
                      "decide_to_talk's %r description does not name %r"
                      % (option, key))

  def test_decide_to_react_names_every_orphan_and_keeps_the_options(self):
    base = probe_questions.base_criteria("decide_to_react", REACT_INPUT,
                                         DOMAINS["decide_to_react"])
    criteria = jev_questions.descriptive_criteria("decide_to_react", base)
    self.assertEqual(set(criteria), set(base))
    for option, description in criteria.items():
      # The paper's Option 1 and Option 2 survive verbatim; the clause is
      # appended, never a rewrite.
      self.assertTrue(description.startswith(base[option]),
                      "decide_to_react's %r option text was rewritten" % option)
      for key in jev_questions.orphan_keys("decide_to_react"):
        self.assertIn("`%s`" % key, description,
                      "decide_to_react's %r description does not name %r"
                      % (option, key))

  def test_the_location_pair_has_no_descriptive_form(self):
    # Twelve options do not get a description each (TS-A).
    self.assertIsNone(jev_questions.descriptive_criteria("action_sector"))
    self.assertIsNone(jev_questions.descriptive_criteria("action_arena"))
    self.assertEqual(probe_questions.forms_for("action_sector"),
                     ("dictionary", "composed"))
    self.assertEqual(probe_questions.forms_for("decide_to_talk"),
                     ("dictionary", "descriptive", "composed"))

  def test_the_two_descriptions_are_each_others_mirror(self):
    # If one carried a reason the other does not, the payload would be
    # arguing rather than naming.
    criteria = jev_questions.descriptive_criteria("decide_to_talk")
    self.assertEqual(criteria["no"],
                     criteria["yes"].replace("would start", "would not start"),
                     "the two descriptions are not the same sentence apart "
                     "from the negation -- one of them is arguing")

  def test_the_orphan_table_matches_what_the_call_sites_send(self):
    # An orphan is a key the instruction never names. Recompute it from the
    # state the call site builds and the instruction beside the question, so
    # the table cannot go stale in silence.
    for label, prompt_input in INPUTS.items():
      state = probe_questions.state_of(label, prompt_input)
      instructions = jev_questions.spec(label)["instructions"]
      found = tuple(k for k in state if "`%s`" % k not in instructions)
      self.assertEqual(set(found), set(jev_questions.orphan_keys(label)),
                       "%s: ORPHAN_KEYS disagrees with the call site" % label)


# --- D8: the target's sentence ----------------------------------------------

class _Scratch(object):
  def __init__(self, act_description, planned_path):
    self.act_description = act_description
    self.planned_path = planned_path
    self.curr_time = datetime.datetime(2023, 2, 13, 9, 30, 0)


class _Memory(object):
  def get_last_chat(self, name):
    return None


class _Persona(object):
  def __init__(self, name, act_description, planned_path):
    self.name = name
    self.scratch = _Scratch(act_description, planned_path)
    self.a_mem = _Memory()


class TargetSentence(unittest.TestCase):
  """S3.1, S3.2, S3.3 -- D8. The paper tested the INITIATOR's activity for
  "waiting" when building the TARGET's sentence, and in that branch wrote the
  initiator's name and activity where the target's belong.

  Run through the real call site: the state's `other_is` IS prompt_input[7],
  so capturing what decision.choose is handed is the same as reading the
  slot the prompt interpolates.
  """

  def setUp(self):
    self.captured = {}
    self._choose = run_gpt_prompt.decision.choose
    self._debug = run_gpt_prompt.debug

    def _capture(label, domain, resolve, **kwargs):
      self.captured["label"] = label
      self.captured.update(kwargs)
      return "no"

    run_gpt_prompt.decision.choose = _capture
    run_gpt_prompt.debug = False

  def tearDown(self):
    run_gpt_prompt.decision.choose = self._choose
    run_gpt_prompt.debug = self._debug

  def _other_is(self, init_act, init_path, target_act, target_path):
    run_gpt_prompt.run_gpt_prompt_decide_to_talk(
        _Persona("Klaus Mueller", init_act, init_path),
        _Persona("Maria Lopez", target_act, target_path),
        {"events": [], "thoughts": []})
    return self.captured["state"]["other_is"]

  def test_an_initiator_who_is_waiting_does_not_take_over_the_sentence(self):
    # S3.1. Before D8 this read "Klaus Mueller is waiting for the bus".
    sentence = self._other_is("waiting for the bus", [], "reading a book",
                              [(1, 2)])
    self.assertEqual(sentence, "Maria Lopez is on the way to reading a book")
    self.assertNotIn("Klaus", sentence)

  def test_an_empty_target_path_reads_already_whatever_the_initiator_does(self):
    # S3.2. Before D8 this read "Klaus Mueller is waiting in line".
    sentence = self._other_is("waiting in line", [(3, 4)], "cooking dinner",
                              [])
    self.assertEqual(sentence, "Maria Lopez is already cooking dinner")

  def test_a_target_who_is_waiting_is_the_one_described_as_waiting(self):
    sentence = self._other_is("reading a book", [], "waiting for the bus", [])
    self.assertEqual(sentence, "Maria Lopez is waiting for the bus")

  def test_the_common_case_did_not_move(self):
    # Neither is waiting: D8 must change nothing here, or it is not a repair.
    self.assertEqual(self._other_is("reading a book", [], "cooking dinner",
                                    []),
                     "Maria Lopez is already cooking dinner")
    self.assertEqual(self._other_is("reading a book", [], "cooking dinner",
                                    [(1, 1)]),
                     "Maria Lopez is on the way to cooking dinner")

  def test_the_initiators_own_sentence_is_untouched(self):
    self._other_is("waiting for the bus", [], "reading a book", [(1, 2)])
    self.assertEqual(self.captured["state"]["subject_is"],
                     "Klaus Mueller is waiting for the bus")


if __name__ == "__main__":
  unittest.main(verbosity=2)
