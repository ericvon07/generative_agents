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
import os
import re
import json
import datetime
import unittest

from persona.prompt_template import jev_questions
from persona.prompt_template.gpt_structure import (generate_prompt,
                                                   composed_state,
                                                   strip_answer_cue)
from persona.prompt_template import run_gpt_prompt
import probe_questions
import probe_budget


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
    # An orphan is a key the DICTIONARY state carried that the DICTIONARY-era
    # instruction never named. The table is the audit of what D7 repaired, so
    # it is computed against the wording D7 replaced -- the composed
    # instruction names no key because a composed state has none, and
    # checking against it would call every key an orphan.
    for label, prompt_input in INPUTS.items():
      state = probe_questions.state_of(label, prompt_input)
      spec = jev_questions.spec(label)
      instructions = spec.get("instructions_before_d7") or spec["instructions"]
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

  def _lines(self, init_act, init_path, target_act, target_path):
    """The two description lines as the backend actually receives them.

    Before D7 this read state["other_is"], which WAS prompt_input[7]. D7 made
    the state the paper's own last block, so the slot is no longer a key --
    it is a line of prose. Reading the line back is the stronger check: it is
    what the model sees, and it would catch the sentence being right in the
    dictionary and wrong in the text.
    """
    run_gpt_prompt.run_gpt_prompt_decide_to_talk(
        _Persona("Klaus Mueller", init_act, init_path),
        _Persona("Maria Lopez", target_act, target_path),
        {"events": [], "thoughts": []})
    state = self.captured["state"]
    self.assertIsInstance(state, str, "D7: the state should be prose now")
    return [l.strip() for l in state.splitlines() if l.strip()]

  def _other_is(self, *args):
    """The one line that describes the TARGET. Exactly one line may."""
    lines = self._lines(*args)
    mine = [l for l in lines if l.startswith("Maria Lopez is ")]
    self.assertEqual(len(mine), 1,
                     "expected exactly one line about the target, got %r"
                     % (mine,))
    return mine[0]

  def _subject_is(self, *args):
    lines = self._lines(*args)
    mine = [l for l in lines if l.startswith("Klaus Mueller is ")]
    self.assertEqual(len(mine), 1,
                     "expected exactly one line about the initiator, got %r"
                     % (mine,))
    return mine[0]

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
    self.assertEqual(
        self._subject_is("waiting for the bus", [], "reading a book", [(1, 2)]),
        "Klaus Mueller is waiting for the bus")

  def test_the_target_line_is_not_a_second_copy_of_the_initiators(self):
    # The exact shape of the bug D8 repaired: before it, BOTH lines named the
    # initiator whenever the initiator was waiting.
    lines = self._lines("waiting for the bus", [], "reading a book", [(1, 2)])
    about_klaus = [l for l in lines if l.startswith("Klaus Mueller is ")]
    self.assertEqual(len(about_klaus), 1,
                     "the initiator is described twice: %r" % (about_klaus,))


# --- D7 on the run path -----------------------------------------------------

class D7OnTheRunPath(unittest.TestCase):
  """S2.1, S2.3, S2.4. The four questions send the paper's own last block,
  and the control arm's prompt did not move to make it possible."""

  FOUR = ("action_sector", "action_arena", "decide_to_talk", "decide_to_react")

  def test_the_four_questions_declare_the_composed_form(self):
    for label in self.FOUR:
      self.assertEqual(jev_questions.state_form(label), "composed",
                       "%s did not take D7" % label)

  def test_no_other_question_took_it(self):
    for label in ("action_game_object", "event_poignancy", "thought_poignancy",
                  "chat_poignancy"):
      self.assertIsNone(jev_questions.state_form(label),
                        "%s took D7 and should not have" % label)

  def test_the_wording_d7_replaced_is_still_readable(self):
    for label in self.FOUR:
      spec = jev_questions.spec(label)
      self.assertIn("instructions_before_d7", spec,
                    "%s lost the wording D7 replaced" % label)
      self.assertNotEqual(spec["instructions"], spec["instructions_before_d7"])

  def test_the_mirrors_line_is_load_bearing_now(self):
    # composed_state reads the same template the entry names, so a wrong
    # `mirrors:` stopped being a documentation slip the day D7 landed -- it
    # now sends the wrong sentences to the backend.
    for label in self.FOUR:
      mirrors = jev_questions.spec(label)["mirrors"]
      self.assertTrue(probe_questions.TEMPLATES[label].endswith(mirrors),
                      "%s: mirrors %r does not name the template that is "
                      "actually rendered (%r)"
                      % (label, mirrors, probe_questions.TEMPLATES[label]))

  def test_the_composed_state_is_what_the_control_arm_reads(self):
    # S2.1 and S2.3 in one: the state is a substring of the prompt the llm
    # backend builds from the SAME prompt_input, so the typed side gained
    # sentences without the control side losing or gaining one character.
    for label, prompt_input in INPUTS.items():
      template = probe_questions.TEMPLATES[label]
      self.assertIn(composed_state(prompt_input, template),
                    generate_prompt(prompt_input, template), label)


class SocialFieldsSurviveAStringState(unittest.TestCase):
  """D7 turns the two social states from a dict into a string, and
  _social_fields used to read `subject` out of that dict. It falls back to the
  trace's own persona scope, and the fallback is exact: across n9win and the
  three probe runs, persona == subject on 694 of 694 social decisions. This
  pins the fallback so the episode -- which S7.2's denominator is built on --
  cannot go quietly missing."""

  def test_a_string_state_still_produces_subject_other_and_episode(self):
    from persona.prompt_template import decision, llm_trace
    with llm_trace.persona_scope("Isabella Rodriguez"):
      fields = decision._social_fields("Maria Lopez",
                                       "Context: prose, not a dict.")
    self.assertEqual(fields["other"], "Maria Lopez")
    self.assertEqual(fields["subject"], "Isabella Rodriguez")
    self.assertIn("Isabella Rodriguez", fields["episode"])
    self.assertIn("Maria Lopez", fields["episode"])

  def test_no_counterparty_still_means_no_social_fields(self):
    from persona.prompt_template import decision
    self.assertIsNone(decision._social_fields(None, "prose"))


class WhatARunWouldAccept(unittest.TestCase):
  """probe_budget.py decides whether a response would be ACCEPTED by asking
  the call sites themselves. `decide_to_talk`'s reader is a module function
  since D10 and is imported; `decide_to_react`'s validator is still a closure
  and is copied, so its cue is read out of the call site's own source and
  these fail the moment it moves."""

  def setUp(self):
    import inspect
    self.source = inspect.getsource(run_gpt_prompt)

  def _body_of(self, function_name):
    """The source of one call-site function, up to the next one."""
    start = self.source.index("def %s(" % function_name)
    rest = self.source[start + 1:]
    nxt = rest.find("\ndef ")
    return rest[:nxt] if nxt != -1 else rest

  def test_decide_to_react_still_splits_on_the_cue_probe_budget_copied(self):
    body = self._body_of("run_gpt_prompt_decide_to_react")
    self.assertIn('split("%s")' % probe_budget.REACT_CUE, body)

  def test_decide_to_talk_reads_through_the_call_sites_own_function(self):
    body = self._body_of("run_gpt_prompt_decide_to_talk")
    self.assertIn("read_yes_no(gpt_response)", body)
    talk, clean = probe_budget.VALIDATORS["decide_to_talk"]
    self.assertTrue(talk("Answer: no"))
    self.assertEqual(clean("Answer: no"), "no")

  def test_the_copied_react_validator_answers_as_the_call_site_does(self):
    react, _ = probe_budget.VALIDATORS["decide_to_react"]
    self.assertTrue(react("Reasoning: blah. Answer: Option 2"))
    self.assertFalse(react("Reasoning: blah."))


class D10TheAnswerIsReadTheWayItIsWritten(unittest.TestCase):
  """The paper's decide_to_talk template asks for the answer after
  `Answer in "yes" or "no":` and its validator split on
  `Answer in yes or no:`. Five models wrote 227 answers across 546 responses
  of the fork's own model and that split accepted none of them (CU-1). D10
  reads the last answer the response states instead. These pin the shapes the
  probe actually observed, so the reader cannot be narrowed back by accident."""

  def test_the_template_still_asks_with_the_quotes(self):
    with open(probe_questions.TEMPLATES["decide_to_talk"]) as handle:
      text = handle.read()
    self.assertIn('Answer in "yes" or "no":', text)

  def test_the_cue_the_paper_validated_on_is_no_longer_split_on(self):
    """The string itself stays in the call site -- the comment beside D10
    quotes it, because a deviation says what it replaced. What must be gone
    is the SPLIT on it."""
    import inspect
    body = inspect.getsource(run_gpt_prompt.run_gpt_prompt_decide_to_talk)
    self.assertNotIn('split("%s")' % probe_budget.TALK_CUE_BEFORE_D10, body)
    self.assertIn(probe_budget.TALK_CUE_BEFORE_D10, body)

  def test_every_shape_the_probe_saw_is_read(self):
    read = run_gpt_prompt.read_yes_no
    for text, expected in (
        ('Answer in "yes" or "no": no', "no"),       # 70b, space-bunny, qwen
        ("Answer in yes or no: yes", "yes"),         # the cue the split wanted
        ("Answer: no", "no"),                        # 141 of the 8b's answers
        ("Answer:** No", "no"),                      # 86 of them, bolded
        ("Answer**: Yes", "yes"),
        ("**Answer:** yes", "yes"),
        ("yes", "yes"),                              # what the split accepted
        ("  NO  ", "no")):
      self.assertEqual(read(text), expected, text)

  def test_an_answer_followed_by_more_document_is_still_read(self):
    """gpt-3.5-turbo answers on line one and then keeps writing. An exact
    split fails on that even with the quotes repaired, which is why the
    repair is a reader and not a second string."""
    self.assertEqual(run_gpt_prompt.read_yes_no(
        'Answer in "yes" or "no": Yes\n---\nReasoning: they are both here.'),
        "yes")

  def test_the_last_answer_wins(self):
    """A response that runs on into a second worked example answers the
    question it was asked first and a different one after."""
    self.assertEqual(run_gpt_prompt.read_yes_no(
        "Answer: yes\n---\nContext: someone else\nAnswer: no"), "no")

  def test_prose_that_merely_contains_a_number_or_a_word_is_not_an_answer(self):
    for text in ("Reasoning: they have no reason to talk.",
                 "They last chatted about 2 hours ago",
                 "Answer: Option 2", "", None):
      self.assertIsNone(run_gpt_prompt.read_yes_no(text), repr(text))

  def test_the_fail_safe_is_still_what_it_was(self):
    """D10 changes what is accepted, never what happens when nothing is.
    The fail-safe on this call site is "yes", and that is the whole reason
    the rejection mattered."""
    import inspect
    body = inspect.getsource(run_gpt_prompt.run_gpt_prompt_decide_to_talk)
    self.assertIn('fs = "yes"', body)


class D9TheBudgetHasANumberBehindIt(unittest.TestCase):
  """011-IP TS-E: the two social prompts get room to answer and the location
  trio does not. 160 is the 95th percentile of where the answer lands on
  google/gemma-4-31b-it -- 101 tokens on decide_to_talk, 104 on
  decide_to_react -- with a margin chosen to cover the longest response the
  probe saw, which ran to 151. S5.3 is the other half and is held here too:
  a session raising the location trio's budget by habit breaks this."""

  BUDGETS = {"decide_to_talk": 160, "decide_to_react": 160,
             "action_sector": 15, "action_arena": 15,
             "action_game_object": 15}

  def _budget_of(self, name):
    import inspect
    source = inspect.getsource(
        getattr(run_gpt_prompt, "run_gpt_prompt_%s" % name))
    found = re.findall(r'^\s*gpt_param = .*?"max_tokens": (\d+)',
                       source, re.M)
    self.assertEqual(len(found), 1,
                     "%s has %d live gpt_param blocks" % (name, len(found)))
    return int(found[0])

  def test_the_two_social_prompts_have_the_measured_budget(self):
    for name in ("decide_to_talk", "decide_to_react"):
      self.assertEqual(self._budget_of(name), self.BUDGETS[name], name)

  def test_the_location_trio_is_untouched(self):
    for name in ("action_sector", "action_arena", "action_game_object"):
      self.assertEqual(self._budget_of(name), 15, name)

  def test_the_budget_covers_every_answer_the_probe_recorded(self):
    """The number is not a guess and this is how it is checked: read the
    archived responses of the model the budget was set from, and no answer
    may end past the budget."""
    import gzip
    import probe_budget
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..",
                        "docs", "research", "model-bench-011-2026-09-23.jsonl.gz")
    if not os.path.exists(path):
      self.skipTest("the benchmark's rows are in the lab, not the fork")
    worst = {}
    with gzip.open(path, "rt") as handle:
      for line in handle:
        row = json.loads(line)
        if row.get("model") != "google/gemma-4-31b-it" or row.get("error"):
          continue
        tokens, _exact = probe_budget.position_of(row)
        if tokens:
          worst[row["label"]] = max(worst.get(row["label"], 0), tokens)
    self.assertTrue(worst, "no gemma-4-31b answers in the archived rows")
    for label, longest in worst.items():
      self.assertLessEqual(longest, self.BUDGETS[label],
                           "%s: an answer ended at token %d, past the budget"
                           % (label, longest))


class TheWatcherSeesWhatItWasBuiltFor(unittest.TestCase):
  """watch_run.py is pointed at a window that has never run this code, so it
  must be known to fire. Each case here is a row shaped exactly as the trace
  writes it, and the brace case is GT-3's own answer out of cmpD2-llm's
  trace -- the one that killed a run one step later."""

  def _kinds(self, row):
    import watch_run
    return set(line.split("]")[0].strip("[") for line in
               watch_run.examine(row, None))

  def _decision(self, **kw):
    row = {"event": "decision", "label": "action_sector", "step": 3420,
           "persona": "Isabella Rodriguez", "answer": "the Ville",
           "is_fail_safe": False, "gave_up": False, "in_domain": True}
    row.update(kw)
    return row

  def test_the_brace_that_killed_a_run_is_seen(self):
    self.assertIn("BRACE", self._kinds(self._decision(answer="{Hobbs Cafe")))

  def test_a_clean_answer_raises_nothing(self):
    self.assertEqual(self._kinds(self._decision()), set())

  def test_a_fail_safe_is_read_from_gave_up(self):
    kinds = self._kinds(self._decision(is_fail_safe=True, gave_up=True,
                                       answer="kitchen"))
    self.assertIn("FAIL-SAFE", kinds)

  def test_an_answer_that_equals_the_fail_safe_is_not_one(self):
    """The shakedown's typed `kitchen` at confidence 1.0: is_fail_safe is
    true because the answer equals the fall-back value, and nothing fell
    back."""
    self.assertEqual(self._kinds(self._decision(
        label="action_arena", is_fail_safe=True, gave_up=False,
        answer="kitchen")), set())

  def test_an_old_trace_without_gave_up_still_reads_is_fail_safe(self):
    self.assertIn("FAIL-SAFE", self._kinds(self._decision(
        is_fail_safe=True, gave_up=None, answer="kitchen")))

  def test_a_possessive_is_not_a_brace(self):
    self.assertEqual(self._kinds(self._decision(
        answer="Wolfgang Schulz's room")), set())

  def test_a_quote_at_the_edge_is_a_brace(self):
    self.assertIn("BRACE", self._kinds(self._decision(answer="'Hobbs Cafe'")))

  def test_a_generator_fail_safe_is_flagged(self):
    self.assertIn("GEN-FAIL-SAFE", self._kinds(
        {"event": "outcome", "label": "run_gpt_prompt_pronunciatio",
         "step": 3241, "persona": "x", "failsafe": True, "attempts": 3}))

  def test_out_of_domain_is_seen_even_when_the_guard_did_not_fire(self):
    self.assertIn("OUT-OF-DOMAIN",
                  self._kinds(self._decision(in_domain=False,
                                             answer="kitchen")))

  def test_the_guard_substituting_is_gt_3(self):
    self.assertIn("GUARD", self._kinds(
        {"event": "guard", "label": "action_arena", "step": 3420,
         "persona": "Isabella Rodriguez", "answer": "kitchen",
         "replacement": "bathroom"}))

  def test_a_social_prompt_cut_at_the_budget_says_d9_is_short(self):
    kinds = self._kinds({"event": "reject", "step": 4000, "persona": "x",
                         "label": "run_gpt_prompt_decide_to_talk",
                         "reason": "truncated"})
    self.assertIn("TRUNCATED", kinds)

  def test_an_ordinary_validation_rejection_is_counted_not_flagged(self):
    """Rejections are the 009-IP tally, not an interruption: a run that
    rejects and then succeeds is the paper's retry loop working."""
    kinds = self._kinds({"event": "reject", "step": 4000, "persona": "x",
                         "label": "run_gpt_prompt_action_sector",
                         "reason": "validate_false"})
    self.assertEqual(kinds, set())

  def test_a_transport_failure_is_flagged_because_the_pin_forbids_fallbacks(self):
    kinds = self._kinds({"event": "reject", "step": 4000, "persona": "x",
                         "label": "run_gpt_prompt_action_sector",
                         "reason": "transport:RateLimitError"})
    self.assertIn("TRANSPORT", kinds)

  def test_the_tally_counts_rejections_by_label_and_reason(self):
    import watch_run
    tally = watch_run.Tally()
    for _ in range(3):
      tally.add({"event": "reject", "label": "run_gpt_prompt_action_sector",
                 "reason": "validate_false", "step": 1})
    tally.add({"event": "decision", "label": "action_sector", "step": 1,
               "is_fail_safe": True, "gave_up": True})
    digest = tally.digest()
    self.assertIn("action_sector", digest)
    self.assertIn("validate_false 3", digest)
    self.assertIn("1 (100%)", digest)


if __name__ == "__main__":
  unittest.main(verbosity=2)
