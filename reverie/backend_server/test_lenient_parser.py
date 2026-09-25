#!/usr/bin/env python3
"""
File: test_lenient_parser.py
Description: 009-IP's tests -- the three tolerances of D6, the flag that
             holds them, and the paper's path untouched when it is off.

Asks no backend and opens no socket: the two request functions are replaced
with fakes for the span of a test. Run it on its own, like the fork's other
tests, from reverie/backend_server:

    ../../../.venv/bin/python test_lenient_parser.py

Scenarios of 009-IP §5 held here: S1.1 (a list is kept), S1.2 (a fenced
object parses), S1.3 (a brace missing or doubled), S1.4 (a comma and a
three-part triple stay rejected), S1.5 (no parser is handed an option set, a
persona or a prompt's inputs), S2.1 (flag off, the paper's path runs) and
S2.2 (the two flag states differ in config_sha1).
"""
import inspect
import json
import os
import tempfile
import unittest

_TRACE_DIR = tempfile.mkdtemp(prefix="lenient_parser_test_")
os.environ["LLM_TRACE_DIR"] = _TRACE_DIR
os.environ["LLM_TRACE_ENABLED"] = "1"
os.environ["DECISION_BACKEND"] = "llm"

from persona.prompt_template import lenient_parser
from persona.prompt_template import gpt_structure
from persona.prompt_template import run_gpt_prompt
from persona.prompt_template import run_manifest
from persona.prompt_template import llm_trace

A_LIST = ["What should Jane do for lunch", "Does Jane like strawberry",
          "Who is Jane"]


class _Persona(object):
  """focal_pt's prompt builder never reads the persona; the paper's debug
  print after the completion path reads its name."""
  name = "Test Persona"


PERSONA = _Persona()
FENCED = '```json\n{"output": %s}\n```' % json.dumps(A_LIST)
BARE = '{"output": %s}' % json.dumps(A_LIST)
QUOTED = '{"output": %s}' % json.dumps(json.dumps(A_LIST))


class _Fakes(object):
  """Replace the two request functions inside gpt_structure, where the
  retry loops look them up, for the span of one test."""

  def __init__(self, chat=None, completion=None):
    self.chat, self.completion = chat, completion
    self.chat_calls, self.completion_calls = [], []

  def __enter__(self):
    self._chat = gpt_structure.ChatGPT_request
    self._completion = gpt_structure.GPT_request

    def fake_chat(prompt):
      self.chat_calls.append(prompt)
      if self.chat is None:
        raise AssertionError("the chat path was not expected to be asked")
      return self.chat

    def fake_completion(prompt, gpt_parameter):
      self.completion_calls.append(prompt)
      if self.completion is None:
        raise AssertionError("the completion path was not expected to be asked")
      return self.completion

    gpt_structure.ChatGPT_request = fake_chat
    gpt_structure.GPT_request = fake_completion
    return self

  def __exit__(self, *exc):
    gpt_structure.ChatGPT_request = self._chat
    gpt_structure.GPT_request = self._completion


class _Flag(unittest.TestCase):

  def setUp(self):
    llm_trace.set_simulation("test-%s" % self.id().rsplit(".", 1)[-1])

  def tearDown(self):
    lenient_parser.set_enabled(False)


class FocalPtKeepsTheList(_Flag):
  """S1.1 and S1.2, end to end through run_gpt_prompt_focal_pt."""

  def test_a_json_array_is_accepted_on_the_first_attempt(self):
    lenient_parser.set_enabled(True)
    with _Fakes(chat=BARE) as fakes:
      output, _ = run_gpt_prompt.run_gpt_prompt_focal_pt(PERSONA, "statements", 3)
    self.assertEqual(output, A_LIST)
    self.assertEqual(len(fakes.chat_calls), 1)
    self.assertEqual(fakes.completion_calls, [])

  def test_a_fenced_array_is_accepted_with_or_without_d11(self):
    lenient_parser.set_enabled(True)
    for no_fence in ("0", "1"):
      os.environ["JSON_NO_FENCE"] = no_fence
      try:
        with _Fakes(chat=FENCED) as fakes:
          output, _ = run_gpt_prompt.run_gpt_prompt_focal_pt(PERSONA, "s", 3)
        self.assertEqual(output, A_LIST)
        self.assertEqual(len(fakes.chat_calls), 1)
      finally:
        del os.environ["JSON_NO_FENCE"]

  def test_an_array_written_inside_quotes_is_the_same_list(self):
    lenient_parser.set_enabled(True)
    with _Fakes(chat=QUOTED):
      output, _ = run_gpt_prompt.run_gpt_prompt_focal_pt(PERSONA, "s", 3)
    self.assertEqual(output, A_LIST)

  def test_a_fenced_scalar_parses_from_the_first_brace(self):
    self.assertEqual(json.loads(lenient_parser.json_cut(
        '```json\n{"output": "1"}\n```'))["output"], "1")
    # With no brace at all, the paper's cut and this one agree: nothing.
    self.assertEqual(lenient_parser.json_cut("no json here"), "")


class ThePapersPathWhenOff(_Flag):
  """S2.1: with the flag off, the paper's validators and cleanups run
  unchanged -- the list is refused three times and the completion prompt
  answers instead, exactly as GT-8 describes."""

  def test_the_list_is_refused_and_the_completion_prompt_answers(self):
    lenient_parser.set_enabled(False)
    completion = "What is a?\n2) What is b?\n3) What is c?"
    with _Fakes(chat=BARE, completion=completion) as fakes:
      output, _ = run_gpt_prompt.run_gpt_prompt_focal_pt(PERSONA, "s", 3)
    self.assertEqual(len(fakes.chat_calls), 3)
    self.assertEqual(len(fakes.completion_calls), 1)
    self.assertEqual(output, ["What is a?", "What is b?", "What is c?"])

  def test_the_fence_still_defeats_the_papers_cut(self):
    lenient_parser.set_enabled(False)
    self.assertEqual(gpt_structure.ChatGPT_request.__name__, "ChatGPT_request")
    with _Fakes(chat=FENCED, completion="1) x") as fakes:
      run_gpt_prompt.run_gpt_prompt_focal_pt(PERSONA, "s", 1)
    self.assertEqual(len(fakes.chat_calls), 3)

  def test_the_papers_lines_are_still_in_the_source(self):
    src = inspect.getsource(run_gpt_prompt)
    self.assertEqual(src.count('cleaned_response = gpt_response.split("}")[0]'), 2)
    self.assertIn("ret = ast.literal_eval(gpt_response)", src)
    # Every gate is the same call, and there are exactly six of them: three
    # validators and three cleanups (S2.1).
    self.assertEqual(src.count("if lenient_parser.enabled():"), 6)
    gs = inspect.getsource(gpt_structure)
    self.assertEqual(gs.count("if lenient_parser.enabled():"), 1)
    self.assertIn("end_index = curr_gpt_response.rfind('}') + 1", gs)

  def test_off_is_the_default(self):
    self.assertFalse(lenient_parser._DEFAULT_ENABLED)


class TheLocationTrio(_Flag):
  """S1.3 and S1.4 on the two location validators' tolerance."""

  def test_a_missing_closing_brace_is_accepted_and_stripped(self):
    self.assertTrue(lenient_parser.location_validate("{Hobbs Cafe"))
    self.assertEqual(lenient_parser.location_clean_up("{Hobbs Cafe"),
                     "Hobbs Cafe")

  def test_both_braces_come_off(self):
    self.assertEqual(lenient_parser.location_clean_up("{Hobbs Cafe}"),
                     "Hobbs Cafe")
    self.assertEqual(lenient_parser.location_clean_up("{common room}"),
                     "common room")

  def test_the_well_formed_answer_reads_as_the_paper_reads_it(self):
    papers = "Hobbs Cafe}".split("}")[0]
    self.assertEqual(lenient_parser.location_clean_up("Hobbs Cafe}"), papers)
    self.assertTrue(lenient_parser.location_validate("Hobbs Cafe}"))

  def test_a_brace_closing_mid_text_cuts_where_the_paper_cuts(self):
    self.assertEqual(lenient_parser.location_clean_up("Hobbs Cafe}\nbecause"),
                     "Hobbs Cafe")
    self.assertEqual(lenient_parser.location_clean_up("Hobbs Cafe\nbecause"),
                     "Hobbs Cafe")

  def test_a_comma_is_still_rejected(self):
    self.assertFalse(lenient_parser.location_validate("Hobbs Cafe, The Rose}"))
    self.assertFalse(lenient_parser.location_validate("{Hobbs Cafe, The Rose"))

  def test_nothing_but_braces_is_rejected(self):
    self.assertFalse(lenient_parser.location_validate("{}"))
    self.assertFalse(lenient_parser.location_validate("   "))
    self.assertFalse(lenient_parser.location_validate(None))


class StillRejected(_Flag):
  """S1.4: what would take a choice is not accepted by any tolerance."""

  def test_a_three_part_triple_is_not_a_list(self):
    self.assertFalse(lenient_parser.focal_pt_validate(
        "be, occupied by, Eddy Lin)"))

  def test_focal_pt_refuses_what_is_not_a_list_of_str(self):
    self.assertFalse(lenient_parser.focal_pt_validate([]))
    self.assertFalse(lenient_parser.focal_pt_validate([1, 2]))
    self.assertFalse(lenient_parser.focal_pt_validate("Who am I"))
    self.assertFalse(lenient_parser.focal_pt_validate({"a": "b"}))


class BlindToContent(unittest.TestCase):
  """S1.5: no parser is ever passed an option set, a persona or a prompt's
  inputs -- each takes the response and nothing else, and the module imports
  nothing that knows the world."""

  def test_every_tolerance_takes_the_response_alone(self):
    for name in ("focal_pt_validate", "focal_pt_clean_up", "json_cut",
                 "location_validate", "location_clean_up"):
      params = list(inspect.signature(getattr(lenient_parser, name)).parameters)
      self.assertEqual(len(params), 1, name)

  def test_the_module_knows_no_domain(self):
    names = set(dir(lenient_parser))
    for forbidden in ("decision", "domain_guard", "maze", "persona",
                      "jev_questions", "run_gpt_prompt", "gpt_structure"):
      self.assertNotIn(forbidden, names)


class TheFlagInTheManifest(unittest.TestCase):
  """S2.2: two configs that differ only in the flag differ in config_sha1."""

  def test_config_sha1_differs(self):
    base = {"model": "x", "domain-guard": {"enabled": True},
            "jev": {"api-key": "secret"}}
    on = dict(base, **{"lenient-parser": {"enabled": True}})
    off = dict(base, **{"lenient-parser": {"enabled": False}})
    self.assertNotEqual(run_manifest._config_fingerprint(on),
                        run_manifest._config_fingerprint(off))
    self.assertNotEqual(run_manifest._config_fingerprint(base),
                        run_manifest._config_fingerprint(off))

  def test_the_environment_override_reads_like_d11s(self):
    for value, expected in (("1", True), ("true", True), ("0", False),
                            ("off", False)):
      os.environ["LENIENT_PARSER"] = value
      try:
        self.assertEqual(lenient_parser._load_enabled(), expected)
      finally:
        del os.environ["LENIENT_PARSER"]


if __name__ == "__main__":
  unittest.main()
