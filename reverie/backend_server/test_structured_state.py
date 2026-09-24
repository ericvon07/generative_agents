"""
The structured form carries the same information as the paper's sentences.

structured_state.build lays a call site's prompt_input out as named fields;
to_prompt_input inverts it. If the inversion renders the paper's last block
character for character, no fact was dropped and none was added -- which is
what law 2 asks of a state the typed backend is shown. Run from
reverie/backend_server:

  ../../../.venv/bin/python test_structured_state.py
"""
import unittest

import probe_questions as P
from persona.prompt_template import structured_state as S
from persona.prompt_template.gpt_structure import composed_state


def _roundtrip(test, label, prompt_input, domain=None):
  state = S.build(label, prompt_input, domain)
  back = S.to_prompt_input(label, state, domain)
  test.assertEqual([str(i) for i in prompt_input], back)
  test.assertEqual(composed_state(prompt_input, P.TEMPLATES[label]),
                   composed_state(back, P.TEMPLATES[label]))
  return state


class RoundTrip(unittest.TestCase):

  def test_hand_written_states(self):
    for case in P.HAND_WRITTEN:
      with self.subTest(case=case["case"]):
        _roundtrip(self, case["label"], case["prompt_input"], case["domain"])

  def test_sector_between_places_is_null(self):
    domain = ["Hobbs Cafe", "Johnson Park"]
    pi = ["Ana", "Ana's house", "kitchen, bathroom", "Ana", "", "", "",
          ", ".join(domain), "Ana", "walking", "taking a walk", "Ana"]
    state = _roundtrip(self, "action_sector", pi, domain)
    self.assertIsNone(state["currently_in"])
    self.assertNotIn("daily_plan_requirement", state)

  def test_talk_keeps_progress_and_last_conversation(self):
    pi = ["Ana was reading. \nAna likes Bo. ", "February 13, 2023, 10:00:00 AM",
          "Ana", "Bo", "February 13, 2023, 09:00:00", "the weather",
          "Ana is waiting for the bus", "Bo is on the way to the park",
          "Ana", "Bo"]
    state = _roundtrip(self, "decide_to_talk", pi)
    self.assertEqual(state["subject"]["progress"], "doing it now")
    self.assertEqual(state["other"]["progress"], "on the way to do it")
    self.assertEqual(state["last_conversation"]["about"], "the weather")
    self.assertEqual(state["context"]["thoughts"], ["Ana likes Bo"])

  def test_talk_with_no_last_conversation_is_null(self):
    pi = ["\n", "now", "Ana", "Bo", "", "", "Ana is already reading",
          "Bo is already sleeping", "Ana", "Bo"]
    state = _roundtrip(self, "decide_to_talk", pi)
    self.assertIsNone(state["last_conversation"])
    self.assertEqual(state["context"],
                     {"recent_events": [], "thoughts": []})

  def test_react_with_no_address_has_null_place(self):
    pi = ["Bo was cooking. \n", "now", "Ana is on the way to cooking at ",
          "Bo is already cooking at stove in kitchen", "Ana", "cooking",
          "Bo", "cooking", "cooking"]
    state = _roundtrip(self, "decide_to_react", pi)
    self.assertIsNone(state["subject"]["place"])
    self.assertEqual(state["other"]["place"], "stove in kitchen")

  def test_a_context_without_the_separator_is_refused(self):
    with self.assertRaises(ValueError):
      S.build("decide_to_talk", ["no newline. ", "now", "Ana", "Bo", "", "",
                                 "Ana is already x", "Bo is already y",
                                 "Ana", "Bo"])

  def test_every_instruction_names_the_fields_it_points_at(self):
    for label, text in S.INSTRUCTIONS.items():
      with self.subTest(label=label):
        state = S.build(label, [c for c in P.HAND_WRITTEN
                                if c["label"] == label][0]["prompt_input"])
        for key in state:
          self.assertIn("`%s" % key, text)


if __name__ == "__main__":
  unittest.main()
