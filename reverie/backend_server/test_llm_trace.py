"""
File: test_llm_trace.py
Description: The five cases that separate "the answer equals the fail-safe"
             from "the decision actually fell back", plus the two that say
             which rejected responses are kept whole.

Run from reverie/backend_server (the cwd the simulation uses):

    python test_llm_trace.py

The bug these cover: record_decision used to infer the fallback by comparing
the answer with the fail-safe value. That only works when the fail-safe is
something the model cannot say, and in this simulator it almost never is --
event_poignancy falls back to 4 on a 1-10 scale, action_sector falls back to
"kitchen", which is a real sector. Cases 1 and 4 are the false positives that
inflated the typed backend's fail-safe rate; case 3 is the one the first
attempt at the fix got wrong, charging a generation prompt's fallback to the
decision that happened to enclose it.

Nothing here touches a model: the retry loops are stood in for by calls to
record_outcome, which is exactly what gpt_structure.py emits before it gives
up (gpt_structure.py:202 and gpt_structure.py:352), and by calls to
record_reject, which is what they emit for every response they throw away
(gpt_structure.py:196 and gpt_structure.py:346).

RejectedText covers the second question the trace has to answer: of the
responses the retry loop discarded on a closed-set decision, how many already
named a member of the set? That cannot be read off a 200-character head, so
those rejects -- and only those -- keep the whole text.
"""
import json
import os
import shutil
import tempfile
import unittest

# The trace and the seam read their configuration at import time, so the
# environment has to be set before the imports below.
_TRACE_DIR = tempfile.mkdtemp(prefix="llm_trace_test_")
os.environ["LLM_TRACE_DIR"] = _TRACE_DIR
os.environ["LLM_TRACE_ENABLED"] = "1"
os.environ["DECISION_BACKEND"] = "llm"

import compare_backends
from persona.prompt_template import decision, llm_trace

SECTORS = ["kitchen", "main room", "bathroom", "common room"]


def exhausted(label, fail_safe):
  """A retry loop that ran out of attempts: it reports the fallback and then
  returns the fail-safe value. This is safe_generate_response's last two
  lines."""
  def resolve():
    llm_trace.record_outcome("safe_generate_response", 3, True, False,
                             label=label)
    return fail_safe
  return resolve


def succeeded(answer, label=None):
  """A prompt that parsed on some attempt. The loop returns before it reaches
  the record_outcome that reports a fallback."""
  def resolve():
    llm_trace.record_outcome("safe_generate_response", 1, False, True,
                             label=label or "run_gpt_prompt_unused")
    return answer
  return resolve


class FailSafeTrace(unittest.TestCase):

  def setUp(self):
    llm_trace.set_simulation("test-%s" % self.id().rsplit(".", 1)[-1])

  def decision_record(self):
    """The last decision written to the trace."""
    records = []
    for name in sorted(os.listdir(_TRACE_DIR)):
      with open(os.path.join(_TRACE_DIR, name), "r", encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if line:
            records.append(json.loads(line))
    decisions = [r for r in records if r.get("event") == "decision"]
    self.assertTrue(decisions, "no decision was written to the trace")
    return decisions[-1]

  # --- case 1: the model genuinely answers the fail-safe value -------------

  def test_genuine_kitchen_is_not_a_fallback(self):
    answer = decision.choose(
        "action_sector", SECTORS,
        succeeded("kitchen", "run_gpt_prompt_action_sector"),
        fail_safe="kitchen")
    record = self.decision_record()
    self.assertEqual("kitchen", answer)
    self.assertTrue(record["is_fail_safe"])   # the value matches
    self.assertFalse(record["gave_up"])       # the decision did not fail
    self.assertTrue(record["in_domain"])

  # --- case 2: the loop exhausts and hands back the fail-safe --------------

  def test_exhausted_loop_is_a_fallback(self):
    answer = decision.choose(
        "action_sector", SECTORS,
        exhausted("run_gpt_prompt_action_sector", "kitchen"),
        fail_safe="kitchen")
    record = self.decision_record()
    self.assertEqual("kitchen", answer)
    self.assertTrue(record["is_fail_safe"])
    self.assertTrue(record["gave_up"])

  # --- case 3: a generation prompt gives up inside the window --------------

  def test_generation_fallback_is_not_charged_to_the_decision(self):
    def resolve():
      # The typed condition keeps generation on the LLM, so daily schedules
      # and object descriptions keep exhausting their retries around a
      # decision that is doing fine.
      llm_trace.record_outcome("safe_generate_response", 3, True, False,
                               label="run_gpt_prompt_daily_plan")
      return "main room"

    answer = decision.choose("action_sector", SECTORS, resolve,
                             fail_safe="kitchen")
    record = self.decision_record()
    self.assertEqual("main room", answer)
    self.assertFalse(record["is_fail_safe"])
    self.assertFalse(record["gave_up"])

  # --- case 4: a genuine 4 on the poignancy scale --------------------------

  def test_genuine_score_of_four_is_not_a_fallback(self):
    answer = decision.score(
        "event_poignancy",
        succeeded(4, "run_gpt_prompt_event_poignancy"),
        fail_safe=4)
    record = self.decision_record()
    self.assertEqual(4, answer)
    self.assertTrue(record["is_fail_safe"])
    self.assertFalse(record["gave_up"])

  # --- case 5: the scorer's loop exhausts ----------------------------------

  def test_exhausted_scorer_is_a_fallback(self):
    answer = decision.score(
        "event_poignancy",
        exhausted("run_gpt_prompt_event_poignancy", 4),
        fail_safe=4)
    record = self.decision_record()
    self.assertEqual(4, answer)
    self.assertTrue(record["is_fail_safe"])
    self.assertTrue(record["gave_up"])


class RejectedText(unittest.TestCase):
  """A rejection inside a closed-set decision keeps its whole text; every
  other rejection keeps the 200-character head and nothing more.

  The asymmetry is the point. The long rejected responses in a real run are
  the generation ones -- a daily schedule, a conversation -- and there are
  some three hundred of them per window against a few dozen decision
  rejections. Keeping every response whole to recover the short ones would
  pay for the wrong thing."""

  def setUp(self):
    llm_trace.set_simulation("test-%s" % self.id().rsplit(".", 1)[-1])

  def reject_record(self):
    """The last reject written to the trace."""
    records = []
    for name in sorted(os.listdir(_TRACE_DIR)):
      with open(os.path.join(_TRACE_DIR, name), "r", encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if line:
            records.append(json.loads(line))
    rejects = [r for r in records if r.get("event") == "reject"]
    self.assertTrue(rejects, "no reject was written to the trace")
    return rejects[-1]

  # A refusal long enough to be truncated, with the answer at the far end --
  # which is where a chatty model puts it, and why the head is not enough.
  WORDY = ("Of course! Here is my reasoning. Isabella is thinking about the "
           "Valentine's Day party and needs a place to prepare, so she would "
           "naturally want somewhere with counters and running water rather "
           "than a place to sit or to sleep, which rules out the other "
           "options one by one. Therefore the answer is: kitchen")

  # --- case 6 (S1.1): a rejection inside a decision is kept whole ----------

  def test_decision_rejection_keeps_the_whole_response(self):
    def resolve():
      llm_trace.record_reject("validate_false", self.WORDY,
                              label="run_gpt_prompt_action_arena")
      llm_trace.record_outcome("safe_generate_response", 2, False, True,
                               label="run_gpt_prompt_action_arena")
      return "kitchen"

    decision.choose("action_arena", SECTORS, resolve, fail_safe="kitchen")
    record = self.reject_record()
    self.assertGreater(len(self.WORDY), len(record["resp_head"]))
    self.assertTrue(record["resp_head"].endswith("..."))
    self.assertEqual(self.WORDY, record["resp_full"])

  # --- case 7 (S1.2): a generation rejection keeps only the head -----------

  def test_generation_rejection_keeps_only_the_head(self):
    llm_trace.record_reject("exception:ValueError", self.WORDY,
                            label="run_gpt_prompt_new_decomp_schedule")
    record = self.reject_record()
    self.assertTrue(record["resp_head"].endswith("..."))
    self.assertNotIn("resp_full", record)


class Backfill(unittest.TestCase):
  """The readers have to recover the signal from traces written before the
  seam recorded it, which is every trace collected up to 22 Sep 2026."""

  def test_backfill_matches_the_outcome_events(self):
    records = [
        {"event": "outcome", "label": "run_gpt_prompt_action_sector",
         "failsafe": True},
        {"event": "decision", "label": "action_sector", "answer": "kitchen"},
        # a generation prompt exhausting between two decisions
        {"event": "outcome", "label": "run_gpt_prompt_daily_plan",
         "failsafe": True},
        {"event": "decision", "label": "action_arena", "answer": "kitchen"},
        # a decision that already carries the field is left alone
        {"event": "decision", "label": "action_sector", "answer": "kitchen",
         "gave_up": False},
    ]
    compare_backends._backfill_gave_up(records)
    decisions = [r for r in records if r["event"] == "decision"]
    self.assertEqual([True, False, False],
                     [compare_backends.gave_up(r) for r in decisions])
    self.assertNotIn("gave_up_derived", decisions[-1])


if __name__ == "__main__":
  try:
    unittest.main(exit=False, verbosity=2)
  finally:
    shutil.rmtree(_TRACE_DIR, ignore_errors=True)
