"""
File: test_domain_guard.py
Description: The four cases the restored action_arena domain guard has to get
             right, plus the two that say the measurement did not move.

Run from reverie/backend_server (the cwd the simulation uses):

    python test_domain_guard.py

Nothing here touches a model. The guard is a pure function of (answer,
domain) plus one call to random.choice, which the tests seed, so the whole
file is deterministic even though a run is not: a simulation already draws
without a seed on every movement (execute.py:82,99,101).

The case that matters is the one that ended cmpB-llm at step 3239: the model
answered "kitchen" for an apartment whose arenas are {main room, bathroom},
copying it from the prompt template's own first few-shot example, and the
maze raised KeyError one step later.
"""
import json
import os
import random
import shutil
import tempfile
import unittest

# The trace and the seam read their configuration at import time, so the
# environment has to be set before the imports below.
_TRACE_DIR = tempfile.mkdtemp(prefix="domain_guard_test_")
os.environ["LLM_TRACE_DIR"] = _TRACE_DIR
os.environ["LLM_TRACE_ENABLED"] = "1"
os.environ["DECISION_BACKEND"] = "llm"

from persona.prompt_template import decision, domain_guard, llm_trace

# Isabella's apartment at step 3239, as the prompt offered it.
ARENAS = ["main room", "bathroom"]


class DomainGuard(unittest.TestCase):

  def setUp(self):
    self.sim = "test-%s" % self.id().rsplit(".", 1)[-1]
    llm_trace.set_simulation(self.sim)
    domain_guard.set_enabled(True)
    random.seed(20260922)

  def tearDown(self):
    domain_guard.set_enabled(True)

  def records(self):
    """Everything this test wrote. One file per test would be enough, but the
    trace rotates lazily, so the sim name is what separates them."""
    out = []
    for name in sorted(os.listdir(_TRACE_DIR)):
      with open(os.path.join(_TRACE_DIR, name), "r", encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if line:
            record = json.loads(line)
            if record.get("sim") == self.sim:
              out.append(record)
    return out

  def guard_records(self):
    return [r for r in self.records() if r.get("event") == "guard"]

  # --- the answer inside the set is not touched ----------------------------

  def test_an_answer_in_the_set_passes_through(self):
    answer = domain_guard.keep_in_domain("action_arena", "bathroom", ARENAS)
    self.assertEqual("bathroom", answer)
    self.assertEqual([], self.guard_records())

  def test_membership_is_the_test_the_trace_used(self):
    """The guard must not fire on an answer record_decision called valid.
    Both sides strip braces, whitespace and case, so the prompt's "{Bathroom}"
    is the maze's "bathroom"."""
    for spelling in ("{bathroom}", " Bathroom ", "MAIN ROOM"):
      answer = domain_guard.keep_in_domain("action_arena", spelling, ARENAS)
      self.assertEqual(spelling, answer)
      self.assertTrue(decision.in_domain(spelling, ARENAS))
    self.assertEqual([], self.guard_records())

  # --- the answer outside the set is replaced ------------------------------

  def test_the_kitchen_that_ended_the_run_is_replaced(self):
    answer = domain_guard.keep_in_domain("action_arena", "kitchen", ARENAS)
    self.assertIn(answer, ARENAS)

  def test_the_replacement_is_recorded(self):
    answer = domain_guard.keep_in_domain("action_arena", "kitchen", ARENAS)
    guards = self.guard_records()
    self.assertEqual(1, len(guards))
    self.assertEqual("action_arena", guards[0]["label"])
    self.assertEqual("kitchen", guards[0]["answer"])     # what the model said
    self.assertEqual(answer, guards[0]["replacement"])   # what the run used
    self.assertEqual(ARENAS, guards[0]["domain"])

  def test_the_decision_event_keeps_the_model_s_answer(self):
    """R2: the out-of-domain rate is measured before the guard runs, so the
    number this project reports does not move when the guard is switched on."""
    output = decision.choose("action_arena", ARENAS, lambda: "kitchen",
                             fail_safe="kitchen")
    output = domain_guard.keep_in_domain("action_arena", output, ARENAS)
    decisions = [r for r in self.records() if r.get("event") == "decision"]
    self.assertEqual(1, len(decisions))
    self.assertEqual("kitchen", decisions[-1]["answer"])
    self.assertIs(False, decisions[-1]["in_domain"])
    self.assertIn(output, ARENAS)

  # --- switched off, nothing happens at all --------------------------------

  def test_disabled_returns_the_answer_untouched(self):
    domain_guard.set_enabled(False)
    answer = domain_guard.keep_in_domain("action_arena", "kitchen", ARENAS)
    self.assertEqual("kitchen", answer)
    self.assertEqual([], self.guard_records())

  # --- degenerate input ----------------------------------------------------

  def test_an_empty_domain_is_left_alone(self):
    """No declared set, nothing to enforce: a guard that invented an answer
    here would be doing something the authors' line never did."""
    self.assertEqual("kitchen",
                     domain_guard.keep_in_domain("action_arena", "kitchen", []))
    self.assertEqual([], self.guard_records())


if __name__ == "__main__":
  try:
    unittest.main(exit=False, verbosity=2)
  finally:
    shutil.rmtree(_TRACE_DIR, ignore_errors=True)
