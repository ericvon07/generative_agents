"""
File: test_llm_trace.py
Description: The five cases that separate "the answer equals the fail-safe"
             from "the decision actually fell back", the two that say which
             rejected responses are kept whole, and the five that say what
             kind of failure a rejection was.

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

TransportFailure covers the third, and it is the one with teeth: a backend
that did not answer must not look like a backend that answered badly. Before
006-IP, GPT_request turned any exception into the string "TOKEN LIMIT
EXCEEDED" and handed it to the validator -- and the validators for sector,
arena and game object accept any non-empty string, so a dead backend read as
a live simulation choosing badly. Two cases, deliberately side by side: the
transport failure that raises and spends no attempt, and the bad answer that
does neither.

Truncation covers the fourth: a response the provider cut for the token
budget is not a response the model got wrong. SR-1 found all 35 discarded
social responses stopping mid-clause on a 20-token budget, and every one of
them filed under validate_false. These three cases are that reading made a
field rather than an afternoon.

RideOut covers the fifth, and it is the one that decides whether a routed run
finishes. rtrA-llm took two upstream 429s from DeepInfra 24 seconds apart and
each of them ended a stage, because RB-C makes a transport failure raise. SV-A
of 008-IP gives RB-C the bounded wait its reopening clause asked for: a 429 is
ridden out, every wait is a record, and when the budget is spent the SAME
exception still propagates. The two cases are deliberately side by side -- the
limit that passes and the limit that does not.

TransportFailure, Truncation and RideOut share a stubbed client through
Stubbed, which is deliberately not a TestCase: inheriting one re-runs its cases
under the subclass's name, against the same trace file, and the counts double.
Episodes covers the sixth: a social decision says who it was about and which
run of consecutive asks it belongs to. 007-ST had to recover both by joining
each decision back to its call event on a timestamp, which succeeded on 76 of
cmpD2-jev's 98 asks and left 22 with no counterparty at all. Four cases: the
counterparty is there, one encounter is one episode, a pair that meets again
after a gap gets a second one, and the two social decisions of the same
encounter share it.
"""
import json
import os
import shutil
import tempfile
import unittest

import httpx
from openai import RateLimitError

# The trace and the seam read their configuration at import time, so the
# environment has to be set before the imports below.
_TRACE_DIR = tempfile.mkdtemp(prefix="llm_trace_test_")
os.environ["LLM_TRACE_DIR"] = _TRACE_DIR
os.environ["LLM_TRACE_ENABLED"] = "1"
os.environ["DECISION_BACKEND"] = "llm"

import compare_backends
from persona.prompt_template import decision, gpt_structure, llm_trace

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


class OwnTrace(object):
  """Every suite reads its own trace file and nobody else's.

  The readers used to concatenate every file in the directory and take the
  last record, which held only while the suites that did it happened to sort
  last. Adding one suite whose method names sort later silently moved four
  cases' "last decision" into a file they never wrote. The file is named for
  the class as well as the method, and the reader filters on that name.
  """

  def sim_prefix(self):
    return "test-%s-%s" % (type(self).__name__, self.id().rsplit(".", 1)[-1])

  def records(self, event):
    out = []
    for name in sorted(os.listdir(_TRACE_DIR)):
      if not name.startswith(self.sim_prefix()):
        continue
      with open(os.path.join(_TRACE_DIR, name), "r", encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if line:
            record = json.loads(line)
            if record.get("event") == event:
              out.append(record)
    return out


class FailSafeTrace(OwnTrace, unittest.TestCase):

  def setUp(self):
    llm_trace.set_simulation(self.sim_prefix())

  def decision_record(self):
    """The last decision written to the trace."""
    decisions = self.records("decision")
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


class RejectedText(OwnTrace, unittest.TestCase):
  """A rejection inside a closed-set decision keeps its whole text; every
  other rejection keeps the 200-character head and nothing more.

  The asymmetry is the point. The long rejected responses in a real run are
  the generation ones -- a daily schedule, a conversation -- and there are
  some three hundred of them per window against a few dozen decision
  rejections. Keeping every response whole to recover the short ones would
  pay for the wrong thing."""

  def setUp(self):
    llm_trace.set_simulation(self.sim_prefix())

  def reject_record(self):
    """The last reject written to the trace."""
    rejects = self.records("reject")
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


class Stub(object):
  """Stands in for gpt_structure.client: answers, or fails to.

  Shaped like the sliver of the OpenAI client the two request paths touch --
  client.chat.completions.create(...) -- and nothing more."""

  def __init__(self, error=None, text="kitchen", finish_reason="stop",
               fails=None):
    self.error = error
    self.text = text
    self.finish_reason = finish_reason
    # How many of the first requests <error> is raised for. None means every
    # one of them, which is what a dead backend looks like; a number is what a
    # provider's bad minute looks like.
    self.fails = fails
    self.calls = 0
    self.chat = self

  @property
  def completions(self):
    return self

  def create(self, **kwargs):
    self.calls += 1
    if self.error is not None and (self.fails is None
                                   or self.calls <= self.fails):
      raise self.error
    message = type("M", (), {"content": self.text})()
    choice = type("C", (), {"message": message,
                            "finish_reason": self.finish_reason})()
    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 2,
                           "total_tokens": 12})()
    return type("R", (), {"choices": [choice], "usage": usage,
                          "model": "stub"})()


class Stubbed(OwnTrace):
  """The machinery the suites below share: a stubbed client, a silenced cost
  ledger, and a reader for the records this test wrote. Not a TestCase --
  inheriting one would re-run its cases under every subclass's name."""

  PARAMS = {"engine": "stub", "max_tokens": 20, "temperature": 0, "top_p": 1,
            "stream": False, "frequency_penalty": 0, "presence_penalty": 0,
            "stop": None}

  def setUp(self):
    llm_trace.set_simulation(self.sim_prefix())
    self.real_client = gpt_structure.client
    self.real_logger = gpt_structure.cost_logger
    # The cost logger writes a JSON ledger per call; nothing here is about
    # the ledger, and a test does not get to touch the run's cost logs.
    gpt_structure.cost_logger = type(
        "Silent", (), {"update_cost": lambda self, *a, **k: None})()

  def tearDown(self):
    gpt_structure.client = self.real_client
    gpt_structure.cost_logger = self.real_logger

class TransportFailure(Stubbed, unittest.TestCase):
  """A backend that did not answer is not a backend that answered badly."""

  # --- case 8 (S2.1, S2.2): the transport fails and the run stops ----------

  def test_a_transport_failure_raises_and_spends_no_attempt(self):
    gpt_structure.client = Stub(error=ConnectionError("connection refused"))

    with self.assertRaises(gpt_structure.BackendTransportError):
      gpt_structure.safe_generate_response(
          "where is Isabella?", self.PARAMS, repeat=5,
          fail_safe_response="kitchen",
          func_validate=lambda r, prompt: True,
          func_clean_up=lambda r, prompt: r)

    # One request, not five: a failure that answered nothing has nothing for
    # the next four attempts to learn from.
    self.assertEqual(1, gpt_structure.client.calls)

    calls = self.records("call")
    self.assertTrue(calls)
    self.assertFalse(calls[-1]["ok"])
    self.assertIn("ConnectionError", calls[-1]["error"])

    rejects = self.records("reject")
    self.assertTrue(rejects)
    self.assertEqual("transport:ConnectionError", rejects[-1]["reason"])
    # The whole point: not one of them may read as the model answering.
    self.assertNotIn("validate_false", [r["reason"] for r in rejects])

    # And no fail-safe reached the world -- safe_generate_response never got
    # to its last line.
    self.assertEqual([], self.records("outcome"))

  # --- case 9 (S2.2): a bad answer is still a bad answer -------------------

  def test_a_bad_answer_does_not_raise(self):
    gpt_structure.client = Stub(text="the Hobbs Cafe, probably")

    answer = gpt_structure.safe_generate_response(
        "where is Isabella?", self.PARAMS, repeat=3,
        fail_safe_response="kitchen",
        func_validate=lambda r, prompt: False,
        func_clean_up=lambda r, prompt: r)

    self.assertEqual("kitchen", answer)          # the fail-safe, as before
    self.assertEqual(3, gpt_structure.client.calls)  # every attempt spent
    reasons = [r["reason"] for r in self.records("reject")]
    self.assertEqual(["validate_false"] * 3, reasons)
    self.assertTrue(self.records("outcome")[-1]["failsafe"])


class Truncation(Stubbed, unittest.TestCase):
  """A response the provider cut for the token budget is not a response the
  model got wrong.

  SR-1 is what this is for. All 35 discarded social responses in the 22 Sep
  window stop mid-clause, because decide_to_talk and decide_to_react set
  max_tokens to 20 -- and every one of them was filed under validate_false,
  which reads as a model that cannot follow a format. Reading that off the
  trace took a person looking at the text by hand. It is now a field.

  Inherits TransportFailure's stub, its client swap and its readers; nothing
  here is about a transport failure.
  """

  # --- case 10 (S3.1): every call says why it stopped ----------------------

  def test_every_call_records_why_generation_stopped(self):
    gpt_structure.client = Stub(text="kitchen", finish_reason="stop")
    gpt_structure.safe_generate_response(
        "where is Isabella?", self.PARAMS, repeat=1,
        fail_safe_response="kitchen",
        func_validate=lambda r, prompt: True,
        func_clean_up=lambda r, prompt: r)
    self.assertEqual("stop", self.records("call")[-1]["finish_reason"])

  # --- case 11 (S3.2): the 20-token budget reads as the budget -------------

  def test_a_truncated_answer_is_not_a_wrong_answer(self):
    # What decide_to_talk gets back from a model that reasons before it
    # answers: 20 tokens of reasoning, cut before the "yes" or "no".
    gpt_structure.client = Stub(
        text="Let me think about this. Isabella and Klaus have been",
        finish_reason="length")

    gpt_structure.safe_generate_response(
        "would Isabella talk to Klaus?", self.PARAMS, repeat=2,
        fail_safe_response="no",
        func_validate=lambda r, prompt: False,
        func_clean_up=lambda r, prompt: r)

    reasons = [r["reason"] for r in self.records("reject")]
    self.assertEqual(["truncated", "truncated"], reasons)
    self.assertNotIn("validate_false", reasons)

  # --- case 12 (S3.2): half a JSON document is the budget, not the model ---

  def test_a_truncated_json_keeps_the_exception_after_the_colon(self):
    gpt_structure.client = Stub(text='{"output": "kit', finish_reason="length")

    gpt_structure.ChatGPT_safe_generate_response(
        "which sector?", example_output="kitchen", special_instruction="",
        repeat=1, fail_safe_response="kitchen",
        func_validate=lambda r, prompt: True,
        func_clean_up=lambda r, prompt: r)

    reason = self.records("reject")[-1]["reason"]
    self.assertTrue(reason.startswith("truncated"), reason)
    self.assertIn("JSONDecodeError", reason)


def rate_limited(retry_after=None):
  """What OpenRouter raises when the provider behind it is overloaded.

  A real openai.RateLimitError, not a stand-in: the ride-out has to recognise
  the exception the client actually constructs, and the reject that follows an
  exhausted ride-out has to read "transport:RateLimitError" -- the type is the
  field, so a stub type would test the stub."""
  request = httpx.Request("POST",
                          "https://openrouter.ai/api/v1/chat/completions")
  headers = {} if retry_after is None else {"retry-after": str(retry_after)}
  response = httpx.Response(429, headers=headers, request=request)
  return RateLimitError(
      "Error code: 429 - {'error': {'message': 'Provider returned error', "
      "'code': 429, 'metadata': {'provider_name': 'DeepInfra', "
      "'provider_error_code': 'engine_overloaded'}}}",
      response=response, body=None)


class RideOut(Stubbed, unittest.TestCase):
  """A provider's rate limit costs time, never a stage and never an answer."""

  POLICY = {"attempts": 3, "base_delay": 2.0, "max_delay": 60.0}

  def setUp(self):
    Stubbed.setUp(self)
    self.slept = []
    self.real_sleep = gpt_structure._sleep
    self.real_policy = gpt_structure.rate_limit_policy
    # The wait is what this code is for, so it cannot be tested by waiting.
    gpt_structure._sleep = self.slept.append
    gpt_structure.rate_limit_policy = dict(self.POLICY)

  def tearDown(self):
    gpt_structure._sleep = self.real_sleep
    gpt_structure.rate_limit_policy = self.real_policy
    Stubbed.tearDown(self)

  # --- case 13 (S1.1, S1.2): the limit passes and the run does not notice ---

  def test_a_rate_limit_is_ridden_out(self):
    gpt_structure.client = Stub(error=rate_limited(), text="kitchen", fails=2)

    answer = gpt_structure.safe_generate_response(
        "where is Isabella?", self.PARAMS, repeat=5,
        fail_safe_response="bed",
        func_validate=lambda r, prompt: True,
        func_clean_up=lambda r, prompt: r)

    self.assertEqual("kitchen", answer)
    # Three requests, one attempt: the ride-out is not the retry loop, and it
    # spends nothing the validators can see.
    self.assertEqual(3, gpt_structure.client.calls)
    self.assertEqual([2.0, 4.0], self.slept)

    waits = self.records("rate_limit")
    self.assertEqual(2, len(waits))
    self.assertEqual([429, 429], [w["status"] for w in waits])
    self.assertEqual([1, 2], [w["ride_attempt"] for w in waits])
    self.assertEqual([3, 3], [w["ride_budget"] for w in waits])
    self.assertEqual([2.0, 4.0], [w["delay_s"] for w in waits])
    self.assertEqual(["GPT_request", "GPT_request"],
                     [w["path"] for w in waits])

    # And exactly one call record, for the request that answered. A wait is
    # not a call: nothing was served and no tokens were spent, and every
    # wall-clock figure in the lab is a count of these.
    calls = self.records("call")
    self.assertEqual(1, len(calls))
    self.assertTrue(calls[0]["ok"])
    # The waits are 6 seconds; the latency is the answering request's alone.
    self.assertLess(calls[0]["latency_ms"], 1000)

    self.assertEqual([], self.records("reject"))
    self.assertFalse(self.records("outcome")[-1]["failsafe"])

  # --- case 14 (S1.3): the budget runs out and RB-C is still RB-C -----------

  def test_a_persistent_rate_limit_exhausts_the_ride_and_raises(self):
    gpt_structure.client = Stub(error=rate_limited())

    with self.assertRaises(gpt_structure.BackendTransportError):
      gpt_structure.safe_generate_response(
          "where is Isabella?", self.PARAMS, repeat=5,
          fail_safe_response="kitchen",
          func_validate=lambda r, prompt: True,
          func_clean_up=lambda r, prompt: r)

    # One request plus the three the budget paid for.
    self.assertEqual(4, gpt_structure.client.calls)
    self.assertEqual([2.0, 4.0, 8.0], self.slept)
    self.assertEqual(3, len(self.records("rate_limit")))

    rejects = self.records("reject")
    self.assertEqual(["transport:RateLimitError"],
                     [r["reason"] for r in rejects])

    # The whole of RB-C in two assertions: the fail-safe never reached the
    # world, and the loop never got to its last line.
    self.assertEqual([], self.records("outcome"))

  # --- case 15 (S1.2): the provider's own number beats our backoff ----------

  def test_the_provider_s_retry_after_is_obeyed(self):
    gpt_structure.client = Stub(error=rate_limited(retry_after=1.5),
                                text="kitchen", fails=1)

    gpt_structure.safe_generate_response(
        "where is Isabella?", self.PARAMS, repeat=5,
        fail_safe_response="bed",
        func_validate=lambda r, prompt: True,
        func_clean_up=lambda r, prompt: r)

    self.assertEqual([1.5], self.slept)
    self.assertEqual(1.5, self.records("rate_limit")[-1]["delay_s"])


class EmptyStub(Stub):
  """The router's ~300 s cut: a 200 with no choices, for the first <fails>."""

  def create(self, **kwargs):
    if self.calls < (self.fails or 0):
      self.calls += 1
      return type("R", (), {"choices": None, "usage": None, "model": None})()
    return Stub.create(self, **kwargs)


class TheRouterSEmptyAnswer(RideOut):
  """011-IP, the night of the windows: an empty body and a hung request are
  ridden out like a 429, and never reach the cost ledger."""

  def test_an_empty_response_is_ridden_out(self):
    gpt_structure.client = EmptyStub(text="kitchen", fails=2)
    answer = gpt_structure.safe_generate_response(
        "where is Isabella?", self.PARAMS, repeat=5,
        fail_safe_response="bed",
        func_validate=lambda r, prompt: True,
        func_clean_up=lambda r, prompt: r)
    self.assertEqual("kitchen", answer)
    self.assertEqual([2.0, 4.0], self.slept)
    waits = self.records("rate_limit")
    self.assertEqual([None, None], [w["status"] for w in waits])
    self.assertEqual(1, len(self.records("call")))

  def test_a_timeout_is_ridden_out(self):
    import httpx
    from openai import APITimeoutError
    timeout = APITimeoutError(request=httpx.Request(
        "POST", "https://openrouter.ai/api/v1/chat/completions"))
    gpt_structure.client = Stub(error=timeout, text="kitchen", fails=1)
    gpt_structure.safe_generate_response(
        "where is Isabella?", self.PARAMS, repeat=5,
        fail_safe_response="bed",
        func_validate=lambda r, prompt: True,
        func_clean_up=lambda r, prompt: r)
    self.assertEqual([2.0], self.slept)

  def test_a_502_a_412_and_a_404_are_ridden_out_and_a_402_is_not(self):
    import httpx
    from openai import InternalServerError, APIStatusError
    def err(cls, code):
      req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
      return cls("Error code: %d" % code, response=httpx.Response(code, request=req),
                 body=None)
    gpt_structure.client = Stub(error=err(InternalServerError, 502),
                                text="kitchen", fails=1)
    gpt_structure.safe_generate_response(
        "where is Isabella?", self.PARAMS, repeat=5, fail_safe_response="bed",
        func_validate=lambda r, prompt: True, func_clean_up=lambda r, prompt: r)
    gpt_structure.client = Stub(error=err(APIStatusError, 412),
                                text="kitchen", fails=1)
    gpt_structure.safe_generate_response(
        "where is Isabella?", self.PARAMS, repeat=5, fail_safe_response="bed",
        func_validate=lambda r, prompt: True, func_clean_up=lambda r, prompt: r)
    gpt_structure.client = Stub(error=err(APIStatusError, 404),
                                text="kitchen", fails=1)
    gpt_structure.safe_generate_response(
        "where is Isabella?", self.PARAMS, repeat=5, fail_safe_response="bed",
        func_validate=lambda r, prompt: True, func_clean_up=lambda r, prompt: r)
    self.assertEqual([502, 412, 404], [w["status"] for w in self.records("rate_limit")])
    gpt_structure.client = Stub(error=err(APIStatusError, 402))
    with self.assertRaises(gpt_structure.BackendTransportError):
      gpt_structure.safe_generate_response(
          "where is Isabella?", self.PARAMS, repeat=5, fail_safe_response="bed",
          func_validate=lambda r, prompt: True, func_clean_up=lambda r, prompt: r)
    # One wait each for the 502, the 412 and the 404, none for the 402.
    self.assertEqual([2.0, 2.0, 2.0], self.slept)

  def test_an_empty_response_that_never_ends_still_raises(self):
    gpt_structure.client = EmptyStub(fails=99)
    with self.assertRaises(gpt_structure.BackendTransportError):
      gpt_structure.safe_generate_response(
          "where is Isabella?", self.PARAMS, repeat=5,
          fail_safe_response="kitchen",
          func_validate=lambda r, prompt: True,
          func_clean_up=lambda r, prompt: r)
    self.assertEqual([], self.records("outcome"))


class ProviderStub(Stub):
  """Answers or fails by which provider the request is pinned to.

  <down> maps a provider slug to how many requests pinned to it fail (None:
  every one). Records the order of providers asked, which is the whole of
  what AP-3 changes."""

  def __init__(self, error, down, text="kitchen"):
    Stub.__init__(self, text=text)
    self.fail_error = error
    self.down = dict(down)
    self.asked = []

  def create(self, **kwargs):
    who = ((kwargs.get("extra_body") or {}).get("provider") or {}).get(
        "order", [None])[0]
    self.asked.append(who)
    if who in self.down:
      left = self.down[who]
      if left is None or left > 0:
        if left is not None:
          self.down[who] = left - 1
        self.calls += 1
        raise self.fail_error
    return Stub.create(self, **kwargs)


class OurSideFallback(Stubbed, unittest.TestCase):
  """AP-3: each request is pinned to one provider; this code moves it to the
  next only after the first failed for its whole budget, and back after a
  cooldown."""

  PIN = {"order": ["crusoe"], "allow_fallbacks": False}

  def setUp(self):
    Stubbed.setUp(self)
    self.slept = []
    self.now = [1000.0]
    self.saved = (gpt_structure._sleep, gpt_structure.rate_limit_policy,
                  gpt_structure.fallback_policy, gpt_structure._clock,
                  dict(gpt_structure._provider_state),
                  {k: gpt_structure.openai_config.get(k) for k in
                   ("model-provider", "model-provider-fallbacks")})
    gpt_structure._sleep = self._sleep
    gpt_structure._clock = lambda: self.now[0]
    gpt_structure.rate_limit_policy = {"attempts": 4, "base_delay": 2.0,
                                       "max_delay": 60.0}
    gpt_structure.fallback_policy = {"waits": 2, "return_after": 900.0}
    gpt_structure._provider_state.update({"index": 0, "since": None})
    gpt_structure.openai_config["model-provider"] = dict(self.PIN)
    gpt_structure.openai_config["model-provider-fallbacks"] = ["novita",
                                                               "friendli"]

  def _sleep(self, seconds):
    self.slept.append(seconds)
    self.now[0] += seconds

  def tearDown(self):
    (gpt_structure._sleep, gpt_structure.rate_limit_policy,
     gpt_structure.fallback_policy, gpt_structure._clock, state,
     config) = self.saved
    gpt_structure._provider_state.clear()
    gpt_structure._provider_state.update(state)
    for k, v in config.items():
      if v is None:
        gpt_structure.openai_config.pop(k, None)
      else:
        gpt_structure.openai_config[k] = v
    Stubbed.tearDown(self)

  def ask(self):
    return gpt_structure.safe_generate_response(
        "where is Isabella?", self.PARAMS, repeat=5, fail_safe_response="bed",
        func_validate=lambda r, prompt: True,
        func_clean_up=lambda r, prompt: r)

  def test_a_blip_never_moves_the_request(self):
    gpt_structure.client = ProviderStub(rate_limited(), {"crusoe": 2})
    self.assertEqual("kitchen", self.ask())
    self.assertEqual(["crusoe"] * 3, gpt_structure.client.asked)
    self.assertEqual([], self.records("provider_switch"))

  def test_a_sustained_failure_moves_to_the_next_and_says_so(self):
    gpt_structure.client = ProviderStub(rate_limited(), {"crusoe": None})
    self.assertEqual("kitchen", self.ask())
    # Two waits on crusoe (its budget), then at once to novita.
    self.assertEqual(["crusoe", "crusoe", "crusoe", "novita"],
                     gpt_structure.client.asked)
    self.assertEqual([2.0, 4.0], self.slept)
    switch = self.records("provider_switch")
    self.assertEqual([("crusoe", "novita", "fallback")],
                     [(r["from"], r["to"], r["reason"]) for r in switch])
    # The call record carries the pin it was actually sent with.
    self.assertEqual(["novita"],
                     self.records("call")[-1]["provider_pin"]["order"])

  def test_the_next_request_stays_on_the_fallback_then_returns(self):
    gpt_structure.client = ProviderStub(rate_limited(), {"crusoe": 3})
    self.ask()
    self.ask()
    self.assertEqual("novita", gpt_structure.client.asked[-1])
    self.now[0] += 901
    self.ask()
    self.assertEqual("crusoe", gpt_structure.client.asked[-1])
    self.assertEqual(["fallback", "return"],
                     [r["reason"] for r in self.records("provider_switch")])

  def test_the_last_provider_gets_the_whole_patience_then_it_raises(self):
    gpt_structure.client = ProviderStub(
        rate_limited(), {"crusoe": None, "novita": None, "friendli": None})
    with self.assertRaises(gpt_structure.BackendTransportError):
      self.ask()
    asked = gpt_structure.client.asked
    self.assertEqual(3, asked.count("crusoe"))
    self.assertEqual(3, asked.count("novita"))
    self.assertEqual(5, asked.count("friendli"))
    self.assertEqual([], self.records("outcome"))

  def test_a_permanent_error_does_not_move_the_request(self):
    import httpx
    from openai import APIStatusError
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    gpt_structure.client = ProviderStub(
        APIStatusError("Error code: 402", response=httpx.Response(402, request=req),
                       body=None), {"crusoe": None})
    with self.assertRaises(gpt_structure.BackendTransportError):
      self.ask()
    self.assertEqual(["crusoe"], gpt_structure.client.asked)

  def test_without_fallbacks_nothing_changes(self):
    gpt_structure.openai_config.pop("model-provider-fallbacks")
    gpt_structure.client = ProviderStub(rate_limited(), {"crusoe": 3})
    self.assertEqual("kitchen", self.ask())
    self.assertEqual(["crusoe"] * 4, gpt_structure.client.asked)
    self.assertEqual([], self.records("provider_switch"))


class Episodes(OwnTrace, unittest.TestCase):
  """A social decision says who it was about, and which encounter it was part
  of.

  What this replaces: in cmpD2-jev Isabella asked about Klaus 14 times between
  11:26 and 11:36 while Klaus asked about her 8 times in the same cafe, and
  the trace called that 22 decisions with no counterparty on any of them. The
  denominator of every social rate in 007-ST had to be reconstructed by
  joining on timestamps, and the join failed on 22 of 98.

  The backend here is "llm" (the suite sets DECISION_BACKEND before its
  imports), which is the arm law 2 binds: the counterparty and the episode are
  written into the trace and the answer that comes back is untouched."""

  def setUp(self):
    llm_trace.set_simulation(self.sim_prefix())
    decision._episodes.clear()

  def tearDown(self):
    llm_trace.set_step(None)
    decision._episodes.clear()

  def ask(self, subject, other, step, label="decide_to_talk", answer="yes"):
    """One social decision at one step, resolved by the paper's path."""
    llm_trace.set_step(step)
    return decision.choose(
        label, ["yes", "no"],
        succeeded(answer, "run_gpt_prompt_" + label),
        fail_safe="yes", counterparty=other,
        state={"subject": subject, "other": other})

  def decisions(self):
    return self.records("decision")

  # --- case 16 (S2.1): the decision names the other agent ------------------

  def test_a_social_decision_names_the_other_agent(self):
    answer = self.ask("Isabella Rodriguez", "Klaus Mueller", 4000)
    record = self.decisions()[-1]
    # The answer is the paper's, untouched: this is a trace change (law 2).
    self.assertEqual("yes", answer)
    self.assertEqual("Klaus Mueller", record["other"])
    self.assertEqual("Isabella Rodriguez", record["subject"])
    self.assertTrue(record["episode"])

  # --- case 17 (S2.2): one encounter is one episode ------------------------

  def test_consecutive_asks_about_a_pair_share_an_episode(self):
    # Isabella and Klaus in the cafe: the same question every step.
    for step in range(4000, 4014):
      self.ask("Isabella Rodriguez", "Klaus Mueller", step)
    episodes = set(r["episode"] for r in self.decisions())
    self.assertEqual(1, len(episodes))

    # And the other direction is its own ordered pair, as 007-ST counted it.
    self.ask("Klaus Mueller", "Isabella Rodriguez", 4005)
    self.assertNotIn(self.decisions()[-1]["episode"], episodes)

  # --- case 18 (S2.2): they part, they meet again, it is a second episode --

  def test_a_pair_that_meets_again_opens_a_new_episode(self):
    self.ask("Isabella Rodriguez", "Klaus Mueller", 4000)
    first = self.decisions()[-1]["episode"]

    # Ten steps of silence is still the same encounter (the default gap); the
    # eleventh is not. The cut is the one 007-ST read the 98 asks under.
    self.ask("Isabella Rodriguez", "Klaus Mueller", 4010)
    self.assertEqual(first, self.decisions()[-1]["episode"])

    self.ask("Isabella Rodriguez", "Klaus Mueller", 4030)
    second = self.decisions()[-1]["episode"]
    self.assertNotEqual(first, second)

    # Still the same encounter as the one that just opened.
    self.ask("Isabella Rodriguez", "Klaus Mueller", 4031)
    self.assertEqual(second, self.decisions()[-1]["episode"])

  # --- case 19 (S2.2): both social decisions of one encounter agree --------

  def test_talk_and_react_share_the_encounter(self):
    """_should_react asks decide_to_react only when decide_to_talk said no.
    Two questions, one pair of agents standing in one room -- and the episode
    is a fact about the room, not about the question."""
    self.ask("Isabella Rodriguez", "Klaus Mueller", 4000, answer="no")
    talk = self.decisions()[-1]["episode"]
    self.ask("Isabella Rodriguez", "Klaus Mueller", 4000,
             label="decide_to_react", answer="3")
    self.assertEqual(talk, self.decisions()[-1]["episode"])

  # --- case 20 (S2.3): a decision with no counterparty is untouched --------

  def test_a_location_decision_gains_nothing(self):
    """The five closed-set decisions that are not about another agent write
    exactly the record they wrote before this field existed."""
    llm_trace.set_step(4000)
    decision.choose("action_sector", SECTORS,
                    succeeded("kitchen", "run_gpt_prompt_action_sector"),
                    fail_safe="kitchen")
    record = self.decisions()[-1]
    for field in ("other", "subject", "episode"):
      self.assertNotIn(field, record)


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
    # Importing gpt_structure builds the run's cost logger, which opens a
    # ledger file under cost-logs/ the moment it is constructed. This suite
    # spends nothing; it does not get to leave a ledger behind claiming it
    # did.
    try:
      ledger = gpt_structure.cost_logger._OpenAICostLogger_Singleton__cost_logger.filepath
      os.remove(ledger)
    except Exception:
      pass
