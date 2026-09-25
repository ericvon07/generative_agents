"""Which failures of the typed backend are worth trying again.

**The scar.** `n7win`, the eight-hour seven-agent window of 2026-09-23, lost
five decisions to the paper's fail-safes between 14:11:59 and 14:16:48. The
backend was answering `HTTP 529 {"error_type": "system_overloaded"}` — a
provider having a bad five minutes, the same shape of failure as the `429`
that SV-A of 008-IP taught the model path to ride out. One of the five was a
`decide_to_talk`, and its fail-safe is `"yes"`: the run recorded a
conversation that no model had agreed to. That is the pathology this whole
lab exists to measure, happening inside the run that measures it.

`jev.py` did not retry once. Not because three retries were too few — because
`529` was not in `_RETRYABLE_STATUS`, so `_retryable` said no, the loop broke
on the first attempt, and `on-error: failsafe` did exactly what it says.

529 is not in any RFC. Anthropic and TypeSafe both use it for "overloaded,
come back shortly", which is the definition of a status worth retrying, and
the neighbours already in the list (429, 500, 502, 503, 504) are there for
the same reason.

Run from reverie/backend_server:

    python test_jev_transport.py
"""
import io
import json
import os
import shutil
import tempfile
import unittest
import urllib.error

_TRACE_DIR = tempfile.mkdtemp(prefix="jev_transport_test_")
os.environ["LLM_TRACE_DIR"] = _TRACE_DIR
os.environ["LLM_TRACE_ENABLED"] = "1"

from persona.prompt_template import jev


ANSWER = {"answers": {"q": {"type": "noul", "probability": 0.23}},
          "usage": {"input_tokens": 100}, "model": "jev-latest"}


def http_error(code, message="overloaded"):
  body = json.dumps({"detail": {"error_type": "system_overloaded",
                                "message": message}}).encode("utf-8")
  return urllib.error.HTTPError(
      "https://api.typesafe.ai/v1/systemone", code, message, {},
      io.BytesIO(body))


class Urlopen(object):
  """Stands in for urllib.request.urlopen: fails a bounded number of times,
  then answers. Shaped like the context manager _post uses and nothing more."""

  def __init__(self, error=None, fails=0):
    self.error = error
    self.fails = fails
    self.calls = 0

  def __call__(self, request, timeout=None):
    self.calls += 1
    if self.error is not None and self.calls <= self.fails:
      raise self.error
    return self

  def __enter__(self):
    return self

  def __exit__(self, *a):
    return False

  def read(self):
    return json.dumps(ANSWER).encode("utf-8")


class Transport(unittest.TestCase):

  def setUp(self):
    self.real_urlopen = jev.urllib.request.urlopen
    self.real_sleep = jev.time.sleep
    self.real_key = os.environ.get(jev._config["api-key-env"])
    os.environ[jev._config["api-key-env"]] = "test-key"
    self.slept = []
    # The wait is the point of the retry, so it cannot be tested by waiting.
    jev.time.sleep = self.slept.append
    jev._consecutive_errors = 0

  def tearDown(self):
    jev.urllib.request.urlopen = self.real_urlopen
    jev.time.sleep = self.real_sleep
    if self.real_key is None:
      os.environ.pop(jev._config["api-key-env"], None)
    else:
      os.environ[jev._config["api-key-env"]] = self.real_key
    jev._consecutive_errors = 0

  def ask(self):
    return jev.ask({"subject": "Isabella"},
                   {"q": jev.noul("Isabella would start a conversation.")},
                   label="decide_to_talk")

  # --- the scar -------------------------------------------------------------

  def test_a_529_is_retried(self):
    """n7win's case: the provider is overloaded for a moment and comes back.
    A decision must not become the paper's fail-safe over it."""
    stub = Urlopen(error=http_error(529), fails=2)
    jev.urllib.request.urlopen = stub

    result = self.ask()
    self.assertEqual(3, stub.calls)          # two refusals, one answer
    self.assertEqual([2.0, 4.0], self.slept)  # and it waited between them
    # ask() hands back the answers map itself; ask_full() is the envelope.
    self.assertIn("q", result)

  # --- the statuses that were already right ---------------------------------

  def test_a_429_is_retried(self):
    stub = Urlopen(error=http_error(429, "rate limited"), fails=1)
    jev.urllib.request.urlopen = stub
    self.ask()
    self.assertEqual(2, stub.calls)

  def test_a_403_is_not_retried(self):
    """GT-1's case. An unidentified client is refused the same way every
    time, and spending three attempts on it only makes the run slower."""
    stub = Urlopen(error=http_error(403, "error code: 1010"), fails=99)
    jev.urllib.request.urlopen = stub
    with self.assertRaises(jev.JevError):
      self.ask()
    self.assertEqual(1, stub.calls)
    self.assertEqual([], self.slept)

  def test_a_400_is_not_retried(self):
    """A malformed question is malformed on every attempt."""
    stub = Urlopen(error=http_error(400, "bad request"), fails=99)
    jev.urllib.request.urlopen = stub
    with self.assertRaises(jev.JevError):
      self.ask()
    self.assertEqual(1, stub.calls)

  # --- how long the backend is given ----------------------------------------

  def test_the_typed_path_waits_as_long_as_the_model_path(self):
    """SV-A gave the model path five waits of 2, 4, 8, 16 and 32 seconds --
    just over a minute against a provider whose own message says "retry
    shortly". The typed path had three attempts and waits of 1 and 2: three
    seconds of patience, against a service that was unstable for
    thirty-five minutes on 2026-09-23. The two paths face the same weather
    and now have the same coat."""
    stub = Urlopen(error=http_error(529), fails=5)
    jev.urllib.request.urlopen = stub

    self.ask()
    self.assertEqual(6, stub.calls)   # one request plus five retries
    self.assertEqual([2.0, 4.0, 8.0, 16.0, 32.0], self.slept)

  # --- the budget still ends ------------------------------------------------

  def test_a_529_that_persists_still_raises(self):
    """The ride is bounded. When it is over the caller decides between the
    fail-safe and stopping, exactly as before -- this changes how long the
    backend is given, never what happens when it is truly down."""
    stub = Urlopen(error=http_error(529), fails=99)
    jev.urllib.request.urlopen = stub
    with self.assertRaises(jev.JevError):
      self.ask()
    self.assertEqual(jev._config["max-retries"], stub.calls)


if __name__ == "__main__":
  try:
    unittest.main(exit=False, verbosity=2)
  finally:
    shutil.rmtree(_TRACE_DIR, ignore_errors=True)
