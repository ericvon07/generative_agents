"""
File: test_seed_history.py
Description: D13, the paper's seed memories loaded at the start of a run
             (016-IP). What has to hold: the kept rows are the paper's, in
             the file's order, for the agents present and no one else; an
             empty phrase is not whispered; a persona's unset clock is lent
             for the load and handed back null, so its first step is still
             the paper's "First day"; the load is written to the trace.

Run from reverie/backend_server (the cwd the simulation uses):

    python test_seed_history.py

Nothing here touches a model: the paper's whisper is replaced by a recorder.
The last case reads the release's own file, to pin what the village of nine
receives from it.
"""
import datetime
import glob
import json
import os
import shutil
import tempfile
import unittest

_TRACE_DIR = tempfile.mkdtemp(prefix="seed_history_test_")
os.environ["LLM_TRACE_DIR"] = _TRACE_DIR
os.environ["LLM_TRACE_ENABLED"] = "1"
os.environ["DECISION_BACKEND"] = "llm"

from persona.prompt_template import llm_trace
import seed_history

PAPER_FILE = "the_ville/agent_history_init_n25.csv"
NINE = ["Isabella Rodriguez", "Klaus Mueller", "Maria Lopez", "Ayesha Khan",
        "Wolfgang Schulz", "Eddy Lin", "Mei Lin", "Sam Moore", "Carmen Ortiz"]
CLOCK = datetime.datetime(2023, 2, 13, 0, 0, 0)


class _Scratch(object):
  def __init__(self, curr_time=None):
    self.curr_time = curr_time


class _Persona(object):
  def __init__(self, curr_time=None):
    self.scratch = _Scratch(curr_time)


class _Server(object):
  def __init__(self, names, curr_time=None):
    self.personas = dict((n, _Persona(curr_time)) for n in names)
    self.step = 0
    self.curr_time = CLOCK


def _csv(rows):
  d = tempfile.mkdtemp(prefix="seed_csv_")
  path = os.path.join(d, "h.csv")
  with open(path, "w") as f:
    f.write("Name,Whisper\n")
    for name, text in rows:
      f.write('%s,"%s"\n' % (name, text))
  return d, path


class SeedHistory(unittest.TestCase):

  def setUp(self):
    llm_trace.set_simulation("test-%s" % self.id().rsplit(".", 1)[-1])

  def test_rows_of_absent_agents_are_skipped_and_order_kept(self):
    d, path = _csv([("Ann", "a1; a2"), ("Bob", "b1"), ("Cid", "c1;c2")])
    try:
      kept, skipped = seed_history.rows_for(path, {"Ann", "Cid"})
    finally:
      shutil.rmtree(d)
    self.assertEqual(kept, [["Ann", "a1"], ["Ann", "a2"],
                            ["Cid", "c1"], ["Cid", "c2"]])
    self.assertEqual(skipped, ["Bob"])

  def test_an_empty_phrase_is_not_whispered(self):
    d, path = _csv([("Sam", "s1; s2;")])
    try:
      kept, _ = seed_history.rows_for(path, {"Sam"})
    finally:
      shutil.rmtree(d)
    self.assertEqual(kept, [["Sam", "s1"], ["Sam", "s2"]])

  def test_the_clock_is_lent_for_the_load_and_handed_back(self):
    rs = _Server(NINE)
    seen = []

    def whisper(personas, rows):
      for agent, phrase in rows:
        seen.append((agent, personas[agent].scratch.curr_time))

    seed_history.load(rs, PAPER_FILE, whisper=whisper)
    self.assertTrue(seen)
    self.assertTrue(all(t == CLOCK for _, t in seen))
    for p in rs.personas.values():
      self.assertIsNone(p.scratch.curr_time)

  def test_a_set_clock_is_left_alone(self):
    later = CLOCK + datetime.timedelta(hours=1)
    rs = _Server(NINE, curr_time=later)
    summary = seed_history.load(rs, PAPER_FILE, whisper=lambda p, r: None)
    self.assertEqual(summary["clock_lent_to"], [])
    for p in rs.personas.values():
      self.assertEqual(p.scratch.curr_time, later)

  def test_the_clock_is_handed_back_when_the_load_raises(self):
    rs = _Server(NINE)

    def whisper(personas, rows):
      raise RuntimeError("transport")

    with self.assertRaises(RuntimeError):
      seed_history.load(rs, PAPER_FILE, whisper=whisper)
    for p in rs.personas.values():
      self.assertIsNone(p.scratch.curr_time)

  def test_the_village_of_nine_gets_95_phrases_and_the_trace_says_so(self):
    rs = _Server(NINE)
    summary = seed_history.load(rs, PAPER_FILE, whisper=lambda p, r: None)
    self.assertEqual(summary["phrases"], 95)
    self.assertEqual(sorted(summary["per_agent"]), sorted(NINE))
    self.assertEqual(len(summary["skipped_agents"]), 16)
    self.assertEqual(summary["per_agent"]["Sam Moore"], 12)
    llm_trace._close_writer()
    records = []
    for f in glob.glob(os.path.join(_TRACE_DIR, "*.jsonl")):
      with open(f) as fh:
        records += [json.loads(l) for l in fh if l.strip()]
    seeds = [r for r in records if r.get("event") == "seed_history"]
    self.assertTrue(any(r["phrases"] == 95 for r in seeds))

  def test_configured_reads_the_key(self):
    self.assertEqual(seed_history.configured({"seed-history": PAPER_FILE}),
                     PAPER_FILE)
    self.assertIsNone(seed_history.configured({}))
    self.assertIsNone(seed_history.configured(None))


if __name__ == "__main__":
  try:
    unittest.main()
  finally:
    shutil.rmtree(_TRACE_DIR, ignore_errors=True)
