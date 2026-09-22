"""
File: test_run_manifest.py
Description: The run is reproducible, and the trace says so itself.

Run from reverie/backend_server (the cwd the simulation uses):

    python test_run_manifest.py

Nothing here touches a model. Two claims are under test:

  1. The seed actually takes. Same seed, same draws -- including the draw
     the restored arena guard makes, which is the one this module was added
     for.
  2. Every trace file opens with a manifest saying which seed, which commit
     and which backend. A stage file that cannot identify itself is a row
     nobody can cite, and the automatic runner writes one file per stage.

The third claim is negative and matters as much: RUN_SEED=off records
"seed": null rather than omitting the field, so an unseeded run cannot be
mistaken for a seeded one when the tables are read back.
"""
import json
import os
import random
import shutil
import tempfile
import unittest

_TRACE_DIR = tempfile.mkdtemp(prefix="run_manifest_test_")
os.environ["LLM_TRACE_DIR"] = _TRACE_DIR
os.environ["LLM_TRACE_ENABLED"] = "1"
os.environ["DECISION_BACKEND"] = "llm"

from persona.prompt_template import llm_trace, run_manifest


class RunManifest(unittest.TestCase):

  def setUp(self):
    self.sim = "test-%s" % self.id().rsplit(".", 1)[-1]
    os.environ.pop("RUN_SEED", None)
    llm_trace.set_simulation(self.sim)

  def tearDown(self):
    os.environ.pop("RUN_SEED", None)

  def manifests(self):
    """The manifest line of every file this test's simulation wrote."""
    out = []
    for name in sorted(os.listdir(_TRACE_DIR)):
      if not name.startswith(self.sim):
        continue
      with open(os.path.join(_TRACE_DIR, name), "r", encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if line and json.loads(line).get("event") == "manifest":
            out.append(json.loads(line))
    return out

  # --- the seed -------------------------------------------------------------

  def test_same_seed_draws_the_same_sequence(self):
    run_manifest.apply(sim_code=self.sim)
    first = [random.random() for _ in range(5)]
    run_manifest.apply(sim_code=self.sim)
    self.assertEqual(first, [random.random() for _ in range(5)])

  def test_the_guards_draw_is_the_one_being_pinned(self):
    """domain_guard picks with random.choice over the offered set. Seeded,
    two runs send the persona to the same arena; that is the difference
    between a replicate and a new experiment."""
    arenas = ["main room", "bathroom"]
    run_manifest.apply(sim_code=self.sim)
    first = [random.choice(arenas) for _ in range(20)]
    run_manifest.apply(sim_code=self.sim)
    self.assertEqual(first, [random.choice(arenas) for _ in range(20)])

  def test_each_stage_draws_its_own_stream(self):
    """The runner builds a ReverieServer per 200-step stage inside one
    process. Seeding the run's seed at each one would make every stage
    repeat the same draws; the stage name is mixed in so they differ and
    stay deterministic."""
    a = run_manifest.apply(sim_code=self.sim + "-s-0-2360-2560")
    first = [random.random() for _ in range(5)]
    b = run_manifest.apply(sim_code=self.sim + "-s-1-2560-2760")
    second = [random.random() for _ in range(5)]
    self.assertEqual(a["seed"], b["seed"])
    self.assertNotEqual(a["stage_seed"], b["stage_seed"])
    self.assertNotEqual(first, second)

    # ... and the same stage name replays exactly.
    run_manifest.apply(sim_code=self.sim + "-s-0-2360-2560")
    self.assertEqual(first, [random.random() for _ in range(5)])

  def test_env_overrides_the_configured_seed(self):
    os.environ["RUN_SEED"] = "7"
    self.assertEqual(run_manifest.apply(sim_code=self.sim)["seed"], 7)

  def test_off_means_unseeded_and_says_so(self):
    os.environ["RUN_SEED"] = "off"
    manifest = run_manifest.apply(sim_code=self.sim)
    self.assertIn("seed", manifest)
    self.assertIsNone(manifest["seed"])
    self.assertEqual(manifest["seeded"], [])
    self.assertIn("UNSEEDED", run_manifest.status())

  def test_a_junk_seed_falls_back_instead_of_killing_the_run(self):
    os.environ["RUN_SEED"] = "not-a-number"
    self.assertIsNotNone(run_manifest.apply(sim_code=self.sim)["seed"])

  # --- the manifest ---------------------------------------------------------

  def test_every_trace_file_opens_with_a_manifest(self):
    run_manifest.apply(fork_sim_code="origin-sim", sim_code=self.sim)
    llm_trace.record_decision("action_arena", ["main room"], "main room",
                              backend="llm")
    # A stage boundary: the runner renames the simulation and the trace
    # rotates. The new file has to identify itself too.
    llm_trace.set_simulation(self.sim + "-s-1")
    llm_trace.record_decision("action_arena", ["main room"], "main room",
                              backend="llm")
    llm_trace.set_simulation(self.sim)

    found = self.manifests()
    self.assertEqual(len(found), 2, "one manifest per trace file")
    for manifest in found:
      self.assertEqual(manifest["seed"], 20260922)
      self.assertEqual(manifest["fork_sim_code"], "origin-sim")
      self.assertEqual(manifest["backend"]["decision"], "llm")
      self.assertIn("code", manifest)
      self.assertIn("config_sha1", manifest)
    self.assertNotEqual(found[0]["sim"], found[1]["sim"])

  def test_the_manifest_is_the_first_line(self):
    run_manifest.apply(sim_code=self.sim)
    llm_trace.record_decision("action_sector", ["cafe"], "cafe",
                              backend="llm")
    path = [n for n in sorted(os.listdir(_TRACE_DIR))
            if n.startswith(self.sim)][0]
    with open(os.path.join(_TRACE_DIR, path), "r", encoding="utf-8") as f:
      self.assertEqual(json.loads(f.readline())["event"], "manifest")

  def test_the_fingerprint_leaves_the_key_out(self):
    """The point of the fingerprint is that it can be published next to the
    numbers, so it must not be computed over the api-key."""
    cfg = {"jev": {"model": "jev-latest", "api-key": "secret-one"}}
    other = {"jev": {"model": "jev-latest", "api-key": "secret-two"}}
    self.assertEqual(run_manifest._config_fingerprint(cfg),
                     run_manifest._config_fingerprint(other))
    changed = {"jev": {"model": "jev-next", "api-key": "secret-one"}}
    self.assertNotEqual(run_manifest._config_fingerprint(cfg),
                        run_manifest._config_fingerprint(changed))

  def test_status_carries_the_commit_and_the_seed(self):
    run_manifest.apply(sim_code=self.sim)
    line = run_manifest.status()
    self.assertIn("seed 20260922", line)
    self.assertIn("run:", line)


if __name__ == "__main__":
  try:
    unittest.main(verbosity=2, exit=False)
  finally:
    shutil.rmtree(_TRACE_DIR, ignore_errors=True)
