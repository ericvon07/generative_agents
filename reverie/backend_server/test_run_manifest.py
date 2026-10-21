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

Since the router there is a fourth: a run that goes through a router records the
provider block it pinned, and every call records the provider that actually
answered. They are two different facts -- what was asked for and what
happened -- and the registrar in the lab invalidates a run where they ever
disagree, because the same model name at two providers is two sets of
weights and two serving stacks.
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

  def calls(self):
    """Every call record this test's simulation wrote."""
    out = []
    for name in sorted(os.listdir(_TRACE_DIR)):
      if not name.startswith(self.sim):
        continue
      with open(os.path.join(_TRACE_DIR, name), "r", encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if line and json.loads(line).get("event") == "call":
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

  def test_the_manifest_names_the_tree_as_well_as_the_commit(self):
    """A commit moves for reasons that have nothing to do with the
    simulator -- two documentation commits from another session split
    cmpD2-llm's eight stages across two shas on 22 Sep 2026 while the code
    was identical. The tree of reverie/backend_server is what the comparison
    compares on."""
    code = run_manifest.apply(sim_code=self.sim)["code"]
    self.assertIn("tree", code)
    if code["sha"] is not None:  # a checkout without git still runs
      self.assertIsNotNone(code["tree"])
      self.assertNotEqual(code["tree"], code["sha"])

  # --- the provider ---------------------------------------------------------

  def test_the_manifest_carries_the_pin_it_was_configured_with(self):
    """The pin is half a routed run's identity. The same model name at two
    providers is two sets of weights, two quantizations and two serving
    stacks, so a run that does not record what it asked for cannot say what
    it measured."""
    pin = {"order": ["deepinfra"], "allow_fallbacks": False,
           "quantizations": ["fp8"]}
    real = run_manifest._config
    run_manifest._config = lambda: {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "model-base-url": "https://openrouter.ai/api/v1",
        "model-provider": pin,
        "run": {"seed": 20260922},
    }
    try:
      manifest = run_manifest.apply(sim_code=self.sim)
    finally:
      run_manifest._config = real
    self.assertEqual(pin, manifest["model"]["provider_pin"])

  def test_a_local_run_pins_nothing_and_says_so(self):
    """Null, not absent and not a guess: a run that does not route asked for
    no provider, and that is a fact about it rather than a gap."""
    real = run_manifest._config
    run_manifest._config = lambda: {"model": "llama3.1:8b",
                                    "run": {"seed": 20260922}}
    try:
      manifest = run_manifest.apply(sim_code=self.sim)
    finally:
      run_manifest._config = real
    self.assertIn("provider_pin", manifest["model"])
    self.assertIsNone(manifest["model"]["provider_pin"])

  # --- the world -----------------------------------------

  def _apply_with(self, cfg):
    real = run_manifest._config
    run_manifest._config = lambda: cfg
    try:
      return run_manifest.apply(sim_code=self.sim)
    finally:
      run_manifest._config = real

  def test_no_world_section_is_the_papers_world(self):
    """Every run before the first designed world was the paper's Ville; the field says so
    rather than being absent."""
    manifest = self._apply_with({"run": {"seed": 20260922}})
    self.assertEqual("W1", manifest["world"])

  def test_a_world_section_names_its_world(self):
    manifest = self._apply_with({"run": {"seed": 20260922},
                                 "world": {"name": "W2"}})
    self.assertEqual("W2", manifest["world"])

  def test_a_world_section_with_no_name_is_refused(self):
    """A run that changed the world and did not say which one could not be
    cited, so it does not start."""
    with self.assertRaises(ValueError):
      self._apply_with({"run": {"seed": 20260922}, "world": {"maze": "x"}})

  def test_the_comparer_refuses_a_pair_across_worlds(self):
    """S1.3: two worlds side by side is refused, not warned."""
    import compare_backends as cb

    class Fake(object):
      def __init__(self, prefix, world):
        self.prefix = prefix
        self.manifests = [{"world": world}] if world else []
      def provenance(self, *path):
        vals = set(m.get(path[0]) for m in self.manifests)
        return vals.pop() if len(vals) == 1 else ("MIXED" if vals else None)

    cb.check_worlds([Fake("a", "W1"), Fake("b", "W1")])
    cb.check_worlds([Fake("a", None), Fake("b", "W1")])  # pre-field trace is W1
    with self.assertRaises(cb.CrossWorldPair):
      cb.check_worlds([Fake("a", "W1"), Fake("b", "W2")])

  def test_a_call_records_what_answered_beside_what_was_asked_for(self):
    """The pin is what the request asked for; "provider" is what the response
    says actually served it. They ride on the same record precisely so a
    reader can watch them disagree -- the registrar invalidates a run where
    they ever do."""
    pin = {"order": ["deepinfra"], "allow_fallbacks": False}
    run_manifest.apply(sim_code=self.sim)

    message = type("M", (), {"content": "kitchen"})()
    choice = type("C", (), {"message": message, "finish_reason": "stop"})()
    usage = type("U", (), {"prompt_tokens": 9, "completion_tokens": 1,
                           "total_tokens": 10, "cost": 0.00000064})()
    response = type("R", (), {"choices": [choice], "usage": usage,
                              "provider": "DeepInfra"})()

    llm_trace.record_call("GPT_request", "meta-llama/llama-3.1-8b-instruct",
                          "where is Isabella?", response=response,
                          extra={"provider_pin": pin})

    call = self.calls()[-1]
    self.assertEqual("DeepInfra", call["provider"])
    self.assertEqual(pin, call["provider_pin"])
    self.assertEqual("stop", call["finish_reason"])
    # And the router's own number for what the call cost, which is what the
    # ledger books rather than a price table.
    self.assertEqual(0.00000064,
                     llm_trace.response_facts(response)["cost"])

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
