"""
File: run_manifest.py
Description: What identifies one run -- the seed it was drawn with, and the
             code and configuration that produced its numbers.

Two things that are really one thing.

The simulation draws without a seed all over the place: which game object a
persona walks to (execute.py), which path it takes (path_finder), and, since
the arena domain guard was restored, which arena replaces an out-of-domain
answer (domain_guard.py). Temperature 0 makes the model's answers stable; it
does not make the run stable. Until this module, two runs of the same window
on the same code could differ and there was no way to tell that apart from a
real effect.

So: seed the generators at boot, and write down what the run was, in the
trace itself, where the numbers are. Every trace file opens with one
"manifest" record -- including each stage file the automatic runner rotates
through, because a stage is the unit anyone actually reads back.

What goes in it is what a reader needs to rerun the row: the commit, whether
the tree was dirty, the models, the provider the requests were pinned to,
the decision backend, whether the domain guard was on, the hash of the
configuration, and the seed.

The pin is the condition; which provider actually ANSWERED is not here,
because it is not a property of the run's setup -- it is a property of every
individual call, and it lives on each "call" record. The registrar in the
lab reads both and refuses a run where they ever disagree.

Configuration, in openai_config.json:

  "run": { "seed": 20260922 }

Environment override: RUN_SEED=<int>, or RUN_SEED=off to draw unseeded the
way every run before 2026-09-22 did. An unseeded run is recorded as
"seed": null, so no trace can quietly claim a reproducibility it does not
have. Replicates of the same condition vary this and nothing else.

One wrinkle from the automatic runner: it builds a fresh ReverieServer for
every 200-step stage, in one process, so seeding the run's seed here would
restart the same stream every stage and every stage's first draw would be
the same draw. The generator is therefore seeded with the run's seed mixed
with the stage's simulation name -- deterministic, since the stage names are
a function of the window, and decorrelated across stages. Both numbers go in
the manifest: "seed" is the run's, "stage_seed" is what was handed to
random.seed().
"""
import binascii
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys

from persona.prompt_template import decision
from persona.prompt_template import domain_guard
from persona.prompt_template import llm_trace

_CONFIG_PATH = "../../openai_config.json"
_DEFAULT_SEED = 20260922

_manifest = {}
_seed = None
_stage = None
_seeded = []


def _config():
  try:
    with open(_CONFIG_PATH, "r") as f:
      return json.load(f)
  except Exception:
    return {}


def _load_seed(cfg):
  """The seed this run should use, or None to draw unseeded."""
  seed = (cfg.get("run") or {}).get("seed", _DEFAULT_SEED)

  flag = os.environ.get("RUN_SEED")
  if flag is not None:
    flag = flag.strip().lower()
    if flag in ("", "off", "none", "no", "false", "random"):
      return None
    try:
      return int(flag)
    except ValueError:
      print("run_manifest: RUN_SEED=%r is not an integer; ignoring it" % flag)
      return seed

  if seed is None:
    return None
  try:
    return int(seed)
  except (TypeError, ValueError):
    print("run_manifest: \"run\": {\"seed\": %r} is not an integer; "
          "using %d" % (seed, _DEFAULT_SEED))
    return _DEFAULT_SEED


def _stage_seed(seed, sim_code):
  """The run's seed, mixed with the stage this process is about to run."""
  if seed is None:
    return None
  if not sim_code:
    return seed
  mix = binascii.crc32(str(sim_code).encode("utf-8")) & 0xffffffff
  return (int(seed) ^ mix) & 0x7fffffff


def _git(*args):
  """Ask git, from the directory this file lives in, and never raise."""
  try:
    out = subprocess.check_output(
        ("git",) + args,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stderr=subprocess.DEVNULL)
    return out.decode("utf-8", "replace").strip()
  except Exception:
    return None


def _code_version():
  """The commit the numbers came from, and whether it is the whole story.

  Two identifiers, because they answer different questions.

  "sha" is the commit: where this run sits in the history. It moves for any
  reason at all, including a commit that touches only documentation -- which
  is what happened on 22 Sep 2026, when two docs commits landed from another
  session during cmpD2-llm's 1h14 and split its eight stages across two
  commits. The comparison called the condition MIXED, correctly by its own
  rule and wrongly about the world: not one line of Python had changed.

  "tree" is the hash of reverie/backend_server as git stores it: the code
  that actually ran, and nothing else. Two runs that share it ran identical
  code no matter what their commits say. That is the one to compare on.

  A dirty tree is recorded rather than refused: a run mid-edit is a normal
  thing to do, and a row whose manifest says dirty=true is a row nobody will
  mistake for a citable one. Note that the tree hash is of HEAD, so it does
  not describe uncommitted edits -- which is exactly why "dirty" is reported
  beside it and why a dirty run is not a citable one.
  """
  sha = _git("rev-parse", "HEAD")
  status = _git("status", "--porcelain")
  return {
      "sha": sha,
      "tree": _git("rev-parse", "HEAD:reverie/backend_server"),
      "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
      "dirty": bool(status) if status is not None else None,
  }


def _config_fingerprint(cfg):
  """A hash of the configuration with the secrets taken out.

  Two runs whose manifests share this ran against the same settings; the
  api-key is removed first so the fingerprint can be published.
  """
  try:
    scrubbed = json.loads(json.dumps(cfg))
    for section in scrubbed.values():
      if isinstance(section, dict):
        section.pop("api-key", None)
    blob = json.dumps(scrubbed, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
  except Exception:
    return None


def _no_fence(cfg):
  """The same rule gpt_structure.no_fence_enabled applies, without importing
  the client at manifest time."""
  flag = os.environ.get("JSON_NO_FENCE")
  if flag is not None:
    return flag.strip().lower() in ("1", "true", "yes", "on")
  return bool((cfg.get("json-no-fence") or {}).get("enabled", False))


# The paper's Ville, as every run before the first designed world ran it. TR-1: a world is not
# a deviation, it is a different experiment, and a number written without
# its world is a bug. So the field is never absent: a config with no "world"
# section IS the paper's world, and says so.
PAPER_WORLD = "W1"


def world_of(cfg):
  """The world a run is in: the config's "world": {"name": ...}, or W1 when
  there is no world section. A world section with no name is refused -- a
  run that changed the world and did not say which one cannot be cited."""
  section = cfg.get("world")
  if section is None:
    return PAPER_WORLD
  name = (section or {}).get("name") if isinstance(section, dict) else None
  if not name:
    raise ValueError("openai_config.json: a \"world\" section needs a "
                     "\"name\" (e.g. \"W2\"); with no section the run is "
                     "the paper's %s" % PAPER_WORLD)
  return str(name)


def _backend_section(cfg):
  """How decisions resolve, asked of the seam rather than of the file, so the
  manifest cannot disagree with what the run actually did."""
  try:
    backend = decision.backend()
    options = decision.options() or {}
  except Exception:
    return {"decision": None}

  out = {
      "decision": backend,
      "noul_threshold": options.get("noul-threshold"),
      "on_error": options.get("on-error"),
  }
  if backend == "jev":
    try:
      from persona.prompt_template import jev
      out["jev_model"] = jev.model()
    except Exception:
      out["jev_model"] = (cfg.get("jev") or {}).get("model")
  return out


def apply(fork_sim_code=None, sim_code=None, steps=None):
  """Seed the generators and assemble the manifest. Called once, at boot,
  before the first step and before the first trace file is opened.

  RETURNS:
    The manifest dict, also handed to llm_trace so every trace file this run
    opens begins with it.
  """
  global _manifest, _seed, _seeded, _stage

  cfg = _config()
  _seed = _load_seed(cfg)
  _stage = _stage_seed(_seed, sim_code)
  _seeded = []

  if _stage is not None:
    random.seed(_stage)
    _seeded.append("random")
    try:
      import numpy
      numpy.random.seed(_stage % (2 ** 32))
      _seeded.append("numpy")
    except Exception:
      pass

  _manifest = {
      "event": "manifest",
      # Which world this run is in. Read by the lab's registrar
      # and by compare_backends.py, which refuses a pair across two worlds.
      "world": world_of(cfg),
      "seed": _seed,
      "stage_seed": _stage,
      "seeded": _seeded,
      "fork_sim_code": fork_sim_code,
      "steps_to": steps,
      "code": _code_version(),
      "backend": _backend_section(cfg),
      "model": {
          "name": cfg.get("model"),
          "base_url": cfg.get("model-base-url"),
          # The provider block sent with every request, when the run goes
          # through a router. It is half of the run's identity there: the
          # same model name at two providers is two sets of weights, two
          # quantizations and two serving stacks. Null on a run that
          # does not route.
          "provider_pin": cfg.get("model-provider"),
          # AP-3: the providers this run's own code may move a request to,
          # in order, after the pinned one fails for its budget. Absent
          # when none are configured.
          "provider_fallbacks": cfg.get("model-provider-fallbacks") or None,
          "embeddings": cfg.get("embeddings"),
          "embeddings_base_url": cfg.get("embeddings-base-url"),
      },
      "domain_guard": _safe(domain_guard.enabled),
      # D11 can be switched from the environment, which the config
      # fingerprint cannot see; the effective value is recorded so two runs
      # whose JSON prompts differ never share an identity.
      "json_no_fence": _no_fence(cfg),
      # D13: the history file whose phrases the runner loads as the
      # paper's seed memories in a run's first stage, or null. Every run
      # before 2026-10-03 started without them.
      "seed_history": cfg.get("seed-history") or None,
      "config_sha1": _config_fingerprint(cfg),
      "python": platform.python_version(),
      "platform": "%s %s" % (platform.system().lower(), platform.machine()),
      "host": _safe(socket.gethostname),
      "argv": " ".join(sys.argv),
  }
  llm_trace.set_manifest(_manifest)
  return _manifest


def _safe(fn):
  try:
    return fn()
  except Exception:
    return None


def manifest():
  """The manifest as applied, empty before apply() has run."""
  return dict(_manifest)


def seed():
  """The seed in force, or None if this run draws unseeded."""
  return _seed


def status():
  """One line for the run's startup, next to decision.check_ready() and
  domain_guard.status()."""
  code = (_manifest.get("code") or {})
  sha = code.get("sha") or "unknown"
  # The commit is what a person recognises; the tree is what the comparison
  # trusts. The startup line prints the commit, since it is read by people.
  where = "%s%s" % (sha[:8], "+dirty" if code.get("dirty") else "")
  if _seed is None:
    return ("run: %s, UNSEEDED -- this run is not reproducible "
            "(set RUN_SEED or \"run\": {\"seed\": N})" % where)
  return ("run: %s, seed %d -> stage seed %d (%s)"
          % (where, _seed, _stage, ", ".join(_seeded) or "none"))
