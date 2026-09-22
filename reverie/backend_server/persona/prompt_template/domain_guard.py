"""
File: domain_guard.py
Description: The authors' own domain guard for a closed-set choice, restored.

The paper's three location choices -- sector, arena, game object -- are
written the same way, and all three were given a domain guard: if the answer
is not one of the options the prompt offered, replace it with one that is.
Two of them are live (run_gpt_prompt.py:684-688 and :867-869). The arena's is
commented out, and has been since upstream 0db6021f.

The reason is visible in defunct_run_gpt_prompt.py:612-615, the authors'
earlier version of that file, still in the tree and imported by nothing. There
the arena guard is active, create_prompt_input has no arena filtering, and the
template is action_location_object_v1.txt: the prompt offered the raw arena
list and the guard checked the raw arena list. The two sets were the same set.
The live version added the "MAR 11 TEMP" filter, which drops other personas'
bedrooms from the options the prompt offers, and switched to
action_location_object_vMar11.txt. The guard's raw list stopped matching the
prompt's filtered list, and the guard was commented out rather than updated.

So the invariant that made the guard correct -- it checks the set the prompt
offered -- was broken elsewhere. This module restores it: same mechanism
(random.choice over the valid set), the set being the one the call site
actually interpolated into its prompt.

What it does not do is correct the measurement. decision.py records the
answer as the model gave it, with in_domain computed on that answer, before
this guard runs; a substitution is a "guard" event of its own. The
out-of-domain rate this project reports is unaffected by whether the guard is
on.

Configuration, in openai_config.json:

  "domain-guard": { "enabled": true }

Environment override: DOMAIN_GUARD_ENABLED=0|1. Off, the answer is returned
untouched and the simulation behaves exactly as it did before this module
existed -- which is how the original KeyError stays reproducible.
"""
import json
import os
import random

from persona.prompt_template import decision
from persona.prompt_template import llm_trace

_DEFAULT_ENABLED = True


def _load_enabled():
  enabled = _DEFAULT_ENABLED
  try:
    with open("../../openai_config.json", "r") as f:
      cfg = json.load(f)
    enabled = bool((cfg.get("domain-guard") or {}).get("enabled",
                                                       _DEFAULT_ENABLED))
  except Exception:
    pass

  flag = os.environ.get("DOMAIN_GUARD_ENABLED")
  if flag is not None:
    enabled = flag.strip().lower() not in ("0", "false", "no", "off", "")
  return enabled


_enabled = _load_enabled()


def enabled():
  """Whether an out-of-domain answer will be replaced in this run."""
  return _enabled


def set_enabled(flag):
  """Turn the guard on or off after import. The environment variable is how a
  run does it; this exists for the tests, and for an arm of the experiment
  that wants both behaviours inside one process."""
  global _enabled
  _enabled = bool(flag)


def status():
  """One line for the run's startup, next to decision.check_ready()."""
  if _enabled:
    return ("domain guard: on (out-of-domain answers are replaced and "
            "recorded; DOMAIN_GUARD_ENABLED=0 turns it off)")
  return ("domain guard: OFF -- an out-of-domain answer will reach the maze "
          "and can end the run")


def keep_in_domain(label, answer, domain):
  """Return <answer> if it is one of <domain>, else one of <domain>.

  ARGS:
    label: the decision's name in the trace, e.g. "action_arena".
    answer: what the decision resolved to.
    domain: the closed set the prompt offered, as a list.

  RETURNS:
    The answer the simulation should act on. An answer already inside the set
    comes back untouched and nothing is recorded: the guard is invisible
    until it is needed.
  """
  if not _enabled or not domain:
    return answer
  # The same membership test the trace used to compute in_domain, so the
  # guard cannot fire on an answer the trace called valid, or stay quiet on
  # one it called invalid.
  if decision.in_domain(answer, domain):
    return answer

  domain = list(domain)
  replacement = random.choice(domain)
  print("domain guard: %s answered %r, which is not one of %s -- using %r"
        % (label, answer, domain, replacement))
  llm_trace.record_guard(label, answer, replacement, domain)
  return replacement
