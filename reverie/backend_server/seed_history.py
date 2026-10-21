"""
File: seed_history.py
Description: D13 -- the paper's seed memories, loaded at the start of a run
             (Park et al. 2023 §3.1: "Each semicolon-delimited phrase
             is entered into the agent's initial memory as memories at the
             start of the simulation").

The release carries those phrases in
environment/frontend_server/static_dirs/assets/the_ville/agent_history_init_n25.csv
and loads them only through a command typed at the server's prompt
(`call -- load history the_ville/agent_history_init_n25.csv`, reverie.py).
The automatic runner never types it, so until 2026-10-03 every run of the lab
started with zero memories: agents who, by the paper's own file, are
dormmates, classmates, family and neighbours met as strangers.

With `"seed-history": "the_ville/agent_history_init_n25.csv"` in the config,
the runner calls load() once, in the first stage of a run whose origin is at
step 0, before the first step. load() is the paper's command with three
differences, each the least that makes it run unattended on a village built
by subtraction:

  * rows whose agent is not in the simulation are skipped (the paper's
    command raises KeyError on the first one); the kept rows are the paper's,
    byte for byte, in the file's order;
  * an empty phrase -- Sam Moore's row ends with ";" -- is not a phrase and is
    not whispered (the paper's command would turn "" into a thought);
  * a persona whose clock is still unset (a base sim: scratch.curr_time is
    null until the first step) is given the server's clock for the load and
    handed back null after it, so its first step is still the paper's
    "First day" (persona.move reads a null clock as the first day).

Each phrase then goes through the paper's own load_history_via_whisper: an
inner thought, an event triple, an importance score and an embedding. The
importance score is the event_poignancy decision, so it is answered by the
run's decision backend, and every call is in the run's trace under phase
"seed" -- the load is part of the run, in its time and in its cost.
"""
import os

from global_methods import read_file_to_list
from persona.prompt_template import llm_trace
from utils import maze_assets_loc


def configured(cfg):
  """The history file a config asks for, or None."""
  return (cfg or {}).get("seed-history") or None


def rows_for(path, names):
  """The phrases of the agents in `names`, as [[agent, phrase], ...] in the
  file's order, split as the paper splits them; and the agents skipped."""
  rows = read_file_to_list(path, header=True, strip_trail=True)[1]
  kept, skipped = [], []
  for row in rows:
    agent = row[0].strip()
    if agent not in names:
      skipped.append(agent)
      continue
    for phrase in row[1].split(";"):
      phrase = phrase.strip()
      if phrase:
        kept.append([agent, phrase])
  return kept, skipped


def load(rs, rel_path, whisper=None):
  """Load the history file into the server's personas, at its current step.
  `whisper` is the paper's load_history_via_whisper unless a test passes
  another. Returns what was loaded, and writes it to the trace."""
  if whisper is None:
    from persona.cognitive_modules.converse import load_history_via_whisper
    whisper = load_history_via_whisper
  path = os.path.join(maze_assets_loc, rel_path)
  kept, skipped = rows_for(path, set(rs.personas))
  llm_trace.set_step(rs.step, rs.curr_time)
  unset = [n for n, p in rs.personas.items() if p.scratch.curr_time is None]
  for n in unset:
    rs.personas[n].scratch.curr_time = rs.curr_time
  try:
    with llm_trace.phase_scope("seed"):
      for agent, phrase in kept:
        with llm_trace.persona_scope(agent):
          whisper(rs.personas, [[agent, phrase]])
  finally:
    for n in unset:
      rs.personas[n].scratch.curr_time = None
  per_agent = {}
  for agent, _ in kept:
    per_agent[agent] = per_agent.get(agent, 0) + 1
  summary = {"file": rel_path, "phrases": len(kept), "per_agent": per_agent,
             "skipped_agents": skipped, "clock_lent_to": unset}
  llm_trace.record_seed_history(**summary)
  print("(seed-history): D13 loaded %d phrases for %d agents from %s; "
        "skipped %d rows of agents not in the simulation"
        % (len(kept), len(per_agent), rel_path, len(skipped)), flush=True)
  return summary
