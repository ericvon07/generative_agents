"""
File: read_valentine.py
Description: the readout of the Valentine's Day story in the paper's Ville.
             Reads a run's traces, its stage folders (memories, scratch,
             movement) and its labelled interviews; never touches a
             simulation.

    python read_valentine.py <run-prefix> [<run-prefix> ...]
        [--start base_the_ville_n9] [--json OUT] [--stamp] [--lines]

Every run is read on its own and printed in its own block. Two runs are never
paired: their numbers sit side by side, each with its own count.

THE MATCH PATTERNS -- committed before any of the runs, and the only
thing that decides whether a memory carries a fact. Case-insensitive, read on a
node's description, subject, predicate and object together:

  the party       `valentine`, or `party` together with `hobbs` or `isabella`
  the candidacy   `mayor`, `election` or `candida`, together with `sam` (a
                  whole word) or `moore` -- in the text, or as the node's
                  subject, or as the utterance's speaker
  a person        the person's full name, or their first name as a whole word

  Amended 2026-10-03, before any run that counts, when D13 put the paper's
  seed memories into the village: Carmen Ortiz's row seeds "You are somewhat
  curious about who will run for the local mayor election", which the first
  candidacy pattern matched although it says nothing of Sam. A node carries
  the candidacy only if it ties the office to him.

THE LABEL RULE -- applied to every answer, and by hand where the rule cannot
settle one; this reader only reads the labels (`label`: "yes" | "no" |
"unsettled", `labelled_by`: "rule" | "author") and never sets one:

  An answer is "yes" when it asserts the specific fact. For the party: a party
  on February 14th, or at Hobbs Cafe, or given by Isabella. For the candidacy:
  that Sam Moore is running for mayor. For a person: that the persona knows who
  that person is. Anything else is "no". An answer the rule cannot settle (a
  hedge, a right fact attached to the wrong person) is "unsettled" and is
  labelled by hand, quoted, by name; that label overrides.

WHAT SUPPORTS A "YES" -- decided by the patterns above, never by hand:

  * a node of the persona's associative memory at the interviewed checkpoint
    that the fact's pattern (or the person's name) matches; the earliest one
    is cited;
  * failing a node, the persona's identity text as the interview prompt
    carried it (innate, learned, currently, lifestyle, daily plan): the
    paper's seed of who a persona is and knows. Cited as "identity". At the
    base every persona has zero nodes, so every supported "yes" there is an
    identity one.

  A "yes" with neither is HALLUCINATED, and counted apart as the paper does.
  An unsettled answer counts in no measure until it is labelled by hand.

THE DEFINITIONS, with the clock at 10 s a step from 00:00 of
13 February 2023 (step 0):

  knows               "yes" and supported.
  heard before the    the persona's earliest party node at the last checkpoint
  party               was created before 17:00 of 14 February (step 14,760).
                      Isabella excluded.
  showed up           the persona's tile lies in Hobbs Cafe (the maze's sector
                      of the tile in movement/<step>.json, not the action's
                      address, which names a destination) at one or more
                      steps 14,760..15,480. Isabella excluded.
  the midnight        step 8,640; the first stage after it is the stage whose
                      range holds it, read from its saved scratch.

THE READINGS, in order:

  1. The paper's three measures: the party known (k of n), the candidacy known, density at
     the start and at the end with the hallucinated share, showed up of heard
     before the party and of those whose plan for the 14th named it; the
     originators' encounter rows printed beside the reach figures. The paper's
     numbers (25 agents, gpt-3.5-turbo, one run) in their own column.
  2. (a) the diffusion path per fact: each persona's first node carrying it,
     its time, type and the persona it came from (a chat's other speaker; an
     event's subject when that is a persona, else the first persona its
     description names; a thought's earliest matching evidence, else the first
     persona it names); beside it the first utterance carrying the fact that
     reached the persona, from the trace.
  3. (b) the encounter graph: steps each pair shared an arena (world, sector,
     arena of the two tiles), and who never shared one with each originator.
  4. (c) the morning of the 14th, and the midnight's gate: per persona, `currently`
     and the day plan before and after the midnight, whether the new-day path
     ran (a call of plan.revise_identity at the midnight step) and whether any
     prompt of that step fell to its fail-safe; the lines naming either fact.
     The gate stops only on the list in gate_stops; any other
     fail-safe at that step, such as the wake-up hour, is printed as counted.
  5. (d) the decision layer: decisions, fail-safes (gave_up), out-of-domain
     answers and calls per decision, the location family, the social pair and
     the scores on separate lines; the conversations held and with
     whom; decide_to_talk as asks and yeses, per ask and per episode.
  6. (e) operations: wall clock, calls by path, rate-limit waits, cost, each
     provider's share of the generator's calls and every switch with its step.
     The one place a ratio between two runs may be read, as operational.
  7. Secondary: party nodes per persona, inflated by the
     conversation stored up to three times; the paper's measures are
     per persona and do not inherit it.

The start density is read on the run's first checkpoint (its stage 0, step 200,
00:33 of 13 February, every persona asleep, no conversation possible), where
D13 has loaded the paper's seed memories; each arm loads its own, so each has
its own start. --start names a checkpoint printed beside it for reference
(default base_the_ville_n9: the release's base, no seed memory at all). --json writes every figure
to a file for the snapshot. --stamp writes each answer's `support` into the
interview files (the only field this reader writes, and only there).
"""
import argparse
import collections
import datetime
import glob
import json
import os
import re
import sys

from compare_backends import decision_label, gave_up

TRACE_DIR = "../../logs/llm_trace"
STORAGE = "../../environment/frontend_server/storage"
INTERVIEWS = "../../logs/interviews"

ISABELLA = "Isabella Rodriguez"
SAM = "Sam Moore"
CAFE = "Hobbs Cafe"

T0 = datetime.datetime(2023, 2, 13, 0, 0, 0)
SEC_PER_STEP = 10
MIDNIGHT = 8640
PARTY_FROM, PARTY_TO = 14760, 15480
END = 17280

PATTERNS = {
    "party": lambda t: bool(re.search(r"valentine", t)) or (
        bool(re.search(r"party", t)) and bool(re.search(r"hobbs|isabella", t))),
    "candidacy": lambda t: bool(re.search(r"mayor|election|candida", t)) and (
        bool(re.search(r"\bsam\b|moore", t))),
}
ORIGINATOR = {"party": ISABELLA, "candidacy": SAM}

LOCATION = ("action_sector", "action_arena", "action_game_object")
SOCIAL = ("decide_to_talk", "decide_to_react")
SCORES = ("event_poignancy", "thought_poignancy", "chat_poignancy")

PAPER = {"party": "13 of 25 (52%)", "candidacy": "8 of 25 (32%)",
         "density": "0.167 -> 0.74", "hallucinated": "1.3%",
         "showed": "5 of 12"}


def step_time(step):
  return T0 + datetime.timedelta(seconds=SEC_PER_STEP * step)


def when(created):
  return datetime.datetime.strptime(created, "%Y-%m-%d %H:%M:%S")


def person_rx(name):
  first = name.split()[0]
  return re.compile(r"%s|\b%s\b" % (re.escape(name.lower()),
                                    re.escape(first.lower())))


def node_text(n):
  return " ".join(str(n.get(k) or "") for k in
                  ("description", "subject", "predicate", "object")).lower()


def matcher(about):
  """The pattern for a fact, or the name pattern for a person."""
  if about in PATTERNS:
    return PATTERNS[about]
  rx = person_rx(about)
  return lambda t: bool(rx.search(t))


# --- the run on disk --------------------------------------------------------

def stages_of(prefix):
  pat = re.compile(r"^%s-s-(\d+)-(\d+)-(\d+)$" % re.escape(prefix))
  out = []
  for name in os.listdir(STORAGE):
    m = pat.match(name)
    if m:
      out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), name))
  return sorted(out)


def trace_rows(prefix):
  pat = re.compile(r"^%s-s-\d+-\d+-\d+_.*\.jsonl$" % re.escape(prefix))
  files = sorted(f for f in glob.glob(os.path.join(TRACE_DIR, "*.jsonl"))
                 if pat.match(os.path.basename(f)))
  rows = []
  for f in files:
    with open(f, encoding="utf-8") as fh:
      for line in fh:
        line = line.strip()
        if line:
          try:
            rows.append(json.loads(line))
          except ValueError:
            pass  # a run killed mid-write leaves one truncated line
  return files, rows


def personas_of(folder):
  with open(os.path.join(STORAGE, folder, "reverie", "meta.json")) as f:
    return json.load(f)["persona_names"]


def load_nodes(folder, name):
  path = os.path.join(STORAGE, folder, "personas", name, "bootstrap_memory",
                      "associative_memory", "nodes.json")
  try:
    with open(path) as f:
      return json.load(f)
  except IOError:
    return {}


def load_scratch(folder, name):
  path = os.path.join(STORAGE, folder, "personas", name, "bootstrap_memory",
                      "scratch.json")
  with open(path) as f:
    return json.load(f)


def load_interview(target, mode):
  path = os.path.join(INTERVIEWS, "%s-%s.json" % (target, mode))
  if not os.path.exists(path):
    return None, path
  with open(path) as f:
    return json.load(f), path


# --- support and labels -----------------------------------------------------

def support_for(nodes, identity, about):
  """The earliest node the fact's pattern matches, else the identity text."""
  match = matcher(about)
  hits = sorted(((n["created"], nid) for nid, n in nodes.items()
                 if match(node_text(n))), key=lambda x: (x[0], x[1]))
  if hits:
    return {"kind": "node", "node": hits[0][1], "created": hits[0][0],
            "nodes": len(hits)}
  if identity and match(identity.lower()):
    return {"kind": "identity", "node": None, "created": None, "nodes": 0}
  return None


def read_interview(interview, path, stamp=False):
  """Each answer with its label and its support at the interviewed checkpoint.
  Returns {persona: {about: (label, support, answer)}} and the counts."""
  out = {}
  counts = collections.Counter()
  for p in interview["personas"]:
    nodes = load_nodes(interview["checkpoint"], p["persona"])
    for a in p["answers"]:
      sup = support_for(nodes, p.get("identity"), a["about"])
      if stamp:
        a["support"] = sup
      label = a.get("label")
      counts["answers"] += 1
      if label is None:
        counts["unlabelled"] += 1
      elif label == "unsettled":
        counts["unsettled"] += 1
      elif label == "yes":
        counts["yes"] += 1
        counts["knows" if sup else "hallucinated"] += 1
      else:
        counts["no"] += 1
      if a.get("labelled_by") == "author":
        counts["by_author"] += 1
      if a.get("fail_safe"):
        counts["fail_safe"] += 1
      out.setdefault(p["persona"], {})[a["about"]] = (label, sup, a["answer"])
  if stamp:
    with open(path, "w") as f:
      json.dump(interview, f, indent=1, ensure_ascii=False)
  return out, counts


def knows(entry):
  label, sup, _ = entry
  return label == "yes" and sup is not None


def density(answers, names):
  edges = []
  for i, a in enumerate(names):
    for b in names[i + 1:]:
      if knows(answers.get(a, {}).get(b, (None, None, ""))) and \
         knows(answers.get(b, {}).get(a, (None, None, ""))):
        edges.append((a, b))
  n = len(names)
  return (2.0 * len(edges) / (n * (n - 1)) if n > 1 else 0.0), edges


# --- the encounter graph ----------------------------------------------------

_maze = None


def maze():
  global _maze
  if _maze is None:
    from maze import Maze
    _maze = Maze("the_ville")
  return _maze


def movement(folder, step):
  path = os.path.join(STORAGE, folder, "movement", "%d.json" % step)
  try:
    with open(path) as f:
      return json.load(f).get("persona") or {}
  except IOError:
    return None


def where(tile):
  t = maze().access_tile(tuple(tile))
  return t.get("world"), t.get("sector"), t.get("arena")


def walk(folder, last_step, names):
  """Per step, each persona's (world, sector, arena) from its tile."""
  shared = collections.Counter()
  cafe = collections.defaultdict(list)
  steps_read = 0
  cache = {}
  for step in range(0, last_step + 1):
    mv = movement(folder, step)
    if mv is None:
      continue
    steps_read += 1
    here = {}
    for name in names:
      tile = (mv.get(name) or {}).get("movement")
      if tile is None:
        continue
      key = tuple(tile)
      if key not in cache:
        cache[key] = where(key)
      here[name] = cache[key]
      if PARTY_FROM <= step <= PARTY_TO and cache[key][1] == CAFE:
        cafe[name].append(step)
    for i, a in enumerate(names):
      for b in names[i + 1:]:
        if a in here and b in here and here[a][2] and here[a] == here[b]:
          shared[(a, b)] += 1
  return shared, cafe, steps_read


def pair(shared, a, b):
  return shared.get((a, b), 0) + shared.get((b, a), 0)


# --- the diffusion path -----------------------------------------------------

def first_persona_named(text, names, me):
  low = text.lower()
  best = None
  for n in names:
    if n == me:
      continue
    m = person_rx(n).search(low)
    if m and (best is None or m.start() < best[0]):
      best = (m.start(), n)
  return best[1] if best else None


def came_from(nid, nodes, names, me, about, depth=0):
  n = nodes[nid]
  if n["type"] == "chat":
    for who in (n.get("object"), n.get("subject")):
      if who in names and who != me:
        return who
  if n["type"] == "event" and n.get("subject") in names and n["subject"] != me:
    return n["subject"]
  if n["type"] == "thought" and depth < 8:
    match = matcher(about)
    ev = sorted((nodes[e]["created"], e) for e in (n.get("filling") or [])
                if isinstance(e, str) and e in nodes
                and match(node_text(nodes[e])))
    if ev:
      return came_from(ev[0][1], nodes, names, me, about, depth + 1)
  return first_persona_named(n.get("description") or "", names, me)


def path_of(folder, names, about):
  match = matcher(about)
  out = {}
  for name in names:
    nodes = load_nodes(folder, name)
    hits = sorted((n["created"], nid) for nid, n in nodes.items()
                  if match(node_text(n)))
    if not hits:
      continue
    nid = hits[0][1]
    n = nodes[nid]
    out[name] = {"node": nid, "created": n["created"], "type": n["type"],
                 "from": came_from(nid, nodes, names, name, about),
                 "description": n.get("description"), "nodes": len(hits)}
  return out


def heard_by_utterance(rows, about):
  match = matcher(about)
  first = {}
  for r in rows:
    said = ("%s %s" % (r.get("speaker") or "", r.get("text") or "")).lower()
    if r.get("event") == "utterance" and match(said):
      who = r.get("listener")
      if who and who not in first:
        first[who] = {"step": r.get("step"), "sim_time": r.get("sim_time"),
                      "speaker": r.get("speaker"), "text": r.get("text")}
  return first


# --- the midnight -----------------------------------------------------------

def midnight(prefix, stages, rows, names):
  after = [s for s in stages if s[1] <= MIDNIGHT < s[2]]
  if not after:
    return None
  _, frm, to, folder = after[0]
  before = [s for s in stages if s[2] == frm]
  at = [r for r in rows if r.get("step") == MIDNIGHT]
  read = {}
  for name in names:
    s = load_scratch(folder, name)
    prev = load_scratch(before[0][3], name) if before else None
    mine = [r for r in at if r.get("persona") == name]
    revised = any(r.get("event") == "call" and
                  str(r.get("label") or "").endswith("revise_identity")
                  for r in mine)
    failsafes = sorted(set(r.get("label") for r in mine
                           if r.get("event") == "outcome" and r.get("failsafe")))
    errored = sorted(set(r.get("label") for r in mine
                         if r.get("event") == "call" and r.get("ok") is False))
    plan = s.get("daily_plan_req") or ""
    hourly = "; ".join("%s (%d min)" % (a, d) for a, d in
                       (s.get("f_daily_schedule_hourly_org") or []))
    names_it = {}
    for about in PATTERNS:
      lines = [("currently", s.get("currently") or ""), ("plan", plan),
               ("hourly", hourly)]
      names_it[about] = [k for k, v in lines if PATTERNS[about](v.lower())]
    read[name] = {"currently": s.get("currently"), "plan": plan,
                  "hourly": hourly, "curr_time": s.get("curr_time"),
                  "currently_before": prev.get("currently") if prev else None,
                  "plan_before": prev.get("daily_plan_req") if prev else None,
                  "changed": bool(prev) and (
                      prev.get("currently") != s.get("currently")),
                  "revise_identity_called": revised,
                  "failsafes": failsafes, "errored_calls": errored,
                  "names": names_it}
  for v in read.values():
    v["stops"] = gate_stops(v)
  gate = not any(v["stops"] for v in read.values())
  return {"stage": folder, "before": before[0][3] if before else None,
          "gate": gate, "personas": read}


def gate_stops(v):
  """The midnight's gate as amended on 2026-10-03: the gate stops only on what the story stands
  on -- the identity revision did not run or errored, `currently` was not
  revised, there is no day plan, or the hourly schedule fell to its fail-safe.
  Any other fail-safe or errored call at the midnight step (the wake-up hour's
  5-token budget, an event triple) is counted, not a stop."""
  why = []
  if not v["revise_identity_called"]:
    why.append("no revise_identity call at step %d" % MIDNIGHT)
  if any(l.endswith("revise_identity") for l in v["errored_calls"]):
    why.append("revise_identity errored")
  if not v["changed"]:
    why.append("currently unchanged")
  if not v["plan"]:
    why.append("no day plan")
  if any(l.endswith("generate_hourly_schedule") for l in v["failsafes"]):
    why.append("hourly schedule fell to its fail-safe")
  return why


# --- the decision layer and operations --------------------------------------

def family_line(decisions, calls, labels):
  group = [r for r in decisions if r.get("label") in labels]
  checked = [r for r in group if r.get("in_domain") is not None]
  served = [r for r in calls if decision_label(r) in labels]
  return {"decisions": len(group),
          "gave_up": sum(1 for r in group if gave_up(r)),
          "out_of_domain": sum(1 for r in checked if r["in_domain"] is False),
          "checked": len(checked),
          "calls": len(served),
          "calls_per_decision": (float(len(served)) / len(group)
                                 if group else None)}


def conversations(rows):
  convos, cur = [], None
  for r in rows:
    if r.get("event") != "utterance":
      continue
    pair_ = frozenset([r["speaker"], r["listener"]])
    if cur is None or cur["pair"] != pair_ or r["step"] - cur["last"] > 1:
      cur = {"pair": pair_, "step": r["step"], "last": r["step"],
             "sim_time": r.get("sim_time"), "lines": 0}
      convos.append(cur)
    cur["lines"] += 1
    cur["last"] = r["step"]
  return convos


def talk(decisions):
  asks = [r for r in decisions if r.get("label") == "decide_to_talk"]
  eps = collections.defaultdict(list)
  for r in asks:
    eps[r.get("episode")].append(r)
  yes = [r for r in asks if str(r.get("answer")).lower() == "yes"]
  return {"asks": len(asks), "yes": len(yes),
          "gave_up": sum(1 for r in asks if gave_up(r)),
          "episodes": len(eps),
          "episodes_with_yes": sum(1 for v in eps.values()
                                   if any(str(r.get("answer")).lower() == "yes"
                                          for r in v))}


def operations(files, rows):
  calls = [r for r in rows if r.get("event") == "call"]
  by_path = collections.Counter(r.get("path") for r in calls)
  gen = [r for r in calls if r.get("path") in ("GPT_request", "ChatGPT_request")]
  providers = collections.Counter(r.get("provider") or "?" for r in gen)
  waits = [r for r in rows if r.get("event") == "rate_limit"]
  switches = [{"step": r.get("step"), "from": r.get("from"), "to": r.get("to"),
               "reason": r.get("reason")}
              for r in rows if r.get("event") == "provider_switch"]
  pins = []
  for r in rows:
    if r.get("event") == "manifest":
      pin = ((r.get("model") or {}).get("provider_pin") or {}).get("order")
      key = (json.dumps(pin), r.get("config_sha1"))
      if not pins or pins[-1][1:] != key:
        pins.append((r.get("sim"),) + key)
  cost = 0.0
  for r in calls:
    c = r.get("cost")
    if isinstance(c, (int, float)):
      cost += c
    elif isinstance(c, dict):
      cost += sum(v for v in c.values() if isinstance(v, (int, float)))
  jev_tok = sum(r.get("in_tok") or 0 for r in calls if r.get("path") == "jev")
  wall = 0.0
  for f in files:
    ts = []
    with open(f, encoding="utf-8") as fh:
      for line in fh:
        try:
          t = json.loads(line).get("ts")
        except ValueError:
          continue
        if t:
          ts.append(t)
    if ts:
      a = datetime.datetime.strptime(min(ts), "%Y-%m-%dT%H:%M:%S.%f")
      b = datetime.datetime.strptime(max(ts), "%Y-%m-%dT%H:%M:%S.%f")
      wall += (b - a).total_seconds()
  seed = [r for r in rows if r.get("phase") == "seed"]
  seed_ts = sorted(r["ts"] for r in seed if r.get("ts"))
  seed_load = {
      "calls": sum(1 for r in seed if r.get("event") == "call"),
      "decisions": sum(1 for r in seed if r.get("event") == "decision"),
      "minutes": ((datetime.datetime.strptime(seed_ts[-1], "%Y-%m-%dT%H:%M:%S.%f")
                   - datetime.datetime.strptime(seed_ts[0], "%Y-%m-%dT%H:%M:%S.%f"))
                  .total_seconds() / 60.0) if seed_ts else 0.0,
      "record": next((r for r in rows if r.get("event") == "seed_history"), None)}
  return {"wall_clock_h": wall / 3600.0, "calls": dict(by_path),
          "seed_load": seed_load,
          "generator_calls": len(gen), "providers": dict(providers),
          "rate_limit_waits": len(waits),
          "rate_limit_wait_s": sum(r.get("delay_s") or 0 for r in waits),
          "switches": switches,
          "pins": [{"from_stage": p[0], "order": json.loads(p[1]),
                    "config_sha1": p[2]} for p in pins],
          "router_cost_usd": cost,
          "jev_cost_usd": jev_tok * 0.042 / 1e6}


# --- the reading ------------------------------------------------------------

def frac(k, n):
  return "%d of %d" % (k, n)


def read(prefix, start, show_lines, stamp):
  stages = stages_of(prefix)
  if not stages:
    raise SystemExit("no stage folder of %s under %s" % (prefix, STORAGE))
  last = stages[-1][3]
  last_step = stages[-1][2]
  names = personas_of(last)
  files, rows = trace_rows(prefix)
  head = next((r for r in rows if r.get("event") == "manifest"), {})
  out = {"run": prefix, "last_stage": last, "last_step": last_step,
         "personas": names, "stages": len(stages), "trace_files": len(files)}

  print("=" * 78)
  print("%s  world %s  backend %s  seed %s  tree %s  dirty %s  config %s"
        % (prefix, head.get("world", "W1"),
           (head.get("backend") or {}).get("decision"), head.get("seed"),
           ((head.get("code") or {}).get("tree") or "")[:8],
           (head.get("code") or {}).get("dirty"), head.get("config_sha1")))
  print("  stages %d (%s .. %s), last step %d (%s), %d personas, %d trace files"
        % (len(stages), stages[0][3], last, last_step, step_time(last_step),
           len(names), len(files)))
  if last_step < END:
    print("  !! the run stopped at step %d, before %d: every figure below is "
          "over what ran" % (last_step, END))

  # the encounter graph first: every reach figure is printed beside it
  shared, cafe, steps_read = walk(last, last_step, names)
  never = {f: [n for n in names if n != o and pair(shared, o, n) == 0]
           for f, o in ORIGINATOR.items()}

  # 1. the paper's three measures
  print("\n1. THE PAPER'S THREE MEASURES   [paper: 25 agents, "
        "gpt-3.5-turbo, one run]")
  facts, fpath = load_interview(prefix, "facts")
  rel_end, rpath = load_interview(prefix, "relations")
  first = stages[0][3]
  rel_first, fpath1 = load_interview(first, "relations")
  rel_base, bpath = load_interview(start, "relations") if start else (None, None)
  measures = {}
  if facts is None:
    print("   the facts interview is not on disk (%s)" % fpath)
  else:
    fa, fc = read_interview(facts, fpath, stamp)
    for about in ("party", "candidacy"):
      knowers = [n for n in names if knows(fa.get(n, {}).get(about, (None, None, "")))]
      halluc = [n for n in names
                if (fa.get(n, {}).get(about) or (None,))[0] == "yes"
                and not knows(fa[n][about])]
      measures[about] = {"knows": knowers, "hallucinated": halluc}
      print("   %-10s known by %-9s (hallucinated yes: %d)   paper: %s"
            % (about, frac(len(knowers), len(names)), len(halluc), PAPER[about]))
      print("              beside it: %s never shared an arena with %s: %s"
            % (len(never[about]), ORIGINATOR[about],
               ", ".join(never[about]) or "none"))
    print("   facts interview: %d answers, %d unlabelled, %d unsettled, %d set "
          "by the author, %d fail-safe"
          % (fc["answers"], fc["unlabelled"], fc["unsettled"], fc["by_author"],
             fc["fail_safe"]))
  for tag, iv, path in (("base", rel_base, bpath), ("start", rel_first, fpath1),
                        ("end", rel_end, rpath)):
    if iv is None:
      print("   density at the %s: the relations interview is not on disk (%s)"
            % (tag, path))
      continue
    ra, rc = read_interview(iv, path, stamp)
    eta, edges = density(ra, names)
    labelled = rc["yes"] + rc["no"]
    measures["density_" + tag] = {"eta": eta, "edges": len(edges),
                                  "hallucinated": rc["hallucinated"],
                                  "answers": rc["answers"],
                                  "unlabelled": rc["unlabelled"],
                                  "unsettled": rc["unsettled"],
                                  "by_author": rc["by_author"]}
    print("   density at the %-5s eta %.3f (%d of %d pairs)  hallucinated %d of "
          "%d answers (%s)  unlabelled %d, unsettled %d, by the author %d   "
          "paper: %s, %s"
          % (tag, eta, len(edges), len(names) * (len(names) - 1) // 2,
             rc["hallucinated"], rc["answers"],
             "%.1f%%" % (100.0 * rc["hallucinated"] / rc["answers"])
             if rc["answers"] else "-", rc["unlabelled"], rc["unsettled"],
             rc["by_author"], PAPER["density"], PAPER["hallucinated"]))

  party_path = path_of(last, names, "party")
  heard = sorted(n for n, p in party_path.items() if n != ISABELLA
                 and when(p["created"]) < step_time(PARTY_FROM))
  mid = midnight(prefix, stages, rows, names)
  planned = sorted(n for n in heard if mid and mid["personas"][n]["names"]["party"])
  showed = sorted(n for n in heard if cafe.get(n))
  measures["coordination"] = {"heard_before": heard, "showed_up": showed,
                              "plan_named_party": planned,
                              "at_cafe_not_heard": sorted(
                                  n for n in cafe if n not in heard
                                  and n != ISABELLA)}
  if last_step < PARTY_TO:
    print("   coordination: the run did not reach the party's end (step %d)"
          % PARTY_TO)
  print("   showed up %s of heard before the party   paper: %s"
        % (frac(len(showed), len(heard)), PAPER["showed"]))
  print("      heard before: %s" % (", ".join(heard) or "none"))
  print("      of them, plan for the 14th named the party: %s"
        % (", ".join(planned) or "none"))
  print("      at the cafe 17-19h without having heard: %s"
        % (", ".join(measures["coordination"]["at_cafe_not_heard"]) or "none"))
  out["measures"] = measures

  # 2. (a) the diffusion path
  print("\n2. THE DIFFUSION PATH: first node carrying the fact, and "
        "the first utterance that brought it")
  out["diffusion"] = {}
  for about in ("party", "candidacy"):
    p = party_path if about == "party" else path_of(last, names, about)
    u = heard_by_utterance(rows, about)
    out["diffusion"][about] = {"nodes": p, "utterances": u}
    print("   %s (originator %s)" % (about, ORIGINATOR[about]))
    for n in sorted(p, key=lambda x: p[x]["created"]):
      x = p[n]
      line = "      %-19s %s %-7s from %-19s (%d nodes)" % (
          n, x["created"][5:16], x["type"], x["from"] or "-", x["nodes"])
      if n in u:
        line += "  | told by %s at %s" % (u[n]["speaker"], u[n]["sim_time"])
      print(line)
      if show_lines:
        print("         %s" % (x["description"] or "")[:200])
    missing = [n for n in names if n not in p]
    print("      no node: %s" % (", ".join(missing) or "none"))

  # 3. (b) the encounter graph
  print("\n3. THE ENCOUNTER GRAPH: steps each pair shared an arena, "
        "over %d steps read" % steps_read)
  short = [n.split()[0][:7] for n in names]
  print("      %-19s %s" % ("", " ".join("%7s" % s for s in short)))
  for a in names:
    print("      %-19s %s" % (a, " ".join(
        "%7s" % ("-" if a == b else pair(shared, a, b)) for b in names)))
  for about, o in ORIGINATOR.items():
    print("      never shared an arena with %s: %s"
          % (o, ", ".join(never[about]) or "none"))
  out["encounters"] = {"%s|%s" % k: v for k, v in shared.items()}
  out["never_met"] = never

  # 4. (c) the morning of the 14th and the midnight's gate
  print("\n4. THE MORNING OF THE 14TH AND THE MIDNIGHT'S GATE")
  out["midnight"] = mid
  if mid is None:
    print("   the run did not cross step %d" % MIDNIGHT)
  else:
    print("   stage %s (scratch saved at its end), against %s"
          % (mid["stage"], mid["before"]))
    print("   GATE: %s" % ("PASSED" if mid["gate"] else "FAILED"))
    for n in names:
      v = mid["personas"][n]
      flags = ["STOPS: %s" % s for s in v["stops"]]
      if v["failsafes"]:
        flags.append("counted fail-safe: %s" % ", ".join(v["failsafes"]))
      if v["errored_calls"]:
        flags.append("counted errored: %s" % ", ".join(v["errored_calls"]))
      named = ["%s in %s" % (k, "/".join(w)) for k, w in v["names"].items() if w]
      print("   %-19s %s%s" % (n, "; ".join(flags) or "ok",
                               ("  NAMES " + ", ".join(named)) if named else ""))
      if show_lines or named:
        print("      currently: %s" % v["currently"])
        print("      plan:      %s" % v["plan"])

  # 5. (d) the decision layer
  decisions = [r for r in rows if r.get("event") == "decision"]
  calls = [r for r in rows if r.get("event") == "call"]
  print("\n5. THE DECISION LAYER")
  out["decisions"] = {}
  for fam, labels in (("location", LOCATION), ("social", SOCIAL),
                      ("scores", SCORES)):
    for label in labels:
      f = family_line(decisions, calls, (label,))
      out["decisions"][label] = f
      if not f["decisions"]:
        continue
      print("   %-8s %-18s %6d decisions  gave up %5d  out of domain %4d of "
            "%-5d calls/decision %s"
            % (fam, label, f["decisions"], f["gave_up"], f["out_of_domain"],
               f["checked"], "%.2f" % f["calls_per_decision"]
               if f["calls_per_decision"] is not None else "-"))
    f = family_line(decisions, calls, labels)
    out["decisions"][fam] = f
    print("   %-8s %-18s %6d decisions  gave up %5d  out of domain %4d of %d"
          % (fam, "(family)", f["decisions"], f["gave_up"],
             f["out_of_domain"], f["checked"]))
  t = talk(decisions)
  out["decide_to_talk"] = t
  print("   decide_to_talk: per ask %d yes of %d asks (%d fail-safe); per "
        "episode %d with a yes of %d episodes"
        % (t["yes"], t["asks"], t["gave_up"], t["episodes_with_yes"],
           t["episodes"]))
  convos = conversations(rows)
  pairs = collections.Counter(" & ".join(sorted(c["pair"])) for c in convos)
  out["conversations"] = {"count": len(convos), "pairs": dict(pairs)}
  print("   conversations held: %d, %d utterances, between:" % (
      len(convos), sum(c["lines"] for c in convos)))
  for p, k in pairs.most_common():
    print("      %3d  %s" % (k, p))

  # 6. (e) operations
  ops = operations(files, rows)
  out["operations"] = ops
  print("\n6. OPERATIONS")
  print("   wall clock in stages %.1f h; calls %s" % (
      ops["wall_clock_h"], ", ".join("%s %d" % kv for kv in
                                     sorted(ops["calls"].items()))))
  print("   generator calls %d by provider: %s" % (
      ops["generator_calls"], ", ".join(
          "%s %d (%.1f%%)" % (k, v, 100.0 * v / ops["generator_calls"])
          for k, v in sorted(ops["providers"].items(), key=lambda x: -x[1]))))
  print("   rate-limit waits %d, %.0f s waited; cost: router $%.4f, typed "
        "backend $%.4f" % (ops["rate_limit_waits"], ops["rate_limit_wait_s"],
                           ops["router_cost_usd"], ops["jev_cost_usd"]))
  sl = ops["seed_load"]
  if sl["record"]:
    print("   seed memories (D13): %d phrases loaded in %.1f min, %d calls and "
          "%d decisions, all counted above" % (sl["record"].get("phrases"),
                                               sl["minutes"], sl["calls"],
                                               sl["decisions"]))
  else:
    print("   seed memories (D13): none loaded in this run")
  for p in ops["pins"]:
    print("   pin from %s: %s  config %s" % (p["from_stage"], p["order"],
                                            p["config_sha1"]))
  for s in ops["switches"]:
    print("   provider switch at step %s: %s -> %s (%s)" % (
        s["step"], s["from"], s["to"], s["reason"]))

  # 7. secondary: the party node count
  print("\n7. SECONDARY: party nodes per persona at step %d ("
        "one conversation can be stored up to three times)" % last_step)
  out["party_nodes"] = {n: (party_path.get(n) or {}).get("nodes", 0)
                        for n in names}
  print("   " + ", ".join("%s %d" % (n.split()[0], out["party_nodes"][n])
                          for n in names))
  return out


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("prefixes", nargs="+")
  ap.add_argument("--start", default="base_the_ville_n9",
                  help="the checkpoint whose relations interview is the start")
  ap.add_argument("--json", help="write every figure to this file")
  ap.add_argument("--stamp", action="store_true",
                  help="write each answer's support into the interview files")
  ap.add_argument("--lines", action="store_true")
  args = ap.parse_args(argv)
  result = [read(p, args.start, args.lines, args.stamp) for p in args.prefixes]
  if args.json:
    with open(args.json, "w") as f:
      json.dump(result, f, indent=1, ensure_ascii=False, default=str)
    print("\nwrote", args.json)
  return 0


if __name__ == "__main__":
  sys.exit(main())
