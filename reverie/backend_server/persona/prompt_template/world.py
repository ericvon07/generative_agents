"""
File: world.py
Description: the lab's worlds (W2 and after) -- what a run carries that no
             prompt of the paper carries (010-IP LK-B, LK-H) -- and, since
             015-IP step 3, the mechanism of W3: an object whose state the
             world holds, a discovery, a route, an opportunity, a take.

A world is a section of openai_config.json:

  "world": {
      "name": "W3",
      "maze": "the_ville_w3",
      "private": {"Tomas Kessler": "<the secret>", ...},
      "stake":   {"Tomas Kessler": "<the need>"},
      "profile": {"<name>": "<five level descriptions>", ...},
      "objects": {                                        (015-IP RP-C, RP-F)
          "shortwave radio": {
              "address": "the Ville:The Rose and Crown Pub:pub:shortwave radio",
              "owner": "Arthur Burton",
              "taker": "Tomas Kessler",
              "description": "an old shortwave radio on the back wall ...",
              "take": {"description": "taking the old shortwave radio ...",
                       "minutes": 10}
          }
      },
      "act-on": "pick",                              (RP-D's amendment)
      "discovery": "perception"                      (RP-D; "given" in twin D)
  }

WHY HERE AND NOWHERE ELSE (LK-B). The secret is not in scratch.json
(Scratch.save would drop an unknown key at the first stage fork), not in any
memory node (retrieval would carry it into the dialogue prompt), and not in
the identity stable set (it is pasted into some forty templates). It reaches
a prompt through one door: the injection of LK-D, and only after the reveal
decision said so. The stake travels through its own door since 015-IP RP-E:
the `ask` route's intention, one line, at the next dialogue with the owner.
Because the whole config is fingerprinted, a run's manifest says which
secret, which stake, which profile and which object it carried.

WITH NO SECTION (LK-H) every function here answers "nothing": enabled() is
False, private_text() and stake() are None, the reveal decision is never
asked, no route, no take, no state file, nothing is injected, and the fork
is the paper's. That is what test_world.py holds (015-IP S2.3).

THE RUNTIME STATE (RP-C). What the world must remember across steps and
across the runner's stage forks -- the object's state, the discovery step,
the routes asked and whether one completed, the pending `ask` intention, the
open opportunity episode -- lives in `world_state.json` inside the
simulation folder beside reverie/meta.json: written at every save the
runner makes, copied by the stage fork with everything else (reverie.py's
copyanything copies the whole folder), read at boot; absent, the world
starts fresh. Nothing of it goes into a persona's files.

THE OBJECT'S STATE ON THE TILES (RP-C). The paper resets every object event
to idle at the start of each cycle (reverie.py, game_obj_cleanup); after
that cleanup write_back() puts `shortwave radio is missing` on the radio's
tiles from the take on, so whoever is in the pub's arena perceives it by
the paper's own path. The frontend never sees it.

THE ROUTE (RP-D). decide_route is asked of the taker the first step the
object is in his spatial memory (the discovery), and at 05:00, 12:00 and
16:00 while no route has completed -- a day-part whose hour passed before
the discovery never fires; under "discovery": "given" the taker is treated
as discovered from midnight, so the first ask is at 05:00. At most four a
day. `ask` marks an intention; the need line joins the taker's next dialogue
with the owner on every round (RP-E), and the route completes when that
dialogue happens. `take` and `give_up` insert nothing; a take at an
opportunity completes the route.

THE OPPORTUNITY (RP-F). Each step of the taker's the world reads the tiles:
an episode opens the first step the taker is in the object's arena and the
owner is not, and closes the first step that breaks. decide_to_take is
asked once per episode, at its first step the taker is free (not chatting,
not waiting), while the last route said `take` or the object came up in a
dialogue with the owner after the last route was asked (the refusal). On
`take` the call site splices the fixed action through the paper's own
reaction path; the object's state becomes `missing` when that action ends.

THE STATES ARE BUILT FROM THE TAKER'S OWN MEMORY, not from the world's
knowledge: `known` from his spatial memory and the chat nodes with the
owner (the owner's own lines about the object, quoted, never judged);
`present` from the tiles, each person with what the taker's planted memory
says of them; `relationship` the planted pair sentence; `other.learned`
without the sentences that name the object (the grids' construction rule,
015-IP step 2: the object's facts travel in `known` alone). The same rule
is applied to the reveal state's `other.learned` since step 3.
"""
import datetime
import json
import os
import re

from persona.prompt_template import llm_trace

_CONFIG_PATH = "../../openai_config.json"
STATE_FILE = "world_state.json"

# RP-D: the day-parts at which the route is re-asked while none completed.
DAY_PARTS = ("05:00", "12:00", "16:00")
MAX_ROUTES_A_DAY = 4
ACT_RULES = ("pick", "sample")
DISCOVERY_MODES = ("perception", "given")
IN_PLACE = "in place"
MISSING = "missing"

_section = None
_loaded = False

# The runtime state (RP-C) and where it is kept.
_state = None
_sim_folder = None
_sec_per_step = 10


def _load():
  global _section, _loaded
  if _loaded:
    return _section
  _loaded = True
  try:
    with open(_CONFIG_PATH, "r") as f:
      _section = json.load(f).get("world")
  except Exception:
    _section = None
  return _section


def configure(section):
  """Set the world directly, as a test or a probe would. None is no world.
  The runtime state starts fresh with it."""
  global _section, _loaded, _state, _sim_folder
  _section = section
  _loaded = True
  _state = None
  _sim_folder = None


def enabled():
  return bool(_load())


def name():
  """The world's name, or None when the run is the paper's (the manifest
  writes W1 for that case; this module does not invent a name)."""
  section = _load()
  return (section or {}).get("name") if section else None


def maze():
  """The matrix folder this world's map lives in, or None for the paper's
  (010-IP LK-O: a renamed cast gets its own copy of the map)."""
  section = _load()
  return (section or {}).get("maze") if section else None


def _per_persona(key, persona_name):
  section = _load()
  if not section:
    return None
  return (section.get(key) or {}).get(persona_name)


def private_text(persona_name):
  """The persona's secret, or None. Only a persona with one is ever asked
  the reveal decision."""
  return _per_persona("private", persona_name)


def stake(persona_name):
  """What the persona needs -- what revealing would buy (W2), the need the
  ask route carries (W3) -- or None."""
  return _per_persona("stake", persona_name)


def profile(persona_name):
  """The persona's signed five-domain profile (LK-N), or None. Carried by
  the derived track's state only; never pasted into a paper prompt."""
  return _per_persona("profile", persona_name)


def keeps_a_secret(persona_name):
  return private_text(persona_name) is not None


# --- the W3 keys of the section (015-IP) --------------------------------------

def act_on():
  """RP-D's amendment: the rule the route and the take act on. `pick` (the
  default) acts on the Choice's answer; `sample` draws one option with the
  Choice's probabilities as weights. Any other value refuses to boot."""
  section = _load()
  if not section:
    return "pick"
  rule = section.get("act-on", "pick")
  if rule not in ACT_RULES:
    raise ValueError("world.act-on must be one of %s, not %r"
                     % ("/".join(ACT_RULES), rule))
  return rule


def discovery_mode():
  """`perception` (the default): the route is first asked the step the
  object enters the taker's spatial memory. `given` (twin D): the stake
  names the object's place, so the taker counts as discovered from
  midnight and the first ask is at the first day-part."""
  section = _load()
  if not section:
    return "perception"
  mode = section.get("discovery", "perception")
  if mode not in DISCOVERY_MODES:
    raise ValueError("world.discovery must be one of %s, not %r"
                     % ("/".join(DISCOVERY_MODES), mode))
  return mode


def objects():
  """{object name: spec} from the section, or {}."""
  section = _load()
  if not section:
    return {}
  return dict(section.get("objects") or {})


def object_of_taker(persona_name):
  """(object name, spec) of the object this persona may take, or None. One
  taker per object; a world with no objects has no takers."""
  for obj_name, spec in objects().items():
    if spec.get("taker") == persona_name:
      return obj_name, spec
  return None


def _noun(obj_name, spec):
  return (spec.get("noun") or obj_name.split()[-1]).lower()


def _arena_of_address(address):
  return address.rsplit(":", 1)[0]


def status():
  """One line for the run's startup, beside decision.check_ready()."""
  if not enabled():
    return "world: none (the paper's Ville)"
  return "world: %s, act-on %s, discovery by %s, %d object(s)" % (
      name(), act_on(), discovery_mode(), len(objects()))


def learned_as_seen(other_name, learned):
  """`other.learned` as the derived track's states carry it: the persona's
  signed `learned` without the sentences that name an object they own.
  The object's facts reach the subject by perception and dialogue and
  travel in `known` (the grids' construction rule, 015-IP step 2); with no
  objects, or for a persona who owns none, the text is returned as it is."""
  if not learned:
    return learned
  for obj_name, spec in objects().items():
    if spec.get("owner") != other_name:
      continue
    noun = _noun(obj_name, spec)
    sentences = re.split(r"(?<=[.!?])\s+", learned.strip())
    learned = " ".join(s for s in sentences if noun not in s.lower())
  return learned


# --- the injections (LK-D; 015-IP RP-E) -----------------------------------------

def injection(persona_name):
  """The one line LK-D appends to PART 1 of the utterance prompt on a
  reveal: 'What {name} knows and has told no one: {secret}'. The secret
  alone since 015-IP RP-E -- the stake has its own door, need_line()."""
  secret = private_text(persona_name)
  if secret is None:
    return None
  return "What %s knows and has told no one: %s" % (persona_name, secret)


def need_line(persona_name):
  """RP-E's line, appended to PART 1 at the taker's next dialogue with the
  owner after the route said `ask`: 'What {name} is looking for: {stake}'.
  None for a persona with no stake."""
  need = stake(persona_name)
  if need is None:
    return None
  return "What %s is looking for: %s" % (persona_name, need)


def inject(part_one, persona_name, reveal, need=False):
  """PART 1 of the utterance prompt, with the need line appended when the
  ask intention is due on this dialogue and the secret's line when the
  reveal decision said reveal -- and exactly as it was otherwise, so a
  withhold with no ask pending (and every persona of a run with no world)
  sends the paper's prompt byte for byte."""
  out = part_one
  if need:
    line = need_line(persona_name)
    if line is not None:
      out = "%s\n%s" % (out, line)
  if reveal:
    line = injection(persona_name)
    if line is not None:
      out = "%s\n%s" % (out, line)
  return out


# --- the runtime state (RP-C) ---------------------------------------------------

def _fresh():
  state = {"objects": {}, "takers": {}}
  for obj_name, spec in objects().items():
    state["objects"][obj_name] = {"state": IN_PLACE, "taken_by": None,
                                  "decided_at": None, "missing_from": None}
    taker = spec.get("taker")
    if taker:
      state["takers"][taker] = {
          "object": obj_name, "discovery": None, "parts_seen": [],
          "routes": [], "route_completed": None, "intention": None,
          "ask_delivered": None, "opportunity": None, "opportunities": 0,
          "takes": []}
  return state


def _ensure():
  global _state
  if _state is None:
    _state = _fresh()
  return _state


def state():
  """The runtime state as held now (a copy); {} with no world."""
  if not enabled():
    return {}
  return json.loads(json.dumps(_ensure()))


def boot(sim_folder, step=None, sec_per_step=10):
  """Read the state file from the simulation folder, or start fresh. With
  no world nothing is read and nothing is kept."""
  global _state, _sim_folder, _sec_per_step
  if not enabled():
    return
  _sim_folder = sim_folder
  _sec_per_step = sec_per_step or 10
  path = os.path.join(sim_folder, STATE_FILE)
  if os.path.exists(path):
    with open(path) as f:
      _state = json.load(f)
  else:
    _state = _fresh()


def save(sim_folder=None):
  """Write the state file beside reverie/meta.json. Nothing with no world."""
  if not enabled():
    return None
  folder = sim_folder or _sim_folder
  if not folder:
    return None
  path = os.path.join(folder, STATE_FILE)
  with open(path, "w") as f:
    f.write(json.dumps(_ensure(), indent=2))
  return path


def _taker(persona_name):
  return _ensure()["takers"].get(persona_name)


def _record(kind, **fields):
  fields["kind"] = kind
  llm_trace.record_world(**fields)


def _now_text(persona):
  return persona.scratch.curr_time.strftime("%B %d, %Y, %I:%M %p")


# --- the object on the tiles (RP-C) --------------------------------------------

def missing_event(address):
  return (address, "is", MISSING, MISSING)


def write_back(maze, step):
  """After the cycle's cleanup of object events: put the state the world
  holds on the object's tiles. An object taken reads `<name> is missing`
  from the step its take action ended; an object in place is left as the
  paper leaves it. Nothing with no world."""
  if not enabled():
    return
  for obj_name, spec in objects().items():
    held = _ensure()["objects"].get(obj_name) or {}
    if held.get("state") != MISSING:
      continue
    since = held.get("missing_from")
    if since is not None and step is not None and step < since:
      continue
    address = spec["address"]
    blank = (address, None, None, None)
    for tile in maze.address_tiles.get(address, ()):
      events = maze.tiles[tile[1]][tile[0]]["events"]
      events.discard(blank)
      events.add(missing_event(address))


def object_state(obj_name):
  return (_ensure()["objects"].get(obj_name) or {}).get("state")


# --- the discovery (RP-D) -------------------------------------------------------

def _in_spatial_memory(persona, address):
  world_, sector, arena, obj = address.split(":")
  tree = getattr(persona.s_mem, "tree", {}) or {}
  return obj in ((tree.get(world_) or {}).get(sector) or {}).get(arena, [])


def note_discovery(persona, step):
  """The first step the object is in the taker's spatial memory. Recorded
  once, as a `world` event; returns True on that step."""
  found = object_of_taker(persona.name)
  if found is None:
    return False
  obj_name, spec = found
  t = _taker(persona.name)
  if t["discovery"] is not None:
    return False
  if not _in_spatial_memory(persona, spec["address"]):
    return False
  t["discovery"] = step
  _record("discovery", subject=persona.name, object=obj_name,
          at=_now_text(persona), tile=list(persona.scratch.curr_tile))
  return True


# --- the route (RP-D) -----------------------------------------------------------

def _clock(part):
  hh, mm = part.split(":")
  return int(hh), int(mm)


def route_due(persona, step):
  """Why the route is asked this step -- "discovery" or a day-part -- or
  None. Marks the day-parts as they pass, so one whose hour went by before
  the discovery never fires later."""
  found = object_of_taker(persona.name)
  if found is None:
    return None
  t = _taker(persona.name)
  now = persona.scratch.curr_time
  discovered = (discovery_mode() == "given"
                or (t["discovery"] is not None and t["discovery"] < step))
  due = None
  if t["discovery"] == step and not any(r["at"] == "discovery"
                                        for r in t["routes"]):
    due = "discovery"
  for part in DAY_PARTS:
    if part in t["parts_seen"]:
      continue
    if (now.hour, now.minute) < _clock(part):
      continue
    t["parts_seen"].append(part)
    if due is None and discovered:
      due = part
  if due is None:
    return None
  if t["route_completed"] or len(t["routes"]) >= MAX_ROUTES_A_DAY:
    return None
  return due


def route_answered(persona, answer, step, at):
  t = _taker(persona.name)
  t["routes"].append({"step": step, "at": at, "answer": answer,
                      "time": persona.scratch.curr_time.strftime(
                          "%Y-%m-%d %H:%M:%S")})
  if answer == "ask":
    t["intention"] = "ask"
  _record("route", subject=persona.name, object=t["object"], at=at,
          answer=answer, ask_number=len(t["routes"]))


def need_due(speaker, listener):
  """RP-E: whether this line of <speaker>'s to <listener> carries the need
  -- an `ask` intention pending and the listener the object's owner."""
  found = object_of_taker(speaker)
  if found is None:
    return False
  obj_name, spec = found
  t = _taker(speaker)
  return t["intention"] == "ask" and listener == spec.get("owner")


def ask_delivered(speaker, listener, step):
  """The dialogue the need entered: the route completes. The intention
  stays until the dialogue ends, so every round of it carries the line."""
  t = _taker(speaker)
  if t is None or t["ask_delivered"] is not None:
    return
  t["ask_delivered"] = {"step": step, "with": listener}
  t["route_completed"] = {"how": "ask", "step": step}
  _record("ask_delivered", subject=speaker, object=t["object"],
          listener=listener)


def conversation_ended(a, b):
  """Called when a dialogue between <a> and <b> ends: an intention that was
  delivered in it is cleared."""
  for who in (a, b):
    t = _taker(who) if enabled() else None
    if t and t["intention"] == "ask" and t["ask_delivered"]:
      t["intention"] = None


# --- the opportunity and the take (RP-F) ----------------------------------------

def _arena(maze, tile):
  try:
    return maze.get_tile_path(tile, "arena")
  except Exception:
    return None


def _chats_with(persona, other):
  """The taker's chat nodes with <other>, latest first."""
  out = []
  for node in getattr(persona.a_mem, "seq_chat", []) or []:
    if node.object == other:
      out.append(node)
  return out


def _came_up(lines, noun):
  return any(noun in (line or "").lower() for _, line in lines)


def _refusal_after_last_route(persona, owner, noun):
  """The object came up in a dialogue with the owner after the last route
  was asked (or with no route asked yet)."""
  t = _taker(persona.name)
  last_route = None
  if t["routes"]:
    last_route = datetime.datetime.strptime(t["routes"][-1]["time"],
                                            "%Y-%m-%d %H:%M:%S")
  for node in _chats_with(persona, owner):
    if not _came_up(node.filling or [], noun):
      continue
    if last_route is None or node.created > last_route:
      return True
  return False


def take_due(persona, maze, personas, step):
  """Reads the tiles, keeps the episode, and says whether decide_to_take is
  asked this step (RP-F)."""
  found = object_of_taker(persona.name)
  if found is None:
    return False
  obj_name, spec = found
  t = _taker(persona.name)
  owner = personas.get(spec.get("owner"))
  if owner is None:
    return False
  arena = _arena_of_address(spec["address"])
  here = _arena(maze, persona.scratch.curr_tile) == arena
  owner_here = _arena(maze, owner.scratch.curr_tile) == arena
  open_ = here and not owner_here
  episode = t["opportunity"]
  if open_ and episode is None:
    t["opportunities"] += 1
    episode = {"since": step, "asked": False, "number": t["opportunities"]}
    t["opportunity"] = episode
    _record("opportunity", subject=persona.name, object=obj_name,
            what="opens", number=episode["number"])
  elif not open_ and episode is not None:
    _record("opportunity", subject=persona.name, object=obj_name,
            what="closes", number=episode["number"], since=episode["since"],
            asked=episode["asked"])
    t["opportunity"] = None
    return False
  if not open_ or episode["asked"]:
    return False
  if object_state(obj_name) != IN_PLACE:
    return False
  if persona.scratch.chatting_with or "<waiting>" in (
      persona.scratch.act_address or ""):
    return False
  last = t["routes"][-1]["answer"] if t["routes"] else None
  if last == "take":
    return True
  return _refusal_after_last_route(persona, owner.name, _noun(obj_name, spec))


def take_action(obj_name, spec=None):
  """The fixed action the call site splices on `take` (RP-F): description,
  address, duration, the event triple the tile carries and the object's
  event while it is being taken."""
  spec = spec or objects()[obj_name]
  take = spec.get("take") or {}
  return {
      "description": take.get("description",
                              "taking the %s" % obj_name),
      "address": spec["address"],
      "minutes": int(take.get("minutes", 10)),
      "event": (spec["taker"], "take", obj_name),
      "obj_event": (spec["address"], "is", "being taken"),
      "obj_description": "being taken by %s" % spec["taker"],
      "pronunciatio": take.get("pronunciatio", "\U0001F4FB"),
  }


def take_answered(persona, answer, step):
  """The episode's one ask is spent. On `take` the object's state becomes
  missing from the step the take action ends, the route completes, and a
  `world` event records the take with its step."""
  found = object_of_taker(persona.name)
  obj_name, spec = found
  t = _taker(persona.name)
  episode = t["opportunity"] or {"since": step, "asked": False, "number": 0}
  episode["asked"] = True
  t["opportunity"] = episode
  if answer != "take":
    return
  act = take_action(obj_name, spec)
  steps = int(round(act["minutes"] * 60.0 / float(_sec_per_step)))
  _ensure()["objects"][obj_name] = {
      "state": MISSING, "taken_by": persona.name, "decided_at": step,
      "missing_from": step + steps}
  t["takes"].append({"step": step, "episode": episode["number"]})
  t["route_completed"] = {"how": "take", "step": step}
  t["intention"] = None
  _record("take", subject=persona.name, object=obj_name,
          episode=episode["number"], at=_now_text(persona),
          missing_from=step + steps, description=act["description"])


# --- the states, from the taker's own memory ------------------------------------

def _planted(persona, other):
  """What the persona's planted memory says of <other>: the `knows` thought
  node (build_w3.py plants one per pair), or None."""
  for node in getattr(persona.a_mem, "seq_thought", []) or []:
    if (node.predicate == "knows" and node.object == other
        and node.subject == persona.name):
      return node.description
  return None


def _owner_lines_about(persona, owner, noun):
  """The owner's lines about the object, from the latest dialogue with him
  in which it came up: every line of his from the first mention on."""
  for node in _chats_with(persona, owner):
    lines = node.filling or []
    if not _came_up(lines, noun):
      continue
    out, started = [], False
    for speaker, line in lines:
      if not started and noun in (line or "").lower():
        started = True
      if started and speaker == owner:
        out.append(line)
    return out
  return []


def known_text(persona, obj_name, spec):
  """`known`: what the taker knows of the object and of its owner, from his
  spatial memory and his chat nodes with the owner. The owner's words are
  quoted; whether they were a refusal is the model's reading."""
  owner = spec["owner"]
  t = _taker(persona.name)
  seen = (t["discovery"] is not None
          or _in_spatial_memory(persona, spec["address"]))
  if not seen:
    return ("%s has not seen the %s himself; what he knows of one is what "
            "`stake` says." % (persona.name, obj_name))
  text = "There is %s. It is %s's." % (spec["description"], owner)
  lines = _owner_lines_about(persona, owner, _noun(obj_name, spec))
  if not lines:
    return text + " %s has not asked him about it." % persona.name
  quoted = " ".join('"%s"' % l for l in lines)
  return text + " %s asked him about it. What %s said of it: %s" % (
      persona.name, owner, quoted)


def present_text(persona, maze, personas, spec):
  """`present`: who is in the object's arena now, each with what they are
  doing and what the taker's planted memory says of them."""
  arena = _arena_of_address(spec["address"])
  room = arena.split(":")[1]
  here = _arena(maze, persona.scratch.curr_tile) == arena
  parts = []
  for other_name, other in personas.items():
    if other_name == persona.name:
      continue
    if _arena(maze, other.scratch.curr_tile) != arena:
      continue
    doing = other.scratch.act_description or "there"
    sentence = "%s is in %s, %s." % (other_name, room, doing)
    memory = _planted(persona, other_name)
    if memory:
      sentence += " " + memory
    else:
      sentence += " %s's memory holds nothing about %s." % (persona.name,
                                                             other_name)
    parts.append(sentence)
  if parts:
    return " ".join(parts)
  if here:
    return "No one else is in %s; %s is alone in the room." % (room,
                                                                persona.name)
  return "No one is in %s." % room


def last_conversation_facts(persona, owner, noun):
  node = next(iter(_chats_with(persona, owner)), None)
  if node is None:
    return "none"
  lines = [[s_, l_] for s_, l_ in (node.filling or [])]
  return {"with": owner,
          "when": "today at %s" % node.created.strftime("%I:%M %p"),
          "lines": [{"speaker": s_, "line": l_} for s_, l_ in lines],
          "radio_came_up": _came_up(lines, noun)}


def route_facts(persona, owner, maze, personas):
  """The keyword arguments of route_state(), from the run."""
  obj_name, spec = object_of_taker(persona.name)
  noun = _noun(obj_name, spec)
  return {
      "secret": private_text(persona.name),
      "stake": stake(persona.name),
      "self_name": persona.name,
      "profile": profile(persona.name),
      "currently": persona.scratch.currently,
      "other_name": owner.name,
      "other_doing": owner.scratch.act_description,
      "other_learned": learned_as_seen(owner.name, owner.scratch.learned),
      "relationship": _planted(persona, owner.name),
      "time": _now_text(persona),
      "known": known_text(persona, obj_name, spec),
      "present": present_text(persona, maze, personas, spec),
      "last_conversation": last_conversation_facts(persona, owner.name, noun),
  }


def object_text(obj_name, spec):
  held = object_state(obj_name) or IN_PLACE
  return "There is %s. It is %s. %s keeps it there." % (
      spec["description"], held, spec["owner"])


def opportunity_text(persona, owner):
  """RP-F: the owner is out; when he is expected back is what the taker's
  memory holds of his schedule -- a stranger knows nothing, so unknown."""
  return ("%s is not in the room; he has left. %s does not know when he "
          "will be back." % (owner.name, persona.name))


def take_facts(persona, owner, maze, personas):
  """The keyword arguments of take_state(), from the run. `other.doing` is
  what the taker can see: the owner is not in the room."""
  obj_name, spec = object_of_taker(persona.name)
  facts = route_facts(persona, owner, maze, personas)
  facts["other_doing"] = "not in the room"
  facts["object_"] = object_text(obj_name, spec)
  facts["opportunity"] = opportunity_text(persona, owner)
  return facts


# --- the states' shapes, one place for the grids and the sim ----------------------

def reveal_state(secret, stake, self_name, profile, currently, other_name,
                 other_doing, other_learned, relationship, conversation, time):
  """The eight-field state of the reveal decision (010-IP LK-C), built in
  one place so the sim (run_gpt_prompt_decide_to_reveal) and the dose grid
  (probe_reveal.py) send the same shape. <conversation> is a list of
  [speaker, line]; the last line not spoken by <self_name> is
  `last_utterance`."""
  last = None
  for speaker, line in reversed(conversation):
    if speaker != self_name:
      last = line
      break
  return {
      "secret": secret,
      "stake": stake,
      "self": {"name": self_name, "profile": profile, "currently": currently},
      "other": {"name": other_name, "doing": other_doing,
                "learned": other_learned},
      "relationship": relationship,
      "conversation": [{"speaker": s_, "line": l_} for s_, l_ in conversation],
      "last_utterance": last,
      "time": time,
      # decision._social_fields reads the subject from here, as on the
      # paper's two social decisions.
      "subject": self_name,
  }


def route_state(secret, stake, self_name, profile, currently, other_name,
                other_doing, other_learned, relationship, time, known,
                present, last_conversation):
  """The nine-field state of the route decision (015-IP RP-D), built in
  one place so the sim and the route grid (probe_route.py) send the same
  shape: the reveal state's fields but the conversation buffer, plus

    known             what the subject knows of the radio and its owner --
                      where it is, whose, and what the owner said of it if
                      asked (from spatial memory and the chat nodes with
                      the owner)
    present           who is in the pub now, with the trade of each as the
                      subject's memory holds it
    last_conversation the last dialogue with the owner and whether the radio
                      came up: {"with", "when", "lines", "radio_came_up"},
                      or the string "none" before one

  <other_learned> is the owner's identity AS THE SUBJECT'S DECISION MAY SEE
  IT: the facts of the radio (that it exists, that the owner will not part
  with it) travel in `known` alone, never here -- otherwise `known` is not
  the lever (the route grid's finding, 015-IP step 2)."""
  return {
      "secret": secret,
      "stake": stake,
      "self": {"name": self_name, "profile": profile, "currently": currently},
      "other": {"name": other_name, "doing": other_doing,
                "learned": other_learned},
      "relationship": relationship,
      "time": time,
      "known": known,
      "present": present,
      "last_conversation": last_conversation,
      "subject": self_name,
  }


def take_state(secret, stake, self_name, profile, currently, other_name,
               other_doing, other_learned, relationship, time, known,
               present, last_conversation, object_, opportunity):
  """The route state plus two fields (015-IP RP-F), built in one place for
  the sim and the take grid (probe_take.py):

    object       the radio: where it is, its state, whose it is
    opportunity  the owner is out; when he is expected back, as the
                 subject's memory holds it

  The refusal, when there was one, is in `known` and in `last_conversation`
  and not in `object`, so the two `known` cells of the grid differ by the
  refusal alone."""
  state = route_state(secret, stake, self_name, profile, currently,
                      other_name, other_doing, other_learned, relationship,
                      time, known, present, last_conversation)
  state["object"] = object_
  state["opportunity"] = opportunity
  return state
