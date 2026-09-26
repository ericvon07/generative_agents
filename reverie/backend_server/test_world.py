"""
File: test_world.py
Description: 010-IP step 5 -- the world module, the reveal decision, the
             injection and the utterance event.

Asks no backend and opens no socket: the typed backend is replaced by a
function that returns fixed answers, and the generator by one that records
the prompt it was handed. Run it on its own, like the fork's other tests,
from reverie/backend_server:

    ../../../.venv/bin/python test_world.py

What it holds (010-IP S2.4):
  - LK-H: with no world section nobody keeps a secret, the reveal decision
    is never asked, and the utterance prompt is the paper's, byte for byte.
  - S2.1: with a world, a persona whose decision said withhold sends a
    prompt with no word of the secret or the stake.
  - S2.3: on reveal, exactly one line is added to PART 1, and the loop
    records the utterance with injected: true.
  - LK-C / S3.1: the reveal decision asks one request with the Choice and
    the four companions, acts on the Choice, and writes the companions on
    the decision event; a backend error is the fail-safe withhold with
    gave_up: true.
  - JF-A: a world on the paper's backend refuses to start.
  - LK-O: the map folder is the paper's with no world section, the copy
    the section names with one, and a folder that does not exist is
    refused before a persona is loaded.
  - 015-IP S1.2: the W3 map copy (the_ville_w3, build_w3.py) is the paper's
    folder with two houses renamed in the special-blocks CSVs and one
    object added -- the radio's row and its two tiles -- and nothing else.
  - 015-IP S5.1: the route and take states (world.route_state,
    world.take_state) carry their named fields; the two grids
    (probe_route.py, probe_take.py) build 90 and 120 states, every one
    whole, no two alike; the questions are one Choice over the declared
    domain and its companions.
  - 015-IP step 3 (S2.3 and the step's own list): with no world section
    no route, no take, no state file, and the utterance prompt the
    paper's even with a need pending; the need line and the secret line
    are two doors (RP-E); the route is asked at the discovery and at the
    day-parts while none completed, at most four, and at 05:00 under
    discovery: given (RP-D); the take is asked once per opportunity
    episode, only after a `take` route or a refusal (RP-F); a take makes
    the object missing from the step the action ends, the write-back
    survives the paper's cleanup and a stage fork (RP-C); the states are
    built from the taker's own memory; act-on: sample draws with the
    probabilities and pick never draws; the utterance event carries need
    and secret; a world react on `take` splices the fixed action.
"""
import datetime
import json
import os
import shutil
import tempfile
import unittest

_TRACE_DIR = tempfile.mkdtemp(prefix="world_test_")
os.environ["LLM_TRACE_DIR"] = _TRACE_DIR
os.environ["LLM_TRACE_ENABLED"] = "1"
os.environ["LLM_TRACE_HEAD"] = "200"
os.environ["DECISION_BACKEND"] = "llm"

from persona.prompt_template import llm_trace
from persona.prompt_template import world
from persona.prompt_template import world_questions
from persona.prompt_template import decision
from persona.prompt_template import jev
from persona.prompt_template import run_gpt_prompt
from persona.cognitive_modules import converse
import utils

SECRET = ("Tomas is secretly building a radio transmitter in the house "
          "garden to reach his brother outside the Ville.")
STAKE = ("The transmitter cannot work without a variable capacitor, which "
         "Priya Raman's grandfather's shortwave radio has.")
W2 = {"name": "W2", "maze": "the_ville_w2",
      "private": {"Tomas Kessler": SECRET},
      "stake": {"Tomas Kessler": STAKE},
      "profile": {"Tomas Kessler": "extraversion: low (quiet)"}}


# --- fakes -------------------------------------------------------------------

class _Scratch(object):
  def __init__(self, name, act, currently="", learned=""):
    self.name = name
    self.act_description = act
    self.currently = currently
    self.learned = learned
    self.curr_tile = (0, 0)
    self.curr_time = datetime.datetime(2023, 2, 13, 6, 40)

  def get_str_iss(self):
    return "Name: %s\nCurrently: %s\n" % (self.name, self.currently)


class _Mem(object):
  seq_chat = []


class _Persona(object):
  def __init__(self, name, act, currently="", learned=""):
    self.name = name
    self.scratch = _Scratch(name, act, currently, learned)
    self.a_mem = _Mem()


class _Maze(object):
  def access_tile(self, tile):
    return {"sector": "Priya Raman and Tomas Kessler's house",
            "arena": "garden"}


TOMAS = _Persona("Tomas Kessler", "working in the garden",
                 "managing Harvey Oak Supply Store", "a shopkeeper")
PRIYA = _Persona("Priya Raman", "drinking coffee in the garden",
                 "preparing lessons", "a retired schoolteacher")


def _utterance_prompt(reveal, need=False):
  """The prompt the utterance builder hands the generator, captured."""
  seen = {}
  real = run_gpt_prompt.ChatGPT_safe_generate_response_OLD

  def fake(prompt, *args, **kwargs):
    seen["prompt"] = prompt
    return {"utterance": "Morning.", "end": False}
  run_gpt_prompt.ChatGPT_safe_generate_response_OLD = fake
  try:
    run_gpt_prompt.run_gpt_generate_iterative_chat_utt(
        _Maze(), TOMAS, PRIYA, {"events": []}, "Tomas saw Priya.", [],
        reveal=reveal, need=need)
  finally:
    run_gpt_prompt.ChatGPT_safe_generate_response_OLD = real
  return seen["prompt"]


# --- the W3 fakes (015-IP step 3) ---------------------------------------------

RADIO = "shortwave radio"
RADIO_ADDRESS = "the Ville:The Rose and Crown Pub:pub:shortwave radio"
PUB_ARENA = "the Ville:The Rose and Crown Pub:pub"
FLAT_ARENA = "the Ville:Arthur Burton's apartment:main room"
SECRET3 = ("Tomas is secretly building a radio transmitter in the house "
           "garden, from parts he takes off the supply store product shelf "
           "without recording them, to reach his brother outside the Ville.")
NEED = ("The transmitter cannot work without a variable capacitor, a part "
        "Harvey Oak Supply Store does not stock and that Tomas has only ever "
        "seen inside old shortwave radios. He does not know who in the Ville "
        "might have one.")
ARTHUR_LEARNED = (
    "Arthur Burton is the owner and bartender of The Rose and Crown Pub, "
    "which he has run for ten years and where he loves to make people feel "
    "welcome. Behind the counter he keeps an old shortwave radio that was "
    "his father's; it has not worked in years and he will not sell it or "
    "lend it to anyone.")
ARTHUR_LEARNED_WITHOUT_THE_RADIO = (
    "Arthur Burton is the owner and bartender of The Rose and Crown Pub, "
    "which he has run for ten years and where he loves to make people feel "
    "welcome.")
PLANTED = {
    "Arthur Burton": ("Arthur Burton is the owner of The Rose and Crown Pub. "
                      "Tomas Kessler knows him by sight and by name, and no "
                      "more."),
    "Sam Moore": ("Sam Moore is a retired navy officer who lives in the "
                  "Ville. Tomas Kessler knows him by sight and by name, and "
                  "no more."),
}
W3 = {
    "name": "W3", "maze": "the_ville_w3",
    "private": {"Tomas Kessler": SECRET3, "Daniel Okafor": "the tip"},
    "stake": {"Tomas Kessler": NEED},
    "profile": {"Tomas Kessler": "extraversion: low (quiet)",
                "Arthur Burton": "extraversion: high"},
    "objects": {RADIO: {
        "address": RADIO_ADDRESS, "owner": "Arthur Burton",
        "taker": "Tomas Kessler",
        "description": ("an old shortwave radio on the back wall behind the "
                        "counter of The Rose and Crown Pub"),
        "take": {"description": ("taking the old shortwave radio from "
                                 "behind the counter"), "minutes": 10}}},
    "act-on": "pick",
}
IN_PUB = (57, 23)
IN_FLAT = (54, 14)
ON_THE_STREET = (57, 30)


class _Node(object):
  def __init__(self, subject, predicate, object_, description,
               created=None, filling=None):
    self.subject = subject
    self.predicate = predicate
    self.object = object_
    self.description = description
    self.created = created
    self.filling = filling


class _W3Mem(object):
  def __init__(self):
    self.seq_thought = []
    self.seq_chat = []


class _W3SpatialMemory(object):
  def __init__(self):
    self.tree = {"the Ville": {"The Rose and Crown Pub": {"pub": [
        "bar customer seating", "behind the bar counter"]}}}


class _W3Persona(_Persona):
  def __init__(self, name, act, tile, currently="", learned=""):
    _Persona.__init__(self, name, act, currently, learned)
    self.scratch.curr_tile = tile
    self.scratch.chatting_with = None
    self.scratch.act_address = "the Ville:somewhere:room:chair"
    self.a_mem = _W3Mem()
    self.s_mem = _W3SpatialMemory()

  def at(self, clock, tile=None):
    hh, mm = clock.split(":")
    self.scratch.curr_time = datetime.datetime(2023, 2, 13, int(hh), int(mm))
    if tile is not None:
      self.scratch.curr_tile = tile
    return self

  def sees_the_radio(self):
    self.s_mem.tree["the Ville"]["The Rose and Crown Pub"]["pub"].append(RADIO)
    return self

  def plant(self, other):
    self.a_mem.seq_thought.append(
        _Node(self.name, "knows", other, PLANTED[other]))
    return self

  def chatted(self, other, lines, clock):
    hh, mm = clock.split(":")
    node = _Node(self.name, "chat with", other, "a chat",
                 created=datetime.datetime(2023, 2, 13, int(hh), int(mm)),
                 filling=[list(l) for l in lines])
    self.a_mem.seq_chat.insert(0, node)
    return self


class _W3Maze(object):
  """The pub's arena, the apartment above it and the street, on tiles; the
  radio's two tiles with the paper's blank object event."""

  def __init__(self):
    self.address_tiles = {RADIO_ADDRESS: {(53, 20), (54, 20)}}
    self.tiles = {20: {53: {"events": {(RADIO_ADDRESS, None, None, None)}},
                       54: {"events": {(RADIO_ADDRESS, None, None, None)}}}}

  def get_tile_path(self, tile, level):
    x, y = tile
    if 53 <= x <= 62 and 20 <= y <= 26:
      return PUB_ARENA
    if 52 <= x <= 63 and 13 <= y <= 19:
      return FLAT_ARENA
    return "the Ville:the street:the street"

  def events_at(self, tile):
    return set(self.tiles[tile[1]][tile[0]]["events"])


def _cast(tomas_tile=IN_PUB, arthur_tile=IN_FLAT, sam_tile=None):
  tomas = _W3Persona("Tomas Kessler", "having breakfast", tomas_tile,
                     "managing Harvey Oak Supply Store", "a shopkeeper")
  arthur = _W3Persona("Arthur Burton", "eating dinner in his apartment",
                      arthur_tile, "running the pub", ARTHUR_LEARNED)
  sam = _W3Persona("Sam Moore", "reading at a table",
                   sam_tile or ON_THE_STREET, "retired", "a navy man")
  tomas.plant("Arthur Burton").plant("Sam Moore")
  tomas.at("16:10")
  return tomas, arthur, sam, {"Tomas Kessler": tomas, "Arthur Burton": arthur,
                              "Sam Moore": sam}


REFUSAL = [["Tomas Kessler", "That old radio back there -- does it work? "
                             "I'd buy it off you."],
           ["Arthur Burton", "That was my father's. I won't sell it or lend "
                             "it to anyone. Sorry."]]

ROUTE_ANSWERS = {
    "decide": {"choice": "ask", "confidence": 0.4,
               "probabilities": {"ask": 0.6, "take": 0.3, "give_up": 0.1}},
    "known_says_refused": {"noul": 0.1},
    "present_says_officer": {"noul": 0.1},
    "would_rather_ask": {"noul": 0.8},
    "stopped_wanting": {"noul": 0.1},
}
TAKE_ANSWERS = {
    "decide": {"choice": "take", "confidence": 0.6,
               "probabilities": {"take": 0.8, "leave": 0.2}},
    "object_not_in_use": {"noul": 0.9},
    "owner_not_back_soon": {"noul": 0.5},
    "known_says_refused": {"noul": 0.9},
    "someone_watching": {"noul": 0.1},
    "breaks_trust": {"noul": 0.7},
}


def _records(event):
  out = []
  for name in sorted(os.listdir(_TRACE_DIR)):
    with open(os.path.join(_TRACE_DIR, name)) as f:
      for line in f:
        r = json.loads(line)
        if r.get("event") == event:
          out.append(r)
  return out


class _TypedBackend(object):
  """decision._backend set to jev, and jev.ask replaced, for one test."""

  def __init__(self, answers=None, error=None):
    self.answers = answers
    self.error = error
    self.asked = []

  def __enter__(self):
    self._backend = decision._backend
    self._ask = jev.ask
    decision._backend = "jev"

    def ask(state, questions, label=None):
      self.asked.append({"state": state, "questions": questions,
                         "label": label})
      if self.error:
        raise jev.JevError(self.error)
      return self.answers
    jev.ask = ask
    return self

  def __exit__(self, *exc):
    decision._backend = self._backend
    jev.ask = self._ask


REVEAL_ANSWERS = {
    "decide": {"choice": "reveal", "confidence": 0.4,
               "probabilities": {"reveal": 0.7, "withhold": 0.3}},
    "other_asks_about_subject": {"noul": 0.9},
    "other_offers_stake": {"noul": 0.8},
    "self_trusts_other": {"noul": 0.6},
    "reveal_serves_goal": {"noul": 0.7},
}


# --- the cases -----------------------------------------------------------------

class NoWorld(unittest.TestCase):
  """LK-H: with no section, the fork is the paper's."""

  def setUp(self):
    world.configure(None)

  def test_nobody_keeps_a_secret(self):
    self.assertFalse(world.enabled())
    self.assertIsNone(world.private_text("Tomas Kessler"))
    self.assertIsNone(world.stake("Tomas Kessler"))
    self.assertIsNone(world.injection("Tomas Kessler"))

  def test_the_utterance_prompt_is_the_papers_even_if_asked_to_reveal(self):
    self.assertEqual(_utterance_prompt(False), _utterance_prompt(True))

  def test_the_reveal_decision_is_never_asked(self):
    real = converse.run_gpt_prompt_decide_to_reveal

    def boom(*a, **k):
      raise AssertionError("asked the reveal decision with no world")
    converse.run_gpt_prompt_decide_to_reveal = boom
    try:
      self.assertFalse(converse._reveal(TOMAS, PRIYA, "housemates", []))
    finally:
      converse.run_gpt_prompt_decide_to_reveal = real

  def test_the_paper_backend_starts(self):
    self.assertIn("llm", decision.check_ready())

  def test_the_map_is_the_papers(self):
    """LK-O, LK-H: no section, the paper's matrix folder."""
    self.assertEqual(utils.env_matrix, utils.matrix_folder())

  def test_no_route_no_take_no_state_file_and_the_papers_prompt(self):
    """015-IP S2.3: the world module does nothing with no section."""
    from persona.cognitive_modules import plan
    self.assertIsNone(world.object_of_taker("Tomas Kessler"))
    self.assertEqual({}, world.state())
    self.assertEqual("pick", world.act_on())
    folder = tempfile.mkdtemp(prefix="world_none_")
    try:
      world.boot(folder, 0)
      self.assertIsNone(world.save(folder))
      self.assertEqual([], os.listdir(folder))
    finally:
      shutil.rmtree(folder, ignore_errors=True)
    maze = _W3Maze()
    before = maze.events_at((53, 20))
    world.write_back(maze, 500)
    self.assertEqual(before, maze.events_at((53, 20)))
    tomas, arthur, sam, personas = _cast()
    tomas.sees_the_radio()
    real = (plan.run_gpt_prompt_decide_route, plan.run_gpt_prompt_decide_to_take)

    def boom(*a, **k):
      raise AssertionError("asked a world decision with no world")
    plan.run_gpt_prompt_decide_route = boom
    plan.run_gpt_prompt_decide_to_take = boom
    try:
      plan._world_react(tomas, maze, personas)
    finally:
      (plan.run_gpt_prompt_decide_route,
       plan.run_gpt_prompt_decide_to_take) = real
    self.assertFalse(converse._need(TOMAS, PRIYA))
    self.assertEqual(_utterance_prompt(False), _utterance_prompt(False, True))
    self.assertEqual(_utterance_prompt(False), _utterance_prompt(True, True))


class W3Map(unittest.TestCase):
  """015-IP S1.2: the_ville_w3 against the paper's folder, file by file."""

  RENAMES = [("Yuriko Yamamoto's house", "Daniel Okafor's house"),
             ("Tamara Taylor and Carmen Ortiz's house", "Tomas Kessler's house"),
             ("Carmen Ortiz's room", "Tomas Kessler's room"),
             ("Tamara Taylor's room", "spare room")]
  RADIO_ROW = "32283, the Ville, <all>, shortwave radio"
  RADIO_TILES = [(53, 20), (54, 20)]

  def setUp(self):
    world.configure({"name": "W3", "maze": "the_ville_w3"})
    self.folder = utils.matrix_folder()
    self.assertEqual("%s/the_ville_w3/matrix" % utils.maze_assets_loc,
                     self.folder)

  def tearDown(self):
    world.configure(None)

  def _pair(self, rel):
    with open(os.path.join(self.folder, rel)) as f:
      ours = f.read()
    with open(os.path.join(utils.env_matrix, rel)) as f:
      papers = f.read()
    return papers, ours

  def test_the_same_files_as_the_paper(self):
    def listing(folder):
      return sorted(os.path.relpath(os.path.join(r, n), folder)
                    for r, _, ns in os.walk(folder) for n in ns
                    if n != ".DS_Store")
    self.assertEqual(listing(utils.env_matrix), listing(self.folder))
    self.assertEqual(*self._pair("maze_meta_info.json"))

  def test_two_houses_renamed_and_the_radio_row_and_nothing_else(self):
    for name in os.listdir(os.path.join(self.folder, "special_blocks")):
      papers, ours = self._pair(os.path.join("special_blocks", name))
      for old, new in self.RENAMES:
        papers = papers.replace(old, new)
      if name == "game_object_blocks.csv":
        papers += "\n" + self.RADIO_ROW
      self.assertEqual(papers, ours, name)

  def test_the_radios_two_tiles_and_no_other_cell(self):
    with open(os.path.join(self.folder, "maze_meta_info.json")) as f:
      width = int(json.load(f)["maze_width"])
    for name in os.listdir(os.path.join(self.folder, "maze")):
      papers, ours = self._pair(os.path.join("maze", name))
      if name != "game_object_maze.csv":
        self.assertEqual(papers, ours, name)
        continue
      a, b = papers.split(", "), ours.split(", ")
      self.assertEqual(len(a), len(b))
      changed = [(i % width, i // width) for i in range(len(a)) if a[i] != b[i]]
      self.assertEqual(self.RADIO_TILES, changed)
      for x, y in self.RADIO_TILES:
        self.assertEqual("0", a[y * width + x])
        self.assertEqual("32283", b[y * width + x])


class W3States(unittest.TestCase):
  """015-IP S5.1: the states of the route and the take, and the two grids."""

  ROUTE_FIELDS = ("secret", "stake", "self", "other", "relationship", "time",
                  "known", "present", "last_conversation")

  def test_the_route_state_has_nine_named_fields_and_the_subject(self):
    state = world.route_state("s", "k", "Tomas Kessler", "p", "c",
                              "Arthur Burton", "d", "l", "r", "t", "kn",
                              "pr", "none")
    for key in self.ROUTE_FIELDS:
      self.assertIn(key, state)
    self.assertEqual(state["subject"], "Tomas Kessler")
    self.assertEqual(state["other"]["learned"], "l")
    self.assertNotIn("object", state)

  def test_the_take_state_adds_the_object_and_the_opportunity(self):
    state = world.take_state("s", "k", "Tomas Kessler", "p", "c",
                             "Arthur Burton", "d", "l", "r", "t", "kn",
                             "pr", "none", "obj", "opp")
    for key in self.ROUTE_FIELDS + ("object", "opportunity"):
      self.assertIn(key, state)
    self.assertEqual(state["object"], "obj")
    self.assertEqual(state["opportunity"], "opp")

  def test_the_questions_are_one_choice_and_its_companions(self):
    route = world_questions.route_questions()
    self.assertEqual(route[world_questions.ROUTE_ACT_ON]["type"], "choice")
    self.assertEqual(sorted(route[world_questions.ROUTE_ACT_ON]["criteria"]),
                     sorted(world_questions.ROUTE_DOMAIN))
    self.assertIn(world_questions.ROUTE_FAIL_SAFE, world_questions.ROUTE_DOMAIN)
    for key in world_questions.ROUTE_COMPANIONS:
      self.assertEqual(route[key]["type"], "noul")
    self.assertEqual(len(route), 1 + len(world_questions.ROUTE_COMPANIONS))
    take = world_questions.take_questions()
    self.assertEqual(sorted(take[world_questions.TAKE_ACT_ON]["criteria"]),
                     sorted(world_questions.TAKE_DOMAIN))
    self.assertIn(world_questions.TAKE_FAIL_SAFE, world_questions.TAKE_DOMAIN)
    self.assertEqual(len(take), 1 + len(world_questions.TAKE_COMPANIONS))

  def test_the_two_grids_are_whole_and_of_the_declared_size(self):
    import probe_route
    import probe_take
    route = probe_route.grid()
    self.assertEqual(len(route), 90)
    self.assertEqual(len(set(c["cell"] for c in route)), 90)
    self.assertTrue(all(probe_route.whole(c["state"]) for c in route))
    take = probe_take.grid()
    self.assertEqual(len(take), 120)
    self.assertEqual(len(set(c["cell"] for c in take)), 120)
    self.assertTrue(all(probe_route.whole(c["state"], probe_take.FIELDS)
                        for c in take))
    # the refusal is in `known` and the conversation at refused, and nowhere
    # in `object`
    for c in take:
      self.assertNotIn("will not", c["state"]["object"])
      self.assertEqual(c["state"]["last_conversation"]["radio_came_up"],
                       c["known"] == "refused")
    # the wholeness rule refuses an emptied field
    broken = dict(route[0]["state"], known="")
    self.assertFalse(probe_route.whole(broken))


class W3Mechanism(unittest.TestCase):
  """015-IP step 3: the world module's mechanism -- RP-C, RP-D, RP-E, RP-F
  and RP-D's amendment -- on fakes, asking no backend."""

  def setUp(self):
    world.configure(json.loads(json.dumps(W3)))
    llm_trace.set_simulation("test-w3-%s" % self.id().rsplit(".", 1)[-1])
    llm_trace.set_step(0, datetime.datetime(2023, 2, 13, 0, 0))

  def tearDown(self):
    world.configure(None)

  def _world_events(self, kind=None):
    sim = "test-w3-%s" % self.id().rsplit(".", 1)[-1]
    out = [r for r in _records("world") if r["sim"] == sim]
    if kind:
      out = [r for r in out if r["kind"] == kind]
    return out

  # --- the construction rule and the two doors ---

  def test_the_owners_learned_loses_the_sentence_that_names_the_radio(self):
    self.assertEqual(ARTHUR_LEARNED_WITHOUT_THE_RADIO,
                     world.learned_as_seen("Arthur Burton", ARTHUR_LEARNED))
    self.assertEqual("a navy man", world.learned_as_seen("Sam Moore",
                                                          "a navy man"))
    self.assertIsNone(world.learned_as_seen("Arthur Burton", None))
    # the reveal state carries it the same way since step 3
    tomas, arthur, sam, personas = _cast()
    with _TypedBackend(REVEAL_ANSWERS) as backend:
      run_gpt_prompt.run_gpt_prompt_decide_to_reveal(tomas, arthur, "r", [])
    self.assertEqual(ARTHUR_LEARNED_WITHOUT_THE_RADIO,
                     backend.asked[0]["state"]["other"]["learned"])

  def test_the_need_line_and_the_secret_line_are_two_doors(self):
    part = "Here is a brief description of Tomas Kessler.\nName: Tomas"
    self.assertEqual(part, world.inject(part, "Tomas Kessler", False))
    need = world.inject(part, "Tomas Kessler", False, need=True)
    added = need[len(part):].strip().splitlines()
    self.assertEqual(["What Tomas Kessler is looking for: %s" % NEED], added)
    # the need's own text names the transmitter; the secret's other words
    # do not reach this door
    self.assertNotIn("brother", need)
    secret = world.inject(part, "Tomas Kessler", True)
    added = secret[len(part):].strip().splitlines()
    self.assertEqual(["What Tomas Kessler knows and has told no one: %s"
                      % SECRET3], added)
    self.assertNotIn("capacitor", secret)
    both = world.inject(part, "Tomas Kessler", True, need=True)
    added = both[len(part):].strip().splitlines()
    self.assertEqual(2, len(added))
    self.assertTrue(added[0].startswith("What Tomas Kessler is looking for"))
    self.assertTrue(added[1].startswith("What Tomas Kessler knows"))
    # a persona with no stake gets no need line, whatever is asked
    self.assertEqual(part, world.inject(part, "Daniel Okafor", False, True))

  # --- the route (RP-D) ---

  def test_the_route_is_asked_at_the_discovery_and_the_day_parts(self):
    tomas, arthur, sam, personas = _cast()
    # 05:00 passes before the discovery: marked, never fired
    tomas.at("05:00")
    self.assertFalse(world.note_discovery(tomas, 1800))
    self.assertIsNone(world.route_due(tomas, 1800))
    # the discovery, at breakfast
    tomas.at("07:20").sees_the_radio()
    self.assertTrue(world.note_discovery(tomas, 2640))
    self.assertEqual("discovery", world.route_due(tomas, 2640))
    world.route_answered(tomas, "give_up", 2640, "discovery")
    self.assertIsNone(world.route_due(tomas, 2640))
    self.assertFalse(world.note_discovery(tomas, 2641))
    tomas.at("07:30")
    self.assertIsNone(world.route_due(tomas, 2700))
    # noon
    tomas.at("12:00")
    self.assertEqual("12:00", world.route_due(tomas, 4320))
    world.route_answered(tomas, "give_up", 4320, "12:00")
    tomas.at("12:10")
    self.assertIsNone(world.route_due(tomas, 4380))
    # four in the afternoon, answered ask: the intention is pending
    tomas.at("16:00")
    self.assertEqual("16:00", world.route_due(tomas, 5760))
    world.route_answered(tomas, "ask", 5760, "16:00")
    self.assertTrue(world.need_due("Tomas Kessler", "Arthur Burton"))
    self.assertFalse(world.need_due("Tomas Kessler", "Sam Moore"))
    self.assertFalse(world.need_due("Sam Moore", "Arthur Burton"))
    # delivered: the route completes, the intention holds until the
    # dialogue ends, and no further day-part asks
    world.ask_delivered("Tomas Kessler", "Arthur Burton", 5800)
    self.assertTrue(world.need_due("Tomas Kessler", "Arthur Burton"))
    world.conversation_ended("Arthur Burton", "Tomas Kessler")
    self.assertFalse(world.need_due("Tomas Kessler", "Arthur Burton"))
    t = world.state()["takers"]["Tomas Kessler"]
    self.assertEqual({"how": "ask", "step": 5800}, t["route_completed"])
    self.assertEqual(2640, t["discovery"])
    self.assertEqual(["discovery", "12:00", "16:00"],
                     [r["at"] for r in t["routes"]])
    kinds = [r["kind"] for r in self._world_events()]
    self.assertEqual(["discovery", "route", "route", "route",
                      "ask_delivered"], kinds)

  def test_at_most_four_routes_a_day_and_none_after_one_completed(self):
    tomas, arthur, sam, personas = _cast()
    tomas.at("07:20").sees_the_radio()
    world.note_discovery(tomas, 2640)
    for i in range(4):
      world.route_answered(tomas, "give_up", 2640 + i, "discovery")
    tomas.at("12:00")
    self.assertIsNone(world.route_due(tomas, 4320))
    world.configure(json.loads(json.dumps(W3)))
    tomas, arthur, sam, personas = _cast()
    tomas.at("07:20").sees_the_radio()
    world.note_discovery(tomas, 2640)
    world.route_answered(tomas, "take", 2640, "discovery")
    world.take_answered(tomas, "take", 3000)
    tomas.at("12:00")
    self.assertIsNone(world.route_due(tomas, 4320))

  def test_under_discovery_given_the_first_ask_is_at_five(self):
    section = json.loads(json.dumps(W3))
    section["discovery"] = "given"
    world.configure(section)
    tomas, arthur, sam, personas = _cast()
    tomas.at("04:50")
    self.assertIsNone(world.route_due(tomas, 1740))
    tomas.at("05:00")
    self.assertEqual("05:00", world.route_due(tomas, 1800))
    world.route_answered(tomas, "give_up", 1800, "05:00")
    # `known` before the radio is seen says so, and the discovery by
    # perception is still recorded and still asks
    self.assertIn("has not seen the shortwave radio himself",
                  world.known_text(tomas, RADIO, W3["objects"][RADIO]))
    tomas.at("07:20").sees_the_radio()
    self.assertTrue(world.note_discovery(tomas, 2640))
    self.assertEqual("discovery", world.route_due(tomas, 2640))

  # --- the opportunity and the take (RP-F) ---

  def test_the_take_is_asked_once_per_episode_after_a_take_route(self):
    tomas, arthur, sam, personas = _cast()
    maze = _W3Maze()
    tomas.sees_the_radio()
    world.note_discovery(tomas, 2640)
    # no route yet, no refusal: an episode opens but nothing is asked
    self.assertFalse(world.take_due(tomas, maze, personas, 5760))
    self.assertEqual(1, world.state()["takers"]["Tomas Kessler"]["opportunities"])
    arthur.at("16:01", IN_PUB)
    self.assertFalse(world.take_due(tomas, maze, personas, 5766))
    self.assertIsNone(world.state()["takers"]["Tomas Kessler"]["opportunity"])
    # the route said take: the next episode asks, once
    world.route_answered(tomas, "take", 5766, "16:00")
    arthur.at("16:05", IN_FLAT)
    self.assertTrue(world.take_due(tomas, maze, personas, 5790))
    world.take_answered(tomas, "leave", 5790)
    self.assertFalse(world.take_due(tomas, maze, personas, 5791))
    self.assertFalse(world.take_due(tomas, maze, personas, 5792))
    # the owner comes back and leaves again: a third episode, asked again
    arthur.at("16:20", IN_PUB)
    self.assertFalse(world.take_due(tomas, maze, personas, 5880))
    arthur.at("16:30", IN_FLAT)
    self.assertTrue(world.take_due(tomas, maze, personas, 5940))
    # Tomas leaves the pub: the episode closes
    tomas.at("16:40", ON_THE_STREET)
    self.assertFalse(world.take_due(tomas, maze, personas, 6000))
    t = world.state()["takers"]["Tomas Kessler"]
    self.assertEqual(3, t["opportunities"])
    self.assertIsNone(t["opportunity"])
    whats = [(r["what"], r["number"]) for r in self._world_events("opportunity")]
    self.assertEqual([("opens", 1), ("closes", 1), ("opens", 2), ("closes", 2),
                      ("opens", 3), ("closes", 3)], whats)

  def test_the_take_waits_for_a_free_step_and_follows_a_refusal(self):
    tomas, arthur, sam, personas = _cast()
    maze = _W3Maze()
    tomas.at("07:20").sees_the_radio()
    world.note_discovery(tomas, 2640)
    world.route_answered(tomas, "ask", 2640, "discovery")
    tomas.at("16:10")
    # an ask route and no refusal: never
    self.assertFalse(world.take_due(tomas, maze, personas, 5760))
    # the radio came up with the owner after the route: the refusal
    tomas.chatted("Arthur Burton", REFUSAL, "07:25")
    arthur.at("16:01", IN_PUB)
    world.take_due(tomas, maze, personas, 5766)
    arthur.at("16:05", IN_FLAT)
    tomas.scratch.chatting_with = "Sam Moore"
    self.assertFalse(world.take_due(tomas, maze, personas, 5790))
    tomas.scratch.chatting_with = None
    self.assertTrue(world.take_due(tomas, maze, personas, 5791))
    self.assertEqual(2, world.state()["takers"]["Tomas Kessler"]["opportunities"])
    # a refusal BEFORE the last route does not count
    world.configure(json.loads(json.dumps(W3)))
    tomas, arthur, sam, personas = _cast()
    tomas.sees_the_radio().chatted("Arthur Burton", REFUSAL, "07:25")
    world.note_discovery(tomas, 2640)
    tomas.at("12:00")
    world.route_answered(tomas, "give_up", 4320, "12:00")
    tomas.at("16:10")
    self.assertFalse(world.take_due(tomas, maze, personas, 5820))

  def test_a_take_makes_the_radio_missing_and_survives_a_fork(self):
    """RP-C, S1.3."""
    tomas, arthur, sam, personas = _cast()
    maze = _W3Maze()
    tomas.sees_the_radio()
    world.note_discovery(tomas, 2640)
    world.route_answered(tomas, "take", 2640, "discovery")
    world.take_due(tomas, maze, personas, 5790)
    world.take_answered(tomas, "take", 5790)
    held = world.state()["objects"][RADIO]
    self.assertEqual("missing", held["state"])
    self.assertEqual(5790 + 60, held["missing_from"])
    self.assertEqual("Tomas Kessler", held["taken_by"])
    self.assertEqual({"how": "take", "step": 5790},
                     world.state()["takers"]["Tomas Kessler"]["route_completed"])
    blank = (RADIO_ADDRESS, None, None, None)
    missing = world.missing_event(RADIO_ADDRESS)
    # while the action runs the tile is the paper's
    world.write_back(maze, 5800)
    self.assertEqual({blank}, maze.events_at((53, 20)))
    # from the step it ends, the tile reads missing
    world.write_back(maze, 5850)
    self.assertEqual({missing}, maze.events_at((53, 20)))
    self.assertEqual({missing}, maze.events_at((54, 20)))
    self.assertEqual("shortwave radio is missing",
                     "%s is %s" % (missing[0].split(":")[-1], missing[3]))
    # the paper's cleanup puts the blank back; the next write-back wins
    for tile in ((53, 20), (54, 20)):
      maze.tiles[tile[1]][tile[0]]["events"].add(blank)
    world.write_back(maze, 5851)
    self.assertEqual({missing}, maze.events_at((54, 20)))
    # saved with the sim, read by the fork of it
    folder = tempfile.mkdtemp(prefix="world_w3_")
    try:
      path = world.save(folder)
      self.assertEqual(os.path.join(folder, "world_state.json"), path)
      copied = os.path.join(folder, "fork")
      shutil.copytree(folder, copied)
      world.configure(json.loads(json.dumps(W3)))
      self.assertEqual("in place", world.object_state(RADIO))
      world.boot(copied, 5851)
      self.assertEqual("missing", world.object_state(RADIO))
      self.assertEqual(held, world.state()["objects"][RADIO])
      fresh = _W3Maze()
      world.write_back(fresh, 5851)
      self.assertEqual({missing}, fresh.events_at((53, 20)))
    finally:
      shutil.rmtree(folder, ignore_errors=True)
    take = self._world_events("take")
    self.assertEqual(1, len(take))
    self.assertEqual(5850, take[0]["missing_from"])
    self.assertEqual("Tomas Kessler", take[0]["subject"])

  # --- the states, from the taker's memory ---

  def test_the_states_are_built_from_the_takers_own_memory(self):
    import probe_route
    import probe_take
    tomas, arthur, sam, personas = _cast(arthur_tile=IN_PUB, sam_tile=IN_PUB)
    maze = _W3Maze()
    tomas.at("07:30").sees_the_radio()
    world.note_discovery(tomas, 2700)
    facts = world.route_facts(tomas, arthur, maze, personas)
    state = world.route_state(**facts)
    self.assertTrue(probe_route.whole(state))
    self.assertEqual(SECRET3, state["secret"])
    self.assertEqual(NEED, state["stake"])
    self.assertEqual(ARTHUR_LEARNED_WITHOUT_THE_RADIO, state["other"]["learned"])
    self.assertEqual("eating dinner in his apartment", state["other"]["doing"])
    self.assertEqual(PLANTED["Arthur Burton"], state["relationship"])
    self.assertEqual("February 13, 2023, 07:30 AM", state["time"])
    self.assertEqual("none", state["last_conversation"])
    self.assertIn("There is an old shortwave radio", state["known"])
    self.assertIn("It is Arthur Burton's.", state["known"])
    self.assertIn("has not asked him about it", state["known"])
    self.assertIn("Arthur Burton is in The Rose and Crown Pub, eating dinner",
                  state["present"])
    self.assertIn(PLANTED["Sam Moore"], state["present"])
    self.assertNotIn("Daniel", state["present"])
    # after the refusal: the owner's words are quoted, never judged
    tomas.chatted("Arthur Burton", REFUSAL, "07:25")
    state = world.route_state(**world.route_facts(tomas, arthur, maze, personas))
    self.assertIn("Tomas Kessler asked him about it. What Arthur Burton said "
                  "of it: \"That was my father's. I won't sell it or lend it "
                  "to anyone. Sorry.\"", state["known"])
    self.assertTrue(state["last_conversation"]["radio_came_up"])
    self.assertEqual("today at 07:25 AM", state["last_conversation"]["when"])
    self.assertEqual(2, len(state["last_conversation"]["lines"]))
    # nobody in the pub, Tomas alone; then Tomas outside and the pub empty
    arthur.at("16:05", IN_FLAT)
    sam.at("16:05", ON_THE_STREET)
    tomas.at("16:10")
    facts = world.take_facts(tomas, arthur, maze, personas)
    state = world.take_state(**facts)
    self.assertTrue(probe_route.whole(state, probe_take.FIELDS))
    self.assertEqual("No one else is in The Rose and Crown Pub; Tomas Kessler "
                     "is alone in the room.", state["present"])
    self.assertEqual("not in the room", state["other"]["doing"])
    self.assertIn("It is in place.", state["object"])
    self.assertIn("Arthur Burton keeps it there.", state["object"])
    self.assertNotIn("will not", state["object"])
    self.assertIn("does not know when he will be back", state["opportunity"])
    tomas.at("16:20", ON_THE_STREET)
    self.assertEqual("No one is in The Rose and Crown Pub.",
                     world.present_text(tomas, maze, personas,
                                        W3["objects"][RADIO]))
    # an unplanted person is named with nothing beside the name
    tomas.a_mem.seq_thought = [n for n in tomas.a_mem.seq_thought
                               if n.object != "Sam Moore"]
    sam.at("16:25", IN_PUB)
    self.assertIn("Tomas Kessler's memory holds nothing about Sam Moore",
                  world.present_text(tomas, maze, personas,
                                     W3["objects"][RADIO]))

  # --- act-on: pick | sample (RP-D's amendment) ---

  def test_act_on_sample_draws_with_the_probabilities_and_pick_never_draws(self):
    import random
    tomas, arthur, sam, personas = _cast()
    maze = _W3Maze()
    tomas.sees_the_radio()
    real = random.random
    drawn = []

    def fixed(value):
      def draw():
        drawn.append(value)
        return value
      return draw
    try:
      # pick: the Choice's answer, and random untouched
      random.random = fixed(0.95)
      with _TypedBackend(ROUTE_ANSWERS):
        answer = run_gpt_prompt.run_gpt_prompt_decide_route(
            tomas, arthur, maze, personas)
      self.assertEqual("ask", answer)
      self.assertEqual([], drawn)
      event = [r for r in _records("decision") if r["label"] == "decide_route"][-1]
      self.assertEqual("pick", event["acted_on"])
      self.assertEqual("ask", event["pick"])
      self.assertNotIn("drawn", event)
      self.assertEqual("Arthur Burton", event["other"])
      self.assertEqual(sorted(world_questions.ROUTE_COMPANIONS),
                       sorted(event["companions"]))
      # sample: one draw, walked in name order (ask .6, give_up .1, take .3)
      section = json.loads(json.dumps(W3))
      section["act-on"] = "sample"
      world.configure(section)
      with _TypedBackend(ROUTE_ANSWERS):
        answer = run_gpt_prompt.run_gpt_prompt_decide_route(
            tomas, arthur, maze, personas)
      self.assertEqual("take", answer)
      self.assertEqual([0.95], drawn)
      event = [r for r in _records("decision") if r["label"] == "decide_route"][-1]
      self.assertEqual("sample", event["acted_on"])
      self.assertEqual("ask", event["pick"])
      self.assertEqual("take", event["drawn"])
      self.assertEqual("take", event["answer"])
      random.random = fixed(0.65)
      with _TypedBackend(ROUTE_ANSWERS):
        self.assertEqual("give_up", run_gpt_prompt.run_gpt_prompt_decide_route(
            tomas, arthur, maze, personas))
      random.random = fixed(0.10)
      with _TypedBackend(TAKE_ANSWERS):
        self.assertEqual("leave", run_gpt_prompt.run_gpt_prompt_decide_to_take(
            tomas, arthur, maze, personas))
      # a backend error is the fail-safe, and nothing is drawn
      drawn[:] = []
      with _TypedBackend(error="HTTP 529"):
        self.assertEqual("give_up", run_gpt_prompt.run_gpt_prompt_decide_route(
            tomas, arthur, maze, personas))
      self.assertEqual([], drawn)
    finally:
      random.random = real
    # an act-on the section does not know refuses
    section["act-on"] = "roll"
    world.configure(section)
    with self.assertRaises(ValueError):
      world.act_on()

  # --- the call site, the splice and the trace ---

  def test_a_world_react_on_take_splices_the_fixed_action(self):
    from persona.cognitive_modules import plan
    tomas, arthur, sam, personas = _cast()
    maze = _W3Maze()
    tomas.at("16:10").sees_the_radio()
    llm_trace.set_step(5820, tomas.scratch.curr_time)
    spliced = []
    real = (plan._create_react, plan.run_gpt_prompt_decide_route,
            plan.run_gpt_prompt_decide_to_take)
    plan._create_react = lambda p, *args: spliced.append((p.name,) + args)
    plan.run_gpt_prompt_decide_route = lambda *a, **k: "take"
    plan.run_gpt_prompt_decide_to_take = lambda *a, **k: "take"
    try:
      plan._world_react(tomas, maze, personas)
      # the same step again: nothing more is asked or spliced
      plan._world_react(tomas, maze, personas)
    finally:
      (plan._create_react, plan.run_gpt_prompt_decide_route,
       plan.run_gpt_prompt_decide_to_take) = real
    self.assertEqual(1, len(spliced))
    name, description, minutes, address, event = spliced[0][:5]
    self.assertEqual("Tomas Kessler", name)
    self.assertEqual("taking the old shortwave radio from behind the counter",
                     description)
    self.assertEqual(10, minutes)
    self.assertEqual(RADIO_ADDRESS, address)
    self.assertEqual(("Tomas Kessler", "take", RADIO), event)
    self.assertEqual((RADIO_ADDRESS, "is", "being taken"), spliced[0][-1])
    t = world.state()["takers"]["Tomas Kessler"]
    self.assertEqual(5820, t["discovery"])
    self.assertEqual([{"step": 5820, "episode": 1}], t["takes"])
    self.assertEqual("missing", world.object_state(RADIO))
    kinds = [r["kind"] for r in self._world_events()]
    self.assertEqual(["discovery", "route", "opportunity", "take"], kinds)
    self.assertEqual("discovery", self._world_events("route")[0]["at"])

  def test_the_utterance_event_carries_need_and_secret(self):
    llm_trace.record_utterance("Tomas Kessler", "Arthur Burton", "That old "
                               "radio -- would you sell it?", False, False,
                               need=True)
    event = _records("utterance")[-1]
    self.assertTrue(event["need"])
    self.assertFalse(event["secret"])
    self.assertFalse(event["injected"])
    llm_trace.record_utterance("Tomas Kessler", "Arthur Burton", "...", True,
                               True)
    event = _records("utterance")[-1]
    self.assertFalse(event["need"])
    self.assertTrue(event["secret"])
    self.assertTrue(event["injected"])


class WithWorld(unittest.TestCase):

  def setUp(self):
    world.configure(dict(W2))
    llm_trace.set_simulation("test-world-%s" % self.id().rsplit(".", 1)[-1])

  def tearDown(self):
    world.configure(None)

  def test_withhold_sends_no_word_of_the_secret(self):
    """S2.1 and S2.3's first half."""
    prompt = _utterance_prompt(False)
    world.configure(None)
    self.assertEqual(_utterance_prompt(False), prompt)
    for word in ("transmitter", "capacitor", "brother", "shortwave"):
      self.assertNotIn(word, prompt)

  def test_reveal_adds_exactly_one_line_to_part_one(self):
    """S2.3's second half."""
    before = _utterance_prompt(False)
    after = _utterance_prompt(True)
    added = [l for l in after.splitlines() if l not in before.splitlines()]
    self.assertEqual(1, len(added))
    self.assertTrue(added[0].startswith(
        "What Tomas Kessler knows and has told no one: "))
    self.assertIn(SECRET, added[0])
    # 015-IP RP-E: the secret alone; the stake has its own door
    self.assertNotIn(STAKE, added[0])
    self.assertLess(after.index(added[0]), after.index("PART 2."))
    self.assertEqual(len(before.splitlines()) + 1, len(after.splitlines()))

  def test_only_the_secret_keeper_is_asked(self):
    real = converse.run_gpt_prompt_decide_to_reveal

    def boom(*a, **k):
      raise AssertionError("asked the reveal decision of the detective")
    converse.run_gpt_prompt_decide_to_reveal = boom
    try:
      self.assertFalse(converse._reveal(PRIYA, TOMAS, "housemates", []))
    finally:
      converse.run_gpt_prompt_decide_to_reveal = real

  def test_the_reveal_decision_is_one_request_with_its_companions(self):
    """LK-C, S3.1, S3.3."""
    chat = [["Tomas Kessler", "Morning."],
            ["Priya Raman", "What are you building, Tomas?"]]
    with _TypedBackend(REVEAL_ANSWERS) as backend:
      answer = run_gpt_prompt.run_gpt_prompt_decide_to_reveal(
          TOMAS, PRIYA, "Priya is his housemate.", chat)
    self.assertEqual("reveal", answer)
    self.assertEqual(1, len(backend.asked))
    asked = backend.asked[0]
    self.assertEqual("decide_to_reveal", asked["label"])
    self.assertEqual(set(world_questions.reveal_questions()),
                     set(asked["questions"]))
    state = asked["state"]
    self.assertEqual(SECRET, state["secret"])
    self.assertEqual(STAKE, state["stake"])
    self.assertEqual("Tomas Kessler", state["self"]["name"])
    self.assertEqual("extraversion: low (quiet)", state["self"]["profile"])
    self.assertEqual("Priya Raman", state["other"]["name"])
    self.assertEqual("Priya is his housemate.", state["relationship"])
    self.assertEqual("What are you building, Tomas?", state["last_utterance"])
    self.assertEqual(2, len(state["conversation"]))
    event = [r for r in _records("decision")
             if r["label"] == "decide_to_reveal"][-1]
    self.assertEqual("reveal", event["answer"])
    self.assertTrue(event["in_domain"])
    self.assertFalse(event["gave_up"])
    self.assertEqual("Priya Raman", event["other"])
    self.assertEqual("Tomas Kessler", event["subject"])
    self.assertEqual({"other_asks_about_subject": 0.9,
                      "other_offers_stake": 0.8,
                      "self_trusts_other": 0.6,
                      "reveal_serves_goal": 0.7}, event["companions"])

  def test_a_backend_error_is_the_fail_safe_withhold(self):
    with _TypedBackend(error="HTTP 529") as backend:
      answer = run_gpt_prompt.run_gpt_prompt_decide_to_reveal(
          TOMAS, PRIYA, "", [])
    self.assertEqual("withhold", answer)
    event = [r for r in _records("decision")
             if r["label"] == "decide_to_reveal"][-1]
    self.assertTrue(event["gave_up"])
    self.assertTrue(event["backend_error"])
    self.assertTrue(event["in_domain"])

  def test_the_utterance_event_is_whole_and_says_whether_injected(self):
    long_line = "x" * 900
    llm_trace.record_utterance("Tomas Kessler", "Priya Raman", long_line,
                               True, False)
    event = _records("utterance")[-1]
    self.assertEqual(long_line, event["text"])
    self.assertTrue(event["injected"])
    self.assertEqual("Priya Raman", event["listener"])

  def test_the_map_is_the_worlds_copy(self):
    """LK-O: the section's maze selects the copy, and only the three
    house strings differ from the paper's folder."""
    folder = utils.matrix_folder()
    self.assertEqual("%s/the_ville_w2/matrix" % utils.maze_assets_loc, folder)
    self.assertTrue(os.path.isdir(folder))
    for name in os.listdir(os.path.join(folder, "special_blocks")):
      with open(os.path.join(folder, "special_blocks", name)) as f:
        ours = f.read()
      with open(os.path.join(utils.env_matrix, "special_blocks", name)) as f:
        papers = f.read()
      self.assertEqual(
          papers.replace("Tamara Taylor and Carmen Ortiz's house",
                         "Priya Raman and Tomas Kessler's house")
                .replace("Tamara Taylor's room", "Priya Raman's room")
                .replace("Carmen Ortiz's room", "Tomas Kessler's room"),
          ours, name)
    for name in os.listdir(os.path.join(folder, "maze")):
      with open(os.path.join(folder, "maze", name)) as f:
        ours = f.read()
      with open(os.path.join(utils.env_matrix, "maze", name)) as f:
        papers = f.read()
      self.assertEqual(papers, ours, name)

  def test_a_maze_that_does_not_exist_is_refused(self):
    section = dict(W2)
    section["maze"] = "the_ville_nowhere"
    world.configure(section)
    with self.assertRaises(ValueError):
      utils.matrix_folder()

  def test_a_world_on_the_papers_backend_refuses_to_start(self):
    """012-ST JF-A: the derived track is typed only."""
    with self.assertRaises(RuntimeError):
      decision.check_ready()


def tearDownModule():
  shutil.rmtree(_TRACE_DIR, ignore_errors=True)


if __name__ == "__main__":
  unittest.main()
