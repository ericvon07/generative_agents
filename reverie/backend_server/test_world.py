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


def _utterance_prompt(reveal):
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
        reveal=reveal)
  finally:
    run_gpt_prompt.ChatGPT_safe_generate_response_OLD = real
  return seen["prompt"]


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
    self.assertIn(STAKE, added[0])
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
