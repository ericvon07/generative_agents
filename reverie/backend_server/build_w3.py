"""
File: build_w3.py
Description: builds the W3 map copy and the W3 simulations by subtraction from
             base_the_ville_n25 (015-IP step 1, RP-A, RP-B; world file
             docs/worlds/w3-the-radio-at-the-pub.md, S1-S3 and S8).

    ../../../.venv/bin/python build_w3.py            # writes the map and the sims
    ../../../.venv/bin/python build_w3.py --check    # rebuilds in memory, diffs

What it writes:

  ../../environment/frontend_server/static_dirs/assets/the_ville_w3/matrix/
      the paper's matrix folder (LK-O), with TWO houses renamed in the three
      special-blocks CSVs -- Yuriko Yamamoto's house is Daniel Okafor's, and
      Tamara Taylor and Carmen Ortiz's house is Tomas Kessler's house, its
      rooms `Tomas Kessler's room` and `spare room` -- and ONE object added:
      the row `32283, the Ville, <all>, shortwave radio` in
      game_object_blocks.csv and its two tiles, (53, 20) and (54, 20), behind
      the counter of The Rose and Crown Pub, painted in game_object_maze.csv.
      Every other byte is the paper's (world file S1, S5).

  ../../environment/frontend_server/storage/
      base_the_ville_w3                 w3-base and w3-discovery-given (RP-B:
                                        the twin D differs in the config only)
      base_the_ville_w3_undercover      twin A: the officer's trade node is not
                                        planted in the other three
      base_the_ville_w3_officer_pm      twin B: Daniel's daily_plan_req has him
                                        back at the pub around 3:30pm
      base_the_ville_w3_owner_reserved  twin C: Arthur's innate is `quiet,
                                        courteous, private`

The four personas come from the paper's twenty-five: Arthur Burton and Sam
Moore keep their names and houses; Carmen Ortiz becomes Tomas Kessler in the
pilot's house renamed; Yuriko Yamamoto becomes Daniel Okafor in her house
renamed. Every text field is the signed text of the world file's S2, byte
for byte. Each persona keeps the paper's scratch settings and the paper's
spatial memory with the house strings renamed; Sam's gains The Rose and
Crown Pub with its seven paper objects (the paper's Sam does not know the
pub, and the typed action_sector Choice is over the sectors he knows); the
radio is in nobody's spatial memory at midnight -- it enters by perception.

Each persona starts with the planted memories of S3 -- one thought node per
pair, the closeness rule of W2 S11 -- with the local embedding the run
would compute (openai_config.json: embeddings). Nothing private is anywhere
in a sim: the secret, the need and the tip live in the config's world
section (LK-B) and reach a prompt only through the seam's door. `--check`
also refuses a sim folder holding a canary word.

The spawn tiles are the paper's for the four rooms (n25's environment/
0.json), the day starts at midnight (step 0), and the meta's maze_name stays
"the_ville" -- the frontend draws the paper's map; only the backend's matrix
folder is the copy, chosen by the config (utils.matrix_folder).
"""
import argparse
import copy
import datetime
import json
import os
import shutil
import sys
import urllib.request

STORAGE = "../../environment/frontend_server/storage"
ASSETS = "../../environment/frontend_server/static_dirs/assets"
ORIGIN = "base_the_ville_n25"
PAPER_MAP = ASSETS + "/the_ville/matrix"
MAP = ASSETS + "/the_ville_w3/matrix"

BASE = "base_the_ville_w3"
UNDERCOVER = "base_the_ville_w3_undercover"
OFFICER_PM = "base_the_ville_w3_officer_pm"
OWNER_RESERVED = "base_the_ville_w3_owner_reserved"
SIMS = (BASE, UNDERCOVER, OFFICER_PM, OWNER_RESERVED)

# The cast (world file S2): the paper's resident -> the signed one, in the
# paper's persona order (n25's meta.json).
CAST = [
    ("Arthur Burton", "Arthur Burton"),
    ("Carmen Ortiz", "Tomas Kessler"),
    ("Sam Moore", "Sam Moore"),
    ("Yuriko Yamamoto", "Daniel Okafor"),
]

# The strings the map copy renames (S1), in the special-blocks CSVs, in
# spatial memory and in living_area. Two houses, not one: the pilot's house
# is renamed in the_ville_w2, not in the paper's folder this copy is made from.
RENAMES = [
    ("Yuriko Yamamoto's house", "Daniel Okafor's house"),
    ("Tamara Taylor and Carmen Ortiz's house", "Tomas Kessler's house"),
    ("Carmen Ortiz's room", "Tomas Kessler's room"),
    ("Tamara Taylor's room", "spare room"),
]

# The radio (S5): one game object, one row, two tiles. 32283 is the first
# id after the paper's last game object (32282, pool table) and is used by
# no block file.
RADIO = "shortwave radio"
RADIO_ID = "32283"
RADIO_TILES = [(53, 20), (54, 20)]

# The pub as the paper's Arthur knows it (S2): added to Sam's spatial memory.
PUB = "The Rose and Crown Pub"
PUB_ARENA = "pub"
PUB_OBJECTS = ["shelf", "refrigerator", "bar customer seating",
               "behind the bar counter", "kitchen sink", "cooking area",
               "microphone"]

# The signed texts, world file S2 (2026-09-25, "Assino como esta"), byte
# for byte. Every lifestyle names whole hours: the paper's wake_up_hour
# cannot parse a half hour (W2 S12).
SIGNED = {
    "Arthur Burton": {
        "first_name": "Arthur", "last_name": "Burton", "age": 42,
        "innate": "friendly, outgoing, generous",
        "learned": ("Arthur Burton is the owner and bartender of The Rose "
                    "and Crown Pub, which he has run for ten years and where "
                    "he loves to make people feel welcome. Behind the counter "
                    "he keeps an old shortwave radio that was his father's; "
                    "it has not worked in years and he will not sell it or "
                    "lend it to anyone."),
        "currently": ("Arthur Burton is running a beloved pub in the town "
                      "that has been around for 10 years, and has lately "
                      "started opening it for breakfast so that the regulars "
                      "have somewhere to start the day. He is also learning "
                      "more about mixology and creating unique cocktails."),
        "lifestyle": ("Arthur Burton goes to bed around midnight, awakes up "
                      "around 6am, eats dinner around 4pm."),
        "daily_plan_req": ("Arthur Burton opens The Rose and Crown Pub for "
                           "breakfast at 6:30am and stands at the counter "
                           "talking with whoever comes in, keeps the pub open "
                           "through the afternoon, steps out to his apartment "
                           "around 4pm to eat dinner for about an hour, and "
                           "comes back to stand at the counter until he "
                           "closes the pub at 10pm."),
        "living_area": "the Ville:Arthur Burton's apartment:main room",
    },
    "Tomas Kessler": {
        "first_name": "Tomas", "last_name": "Kessler", "age": 29,
        "innate": "quiet, orderly, loyal",
        "learned": ("Tomas Kessler is a shopkeeper at Harvey Oak Supply "
                    "Store, which he inherited from his uncle two years ago "
                    "and is still learning part by part. He repairs old "
                    "radios as a hobby and keeps his projects to himself; he "
                    "dislikes being asked what he is working on."),
        "currently": ("Tomas Kessler is managing Harvey Oak Supply Store and "
                      "lives alone in his house. Lately he has been getting "
                      "up before dawn to work on something in the house "
                      "garden that he does not talk about, and he has decided "
                      "to start taking his breakfast at The Rose and Crown "
                      "Pub before opening the store."),
        "lifestyle": ("Tomas Kessler goes to bed around 10pm, awakes up "
                      "around 5am, eats dinner around 8pm."),
        "daily_plan_req": ("Tomas works alone in the house garden from 5am, "
                           "has breakfast at The Rose and Crown Pub around "
                           "7am, sorts parts on the supply store product "
                           "shelf before opening Harvey Oak Supply Store at "
                           "8am, stands behind the counter until 3pm, goes "
                           "back to the pub around 3:30pm and stays until "
                           "about 6pm, and goes home after."),
        "living_area": "the Ville:Tomas Kessler's house:Tomas Kessler's room",
    },
    "Sam Moore": {
        "first_name": "Sam", "last_name": "Moore", "age": 65,
        "innate": "wise, resourceful, humorous",
        "learned": ("Sam Moore is a retired navy officer who loves to share "
                    "stories from his time in the military. He is always "
                    "full of interesting stories and advice. He has "
                    "breakfast at The Rose and Crown Pub most mornings and "
                    "goes back in the late afternoon to talk with whoever is "
                    "there."),
        "currently": ("Sam Moore lives with his wife of 40 years, Jennifer "
                      "Moore, and spends his free time tending the park and "
                      "is an avid reader."),
        "lifestyle": ("Sam Moore goes to bed around 9pm, awakes up around "
                      "5am, eats dinner around 5pm."),
        "daily_plan_req": ("Sam Moore has breakfast at The Rose and Crown Pub "
                           "around 7am, likes to take a walk around Johnson "
                           "Park and sit at Hobbs Cafe reading through the "
                           "middle of the day, and goes back to the pub "
                           "around 3pm to talk with whoever is there until "
                           "he goes home for dinner."),
        "living_area": "the Ville:Moore family's house:main room",
    },
    "Daniel Okafor": {
        "first_name": "Daniel", "last_name": "Okafor", "age": 38,
        "innate": "curious, discreet, easygoing",
        "learned": ("Daniel Okafor is a police officer who moved to the "
                    "Ville a month ago and takes his breakfast at The Rose "
                    "and Crown Pub, where he likes to sit with whoever is "
                    "talking and listen."),
        "currently": ("Daniel Okafor is settling into the Ville and getting "
                      "to know the people who come to the pub."),
        "lifestyle": ("Daniel Okafor goes to bed around 11pm, awakes up "
                      "around 6am, eats dinner around 7pm."),
        "daily_plan_req": ("Daniel Okafor has breakfast at The Rose and Crown "
                           "Pub around 7:30am and stays talking there until "
                           "about 9am, goes to The Willows Market and "
                           "Pharmacy for his groceries in the late morning, "
                           "and spends the afternoon and evening at home "
                           "reading."),
        "living_area": "the Ville:Daniel Okafor's house:main room",
    },
}

# The twins' fields (S2, S8): the one field each twin changes. w3-undercover
# changes no field (it does not plant the announced node); w3-discovery-given
# changes the config only.
TWIN_FIELDS = {
    BASE: {},
    UNDERCOVER: {},
    OFFICER_PM: {
        "Daniel Okafor": {
            "daily_plan_req": (
                "Daniel Okafor has breakfast at The Rose and Crown Pub around "
                "7:30am and stays talking there until about 9am, goes to The "
                "Willows Market and Pharmacy for his groceries in the late "
                "morning, has lunch at home, and goes back to the pub around "
                "3:30pm and stays there through the afternoon."),
        },
    },
    OWNER_RESERVED: {
        "Arthur Burton": {"innate": "quiet, courteous, private"},
    },
}

# What each knows of the others (S3): (holder, of whom, the planted
# sentence). One thought node per row, in this order, in the holder.
KNOWS = [
    ("Tomas Kessler", "Arthur Burton",
     "Arthur Burton is the owner of The Rose and Crown Pub. Tomas Kessler "
     "knows him by sight and by name, and no more."),
    ("Arthur Burton", "Tomas Kessler",
     "Tomas Kessler runs Harvey Oak Supply Store, which was his uncle's. "
     "Arthur Burton knows him by sight and by name, and no more."),
    ("Sam Moore", "Arthur Burton",
     "Sam Moore has had his breakfast at The Rose and Crown Pub most mornings "
     "for years. He and Arthur Burton are on easy terms and talk about the "
     "town and Sam's navy days."),
    ("Arthur Burton", "Sam Moore",
     "Sam Moore has had his breakfast at The Rose and Crown Pub most mornings "
     "for years. He and Arthur Burton are on easy terms and talk about the "
     "town and Sam's navy days."),
    ("Tomas Kessler", "Sam Moore",
     "Sam Moore is a retired navy officer who lives in the Ville. Tomas "
     "Kessler knows him by sight and by name, and no more."),
    ("Sam Moore", "Tomas Kessler",
     "Tomas Kessler runs Harvey Oak Supply Store. Sam Moore knows him by "
     "sight and by name, and no more."),
    ("Daniel Okafor", "Arthur Burton",
     "Arthur Burton is the owner of The Rose and Crown Pub. Daniel Okafor "
     "knows him by name and by trade, and no more."),
    ("Daniel Okafor", "Tomas Kessler",
     "Tomas Kessler runs Harvey Oak Supply Store. Daniel Okafor knows him by "
     "name and by trade, and no more."),
    ("Daniel Okafor", "Sam Moore",
     "Sam Moore is a retired navy officer who lives in the Ville. Daniel "
     "Okafor knows him by name and by trade, and no more."),
]

# The officer's trade, announced (S3, lever A): planted in the other three,
# except under w3-undercover.
ANNOUNCED = ("Daniel Okafor is a police officer who moved to the Ville a "
             "month ago.")
ANNOUNCED_TO = ["Arthur Burton", "Tomas Kessler", "Sam Moore"]

# The canary words of the secret, the need and the tip (S6): no sim file may
# hold one. `shortwave` is fact 3, in Arthur's learned, and is allowed.
CANARY = ["transmitter", "brother", "outside the Ville", "without recording",
          "takes parts", "taking parts", "capacitor", "the tip",
          "someone means to take", "intends to take", "warned off",
          "been told that someone"]

START = "February 13, 2023"


def _config():
  with open("../../openai_config.json") as f:
    return json.load(f)


def embed(text):
  """The embedding the run would compute for this text, from the same
  local model the config names (nomic-embed-text through Ollama)."""
  cfg = _config()
  url = cfg["embeddings-base-url"].rstrip("/") + "/embeddings"
  body = json.dumps({"input": [text.replace("\n", " ")],
                     "model": cfg["embeddings"]}).encode("utf-8")
  req = urllib.request.Request(
      url, data=body, headers={"Content-Type": "application/json",
                               "Authorization": "Bearer %s"
                               % cfg.get("embeddings-key", "")})
  with urllib.request.urlopen(req, timeout=60) as resp:
    return json.load(resp)["data"][0]["embedding"]


def rename(text):
  for old, new in RENAMES:
    text = text.replace(old, new)
  return text


# --- the map -----------------------------------------------------------------

def build_map():
  """The map copy, as {relative path: text}: the paper's folder with the
  house strings renamed and the radio added, every other byte the same."""
  files = {}
  with open(PAPER_MAP + "/maze_meta_info.json") as f:
    meta_text = f.read()
  files["maze_meta_info.json"] = meta_text
  width = int(json.loads(meta_text)["maze_width"])
  for name in sorted(os.listdir(PAPER_MAP + "/special_blocks")):
    with open(PAPER_MAP + "/special_blocks/" + name) as f:
      text = rename(f.read())
    if name == "game_object_blocks.csv":
      assert RADIO_ID not in text and RADIO not in text
      text += "\n%s, the Ville, <all>, %s" % (RADIO_ID, RADIO)
    files["special_blocks/" + name] = text
  for name in sorted(os.listdir(PAPER_MAP + "/maze")):
    with open(PAPER_MAP + "/maze/" + name) as f:
      text = f.read()
    if name == "game_object_maze.csv":
      cells = text.split(", ")
      assert ", ".join(cells) == text, "the maze CSV is not ', '-joined"
      for x, y in RADIO_TILES:
        assert cells[y * width + x] == "0", (x, y)
        cells[y * width + x] = RADIO_ID
      text = ", ".join(cells)
    files["maze/" + name] = text
  return files


# --- the sims ----------------------------------------------------------------

def _knows(new_name, announced):
  rows = [(other, text) for holder, other, text in KNOWS if holder == new_name]
  if announced and new_name in ANNOUNCED_TO:
    rows.append(("Daniel Okafor", ANNOUNCED))
  return rows


def persona_files(paper_name, new_name, fields, announced, embeddings):
  """The bootstrap files of one persona, from the paper's."""
  base = "%s/%s/personas/%s/bootstrap_memory" % (STORAGE, ORIGIN, paper_name)
  with open(base + "/scratch.json") as f:
    scratch = json.load(f)
  signed = dict(SIGNED[new_name])
  signed.update(fields.get(new_name, {}))
  scratch["name"] = new_name
  scratch["act_event"] = [new_name, None, None]
  for key in ("first_name", "last_name", "age", "innate", "learned",
              "currently", "lifestyle", "daily_plan_req", "living_area"):
    scratch[key] = signed[key]

  with open(base + "/spatial_memory.json") as f:
    spatial = json.loads(rename(f.read()))
  if new_name == "Sam Moore":
    assert PUB not in spatial["the Ville"]
    spatial["the Ville"][PUB] = {PUB_ARENA: list(PUB_OBJECTS)}

  created = datetime.datetime.strptime(START + ", 00:00:00",
                                       "%B %d, %Y, %H:%M:%S")
  nodes, node_embeddings, strength = {}, {}, {}
  for i, (other, text) in enumerate(_knows(new_name, announced), 1):
    nodes["node_%d" % i] = {
        "node_count": i, "type_count": i, "type": "thought", "depth": 1,
        "created": created.strftime("%Y-%m-%d %H:%M:%S"), "expiration": None,
        "subject": new_name, "predicate": "knows", "object": other,
        "description": text, "embedding_key": text, "poignancy": 5,
        "keywords": [new_name, other], "filling": [],
    }
    node_embeddings[text] = embeddings[text]
    for kw in (new_name.lower(), other.lower()):
      strength[kw] = strength.get(kw, 0) + 1
  kw_strength = {"kw_strength_event": {}, "kw_strength_thought": strength}
  return scratch, spatial, nodes, node_embeddings, kw_strength


def build(name, embeddings):
  """Everything the sim folder holds, as {relative path: object}."""
  with open("%s/%s/environment/0.json" % (STORAGE, ORIGIN)) as f:
    env0 = json.load(f)
  with open("%s/%s/reverie/meta.json" % (STORAGE, ORIGIN)) as f:
    meta = json.load(f)
  files = {}
  meta = copy.deepcopy(meta)
  meta["fork_sim_code"] = name
  meta["persona_names"] = [new for _, new in CAST]
  files["reverie/meta.json"] = meta
  files["environment/0.json"] = {new: env0[paper] for paper, new in CAST}
  for paper_name, new_name in CAST:
    scratch, spatial, nodes, node_embeddings, kw = persona_files(
        paper_name, new_name, TWIN_FIELDS[name], name != UNDERCOVER,
        embeddings)
    root = "personas/%s/bootstrap_memory/" % new_name
    files[root + "scratch.json"] = scratch
    files[root + "spatial_memory.json"] = spatial
    files[root + "associative_memory/nodes.json"] = nodes
    files[root + "associative_memory/embeddings.json"] = node_embeddings
    files[root + "associative_memory/kw_strength.json"] = kw
  return files


def sentences():
  texts = [text for _, _, text in KNOWS] + [ANNOUNCED]
  return sorted(set(texts))


# --- write and check ---------------------------------------------------------

def _dump(obj):
  if isinstance(obj, str):
    return obj
  return json.dumps(obj, indent=2) + "\n"


def canary_hits(folder):
  hits = []
  for root, _, names in os.walk(folder):
    for n in names:
      with open(os.path.join(root, n), errors="replace") as f:
        text = f.read()
      for word in CANARY:
        if word in text:
          hits.append("%s: %r" % (os.path.relpath(os.path.join(root, n),
                                                  folder), word))
  return hits


def write(folder, files):
  for text in (_dump(obj) for obj in files.values()):
    for word in CANARY:
      assert word not in text, "a canary word would be written: %r" % word
  if os.path.isdir(folder):
    shutil.rmtree(folder)
  for rel, obj in files.items():
    path = os.path.join(folder, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      f.write(_dump(obj))
  return folder


def check(folder, files):
  """--check: what is on disk against a fresh build; an embedding is
  compared by length only (the model is not bit-deterministic)."""
  bad = []
  for rel, obj in files.items():
    path = os.path.join(folder, rel)
    if not os.path.exists(path):
      bad.append("missing " + rel)
      continue
    with open(path) as f:
      on_disk = f.read()
    if isinstance(obj, str):
      same = on_disk == obj
    elif rel.endswith("embeddings.json"):
      same = ({k: len(v) for k, v in json.loads(on_disk).items()}
              == {k: len(v) for k, v in obj.items()})
    else:
      same = json.loads(on_disk) == obj
    if not same:
      bad.append("differs " + rel)
  for rel in _walk(folder):
    if rel not in files:
      bad.append("extra " + rel)
  bad.extend("canary " + hit for hit in canary_hits(folder))
  return bad


def _walk(folder):
  for root, _, names in os.walk(folder):
    for n in names:
      if n == ".DS_Store":
        continue
      yield os.path.relpath(os.path.join(root, n), folder)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--check", action="store_true")
  args = ap.parse_args(argv)
  embeddings = {text: embed(text) for text in sentences()}
  targets = {MAP: build_map()}
  for name in SIMS:
    targets["%s/%s" % (STORAGE, name)] = build(name, embeddings)
  if args.check:
    bad = {folder: check(folder, files) for folder, files in targets.items()}
    for folder, problems in bad.items():
      print("%s: %s" % (folder, "as built" if not problems else problems))
    return 1 if any(bad.values()) else 0
  for folder, files in targets.items():
    print("wrote", write(folder, files), "(%d files)" % len(files))
  return 0


if __name__ == "__main__":
  sys.exit(main())
