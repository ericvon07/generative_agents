"""
File: build_w2.py
Description: builds the W2 simulations by subtraction from base_the_ville_n25
             (010-IP step 7, LK-E, LK-O, world file §9 and §11).

    ../../../.venv/bin/python build_w2.py            # writes both sims
    ../../../.venv/bin/python build_w2.py --check    # rebuilds in memory, diffs

What it writes, under ../../environment/frontend_server/storage/:

  base_the_ville_w2        the gated condition's base: the two residents of
                           the paper's house, Carmen Ortiz and Tamara Taylor,
                           renamed Tomas Kessler and Priya Raman, every text
                           field replaced by the signed text of the world
                           file's §9, and nothing private anywhere in it --
                           the secret and the stake live in the config's
                           world section (LK-B).
  base_the_ville_w2_text   the text condition's base (LK-F, w2j-text): the
                           same sim, with the secret and the stake appended
                           to Tomas's `currently` under clause a (none) --
                           the way the paper would carry a secret.

Each persona keeps the paper's scratch settings (vision, retention,
reflection thresholds...), the paper's spatial memory with the three house
strings renamed to the map copy's (LK-O), and starts with ONE memory: the
relationship sentence the author signed (§11), planted as a thought, so
that the paper's relationship summary (converse.generate_summarize_agent_
relationship) starts from it and not from nothing. Its embedding is asked
of the same local model the run uses (openai_config.json: embeddings).

The spawn tiles are the paper's for the two rooms (n25's environment/0.json
for Carmen and Tamara), the day starts at midnight (step 0), and the meta's
maze_name stays "the_ville" -- the frontend draws the paper's map; only the
backend's matrix folder is the copy, chosen by the config (utils.
matrix_folder).
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
ORIGIN = "base_the_ville_n25"
GATED = "base_the_ville_w2"
TEXT = "base_the_ville_w2_text"

# The cast (world file §9): the paper's resident -> the signed one.
CAST = {"Carmen Ortiz": "Tomas Kessler", "Tamara Taylor": "Priya Raman"}

# The three strings the map copy renames (LK-O), in spatial memory and in
# living_area.
RENAMES = [
    ("Tamara Taylor and Carmen Ortiz's house",
     "Priya Raman and Tomas Kessler's house"),
    ("Tamara Taylor's room", "Priya Raman's room"),
    ("Carmen Ortiz's room", "Tomas Kessler's room"),
]

# The signed texts, world file §9 (2026-09-25, "aprovado"), byte for byte;
# Priya's lifestyle re-signed the same day ("pode ser 6am"): the paper's
# wake_up_hour reads whole hours, and 5:30am fell to its fail-safe 8.
SIGNED = {
    "Tomas Kessler": {
        "first_name": "Tomas", "last_name": "Kessler", "age": 29,
        "innate": "quiet, orderly, loyal",
        "learned": ("Tomas Kessler is a shopkeeper at Harvey Oak Supply "
                    "Store, which he inherited from his uncle two years ago "
                    "and is still learning part by part. He repairs old "
                    "radios as a hobby and keeps his projects to himself; he "
                    "dislikes being asked what he is working on."),
        "currently": ("Tomas Kessler is managing Harvey Oak Supply Store and "
                      "lives with a housemate, Priya Raman. Lately he has "
                      "been getting up before dawn to work on something in "
                      "the house garden that he does not talk about, and he "
                      "has stopped going to the pub after closing."),
        "lifestyle": ("Tomas Kessler goes to bed around 10pm, awakes up "
                      "around 5am, eats dinner around 8pm."),
        "daily_plan_req": ("Tomas works alone in the house garden from 5am "
                           "until he leaves for Harvey Oak Supply Store, "
                           "sorts parts on the supply store product shelf "
                           "before opening at 8am, stands behind the counter "
                           "until 6pm, and goes straight home after closing."),
        "living_area": ("the Ville:Priya Raman and Tomas Kessler's house:"
                        "Tomas Kessler's room"),
    },
    "Priya Raman": {
        "first_name": "Priya", "last_name": "Raman", "age": 45,
        "innate": "methodical, kind, perceptive",
        "learned": ("Priya Raman is a retired schoolteacher who tutors "
                    "children from home in the afternoons and notices small "
                    "changes in the people around her. She keeps her "
                    "grandfather's old shortwave radio in her room and has "
                    "said she would give it to anyone who could get it "
                    "working again."),
        "currently": ("Priya Raman lives with a housemate, Tomas Kessler, "
                      "and is preparing lessons for the children she tutors. "
                      "She has heard that someone in the Ville is hiding "
                      "something, and she is spending her days paying "
                      "attention to who goes where at odd hours and asking "
                      "people about it, one question at a time."),
        "lifestyle": ("Priya Raman goes to bed around 10pm, awakes up around "
                      "6am, eats dinner around 6pm."),
        "daily_plan_req": ("Priya prepares lessons at the desk in her room "
                           "from 6am, takes her coffee out to the house "
                           "garden around 6:30am and keeps working there "
                           "until 8am, works at her desk until noon, walks to "
                           "Harvey Oak Supply Store and Johnson Park after "
                           "lunch, and tutors at home from 3pm until 5pm."),
        "living_area": ("the Ville:Priya Raman and Tomas Kessler's house:"
                        "Priya Raman's room"),
    },
}

# The secret and the stake (§9). In the gated sim they are NOT here: they
# live in the config's world section and reach a prompt only through the
# reveal decision (LK-B). In the text sim they sit inside `currently`, the
# way the paper would carry them, under clause a (none).
SECRET = ("Tomas is secretly building a radio transmitter in the house "
          "garden, from parts he takes off the supply store product shelf "
          "without recording them, to reach his brother outside the Ville. "
          "Nobody knows, and he intends to keep it that way.")
STAKE = ("The transmitter cannot work without a variable capacitor, a part "
         "Harvey Oak Supply Store does not stock. Priya Raman keeps an old "
         "shortwave radio of her grandfather's in her room, and its "
         "capacitor is the one Tomas needs.")
CLAUSE_A = ""

# The relationship, signed 2026-09-25 (§11, "manda bala"): the pair's first
# memory in both sims.
RELATIONSHIP = ("Tomas Kessler and Priya Raman have shared a house for two "
                "years. They get along well and talk most days about the "
                "house, the weather and the store, but each keeps their own "
                "projects and worries to themselves.")
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


def persona_files(paper_name, new_name, secret_in_currently, embedding):
  """The three bootstrap files of one persona, from the paper's."""
  base = "%s/%s/personas/%s/bootstrap_memory" % (STORAGE, ORIGIN, paper_name)
  with open(base + "/scratch.json") as f:
    scratch = json.load(f)
  signed = SIGNED[new_name]
  scratch["name"] = new_name
  scratch["act_event"] = [new_name, None, None]
  for key in ("first_name", "last_name", "age", "innate", "learned",
              "currently", "lifestyle", "daily_plan_req", "living_area"):
    scratch[key] = signed[key]
  if secret_in_currently:
    scratch["currently"] = " ".join(
        s for s in (signed["currently"], SECRET, STAKE, CLAUSE_A) if s)

  with open(base + "/spatial_memory.json") as f:
    spatial = json.loads(rename(f.read()))

  other = [n for n in CAST.values() if n != new_name][0]
  created = datetime.datetime.strptime(START + ", 00:00:00",
                                       "%B %d, %Y, %H:%M:%S")
  node = {
      "node_count": 1, "type_count": 1, "type": "thought", "depth": 1,
      "created": created.strftime("%Y-%m-%d %H:%M:%S"), "expiration": None,
      "subject": new_name, "predicate": "shares a house with",
      "object": other, "description": RELATIONSHIP,
      "embedding_key": RELATIONSHIP, "poignancy": 5,
      "keywords": [new_name, other], "filling": [],
  }
  nodes = {"node_1": node}
  embeddings = {RELATIONSHIP: embedding}
  kw_strength = {"kw_strength_event": {},
                 "kw_strength_thought": {new_name.lower(): 1,
                                         other.lower(): 1}}
  return scratch, spatial, nodes, embeddings, kw_strength


def build(name, secret_in_currently, embedding):
  """Everything the sim folder holds, as {relative path: object}."""
  with open("%s/%s/environment/0.json" % (STORAGE, ORIGIN)) as f:
    env0 = json.load(f)
  with open("%s/%s/reverie/meta.json" % (STORAGE, ORIGIN)) as f:
    meta = json.load(f)
  files = {}
  meta = copy.deepcopy(meta)
  meta["fork_sim_code"] = name
  meta["persona_names"] = [CAST[p] for p in CAST]
  files["reverie/meta.json"] = meta
  files["environment/0.json"] = {CAST[p]: env0[p] for p in CAST}
  for paper_name, new_name in CAST.items():
    scratch, spatial, nodes, embeddings, kw = persona_files(
        paper_name, new_name, secret_in_currently and new_name == "Tomas Kessler",
        embedding)
    root = "personas/%s/bootstrap_memory/" % new_name
    files[root + "scratch.json"] = scratch
    files[root + "spatial_memory.json"] = spatial
    files[root + "associative_memory/nodes.json"] = nodes
    files[root + "associative_memory/embeddings.json"] = embeddings
    files[root + "associative_memory/kw_strength.json"] = kw
  return files


def write(name, files):
  folder = "%s/%s" % (STORAGE, name)
  if os.path.isdir(folder):
    shutil.rmtree(folder)
  for rel, obj in files.items():
    path = os.path.join(folder, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      json.dump(obj, f, indent=2)
      f.write("\n")
  return folder


def check(name, files):
  """--check: what is on disk against a fresh build; the embedding is
  compared by length only (the model is not bit-deterministic)."""
  folder = "%s/%s" % (STORAGE, name)
  bad = []
  for rel, obj in files.items():
    path = os.path.join(folder, rel)
    if not os.path.exists(path):
      bad.append("missing " + rel)
      continue
    with open(path) as f:
      on_disk = json.load(f)
    if rel.endswith("embeddings.json"):
      same = ({k: len(v) for k, v in on_disk.items()}
              == {k: len(v) for k, v in obj.items()})
    else:
      same = on_disk == obj
    if not same:
      bad.append("differs " + rel)
  return bad


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--check", action="store_true")
  args = ap.parse_args(argv)
  embedding = embed(RELATIONSHIP)
  sims = {GATED: build(GATED, False, embedding),
          TEXT: build(TEXT, True, embedding)}
  if args.check:
    bad = {name: check(name, files) for name, files in sims.items()}
    for name, problems in bad.items():
      print("%s: %s" % (name, "as built" if not problems else problems))
    return 1 if any(bad.values()) else 0
  for name, files in sims.items():
    print("wrote", write(name, files), "(%d files)" % len(files))
  return 0


if __name__ == "__main__":
  sys.exit(main())
