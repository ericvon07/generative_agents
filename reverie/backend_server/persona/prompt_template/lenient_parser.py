"""
File: lenient_parser.py
Description: Deviation D6 -- the paper's parsers stop throwing away
             an answer that is unambiguously the one the prompt asked for.

Three of the paper's validators refuse a correct answer over its wrapping:

  1. focal_pt's chat validator calls the TEXT path's cleanup, which prepends
     "1) " to the response -- a TypeError on the list the model correctly
     returns -- so the list is refused three times, the call falls through to
     the paper's old completion prompt, and if that fails too the fail-safe
     ["Who am I"] * n seeds reflection. On the generator the lab runs, every
     one of its rejections was a correct list (GT-8).
  2. ChatGPT_safe_generate_response cuts the response at the last "}" and never
     at the first "{", so a JSON object inside a markdown fence fails
     json.loads at sixteen call sites (GT-8). D11 asks the model not to fence;
     this is the parser's half, for a model that does not listen.
  3. The two location validators (action_sector, action_arena) demand a closing
     brace because the template ends in "Answer: {", and the cleanup removes
     only the closing one -- so "{Hobbs Cafe" is refused (OD-1) and
     "{common room}" reaches the maze wearing a brace and kills the run (GT-3).

The rule every function here keeps: the tolerance is
SYNTACTIC and BLIND TO CONTENT. No function is ever handed the option set, the
persona, or the prompt's inputs -- a validator that checked membership would be
the typed backend wearing a regex. What is unambiguous is accepted; what would
need a choice (a comma between two options, a triple with three parts after
the pre-filled subject) is still rejected, and counted.

Configuration, in openai_config.json:

  "lenient-parser": { "enabled": false }

Environment override: LENIENT_PARSER=0|1. Off -- the default -- the paper's
validators and cleanups run unchanged, byte for byte (law 2). One flag for the
three tolerances: they are one deviation with one reason. The manifest's
config_sha1 covers the whole config, so every run says which parser made it.
"""
import ast
import json
import os

_DEFAULT_ENABLED = False


def _load_enabled():
  enabled = _DEFAULT_ENABLED
  try:
    with open("../../openai_config.json", "r") as f:
      cfg = json.load(f)
    enabled = bool((cfg.get("lenient-parser") or {}).get("enabled",
                                                         _DEFAULT_ENABLED))
  except Exception:
    pass

  flag = os.environ.get("LENIENT_PARSER")
  if flag is not None:
    enabled = flag.strip().lower() in ("1", "true", "yes", "on")
  return enabled


_enabled = _load_enabled()


def enabled():
  """Whether the three tolerances below are applied in this run."""
  return _enabled


lenient_enabled = enabled


def set_enabled(flag):
  """Turn D6 on or off after import. The environment variable is how a run
  does it; this exists for the tests and for the replay tool."""
  global _enabled
  _enabled = bool(flag)


def status():
  """One line for the run's startup, next to the guard's."""
  if _enabled:
    return ("lenient parser (D6): on -- focal_pt keeps its list, the JSON "
            "cut starts at the first '{', the location trio tolerates a "
            "missing brace; LENIENT_PARSER=0 turns it off")
  return ("lenient parser (D6): off -- the paper's validators and cleanups "
          "run unchanged")


# --- tolerance 1: focal_pt keeps the list it was given -----------------------

def focal_pt_clean_up(response):
  """The list the model returned, as the list it is.

  The JSON path hands the validator the parsed "output" value: a Python list
  when the model wrote a JSON array, a string when it wrote the array inside
  quotes (the paper's own example output does exactly that). Both are one
  answer -- a list of strings -- and both come back as that list. Anything
  else raises, and the validator reads a raise as a rejection.
  """
  if isinstance(response, str):
    response = ast.literal_eval(response.strip())
  if not isinstance(response, list) or not response:
    raise ValueError("focal_pt: not a non-empty list")
  if not all(isinstance(item, str) for item in response):
    raise ValueError("focal_pt: not a list of str")
  return list(response)


def focal_pt_validate(response):
  try:
    focal_pt_clean_up(response)
    return True
  except Exception:
    return False


# --- tolerance 2: the JSON object starts at the first "{" --------------------

def json_cut(text):
  """The paper cuts at the last "}" and keeps everything before it; this also
  drops everything before the first "{", so a fenced object parses. With no
  brace at all it returns what the paper's cut returns: the empty string."""
  text = str(text)
  end = text.rfind("}") + 1
  start = text.find("{")
  if start < 0:
    start = 0
  return text[start:end]


# --- tolerance 3: the location trio tolerates a missing closing brace --------

def location_clean_up(response):
  """The paper's cut at the closing brace, kept; when no brace closes the
  answer, the cut is the end of the first line instead. Then every brace and
  the surrounding whitespace come off, so "{Hobbs Cafe", "Hobbs Cafe}" and
  "{Hobbs Cafe}" all read "Hobbs Cafe" -- and the maze is never handed the
  opening brace the template put in the model's mouth (GT-3)."""
  text = str(response)
  if "}" in text:
    text = text.split("}")[0]
  else:
    text = text.strip().split("\n")[0]
  return text.replace("{", "").replace("}", "").strip()


def location_validate(response):
  """Non-empty and no comma, as the paper asks; the closing brace is no longer
  demanded. A comma names more than one option and stays rejected."""
  if response is None:
    return False
  text = str(response)
  if len(text.strip()) < 1:
    return False
  if "," in text:
    return False
  return len(location_clean_up(text)) > 0
