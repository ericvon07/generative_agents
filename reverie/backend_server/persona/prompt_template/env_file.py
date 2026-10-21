"""
File: env_file.py
Description: Read the `.env` beside the fork, or beside the lab, into the
             process environment.

The fork needs one secret it may not hold: the key for the routed remote
model (the key lives in the environment, never in a config file). Nothing in the paper's code loads a `.env`, and the run scripts do not
either, so a key the author put in `<lab>/.env` never reached the process.
This module is that one missing step, and it is deliberately smaller than
python-dotenv: no interpolation, no `export` parsing, no multi-line values.

**An existing variable always wins.** A value already in the environment was
put there by whoever launched the run; a file on disk does not get to
override that, or `OPEN_ROUTER_API=... ./run_backend.sh` would silently mean
something else.

**Three kinds of whitespace are stripped**, because the author's line carries
all three and a key with a quote around it authenticates as nothing:

    OPEN_ROUTER_API = 'sk-or-v1-...'
                   ^ ^ ^           ^
                   | | |           the closing quote
                   | | the leading space of the value
                   | the trailing space of the name
                   (and the name's leading space, if any)

The file is never rewritten: it is the author's, it holds a secret, and a
reader that tolerates the line is worth more than a file that was normalised
once and will be typed the same way again.
"""
import os

# cwd is reverie/backend_server, as it is for openai_config.json.
_CANDIDATES = ("../../.env",      # the fork's own
               "../../../.env")   # the lab's, one level up (LB-1)

_loaded = []


def _parse(line):
  """One `NAME=value` line, or None for a blank or a comment."""
  line = line.strip()
  if not line or line.startswith("#"):
    return None
  if "=" not in line:
    return None
  name, _, value = line.partition("=")
  name = name.strip()
  if not name:
    return None
  value = value.strip()
  if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
    value = value[1:-1]
  return name, value


def load():
  """Read every candidate `.env` into os.environ, without overriding.

  Called once, at import of gpt_structure. Never raises: a missing or
  unreadable `.env` is the normal case for a run whose keys are already in
  the environment.

  RETURNS:
    The list of files actually read, for the startup line.
  """
  global _loaded
  _loaded = []
  for path in _CANDIDATES:
    try:
      if not os.path.exists(path):
        continue
      with open(path, "r", encoding="utf-8") as f:
        for line in f:
          parsed = _parse(line)
          if parsed is None:
            continue
          name, value = parsed
          if name not in os.environ:
            os.environ[name] = value
      _loaded.append(path)
    except Exception as e:
      print("env_file: could not read %s (%s)" % (path, e))
  return _loaded


def loaded():
  """Which `.env` files were read. Never their contents."""
  return list(_loaded)
