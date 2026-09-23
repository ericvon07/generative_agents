"""
File: compare_backends.py
Description: Put two runs of the same window side by side -- the paper's
             prompts against the typed backend -- from their llm_trace files.

Usage (from reverie/backend_server, with the project venv):

  python compare_backends.py cmpB-llm cmpB-jev
  python compare_backends.py cmpB-llm cmpB-jev --decision action_arena

Each argument is the prefix of a condition's simulation name. The automatic
runner splits a run into stages (`<target>-s-<i>-<from>-<to>`), each with its
own trace file, so a prefix is what identifies a condition.

What it does NOT do is claim the two runs visit the same states. They diverge
the moment the first decision differs, and that divergence is the experiment.
Everything below is therefore a rate or a per-decision figure, never a paired
comparison -- except the step range, which says whether both conditions
covered the window at all. A condition that died early is the headline, not a
footnote, so it is printed first.

Reads only; it never touches a simulation.
"""
import argparse
import collections
import glob
import json
import os
import statistics
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "logs", "llm_trace")
CONFIG = os.path.join(SCRIPT_DIR, "..", "..", "openai_config.json")


def prices():
  """$ per 1M input tokens, per code path.

  A run mixes three: the decision path, the generation path (plans,
  reflection, dialogue -- which stay on the LLM in BOTH conditions) and
  embeddings. Pricing them all at one rate is how a local-model run acquires
  an imaginary bill, so they are kept apart.
  """
  out = {"jev": 0.042, "model": 0.0, "embedding": 0.0}
  try:
    with open(CONFIG) as f:
      cfg = json.load(f)
    out["model"] = float((cfg.get("model-costs") or {}).get("input", 0.0))
    out["embedding"] = float((cfg.get("embeddings-costs") or {})
                             .get("input", 0.0))
    out["jev"] = float(((cfg.get("jev") or {}).get("costs") or {})
                       .get("input", 0.042))
  except Exception:
    pass
  return out


def price_of(record, price_by_path):
  path = record.get("path")
  if path == "jev":
    return price_by_path["jev"]
  if path == "embedding":
    return price_by_path["embedding"]
  return price_by_path["model"]


def decision_label(record):
  """The two event kinds name the same decision differently: a "call" is
  labelled by stack inspection, so it carries the prompt builder's name
  (run_gpt_prompt_action_sector), while a "decision" carries the seam's label
  (action_sector). The typed backend passes the seam's label straight through.
  Strip the prefix so the two line up."""
  label = record.get("label") or ""
  prefix = "run_gpt_prompt_"
  return label[len(prefix):] if label.startswith(prefix) else label


class Condition(object):
  def __init__(self, prefix, directory):
    self.prefix = prefix
    self.files = sorted(
        p for p in glob.glob(os.path.join(directory, "*.jsonl"))
        if os.path.basename(p).startswith(prefix))
    self.records = []
    for path in self.files:
      with open(path, "r", encoding="utf-8") as f:
        records = []
        for line in f:
          line = line.strip()
          if not line:
            continue
          try:
            records.append(json.loads(line))
          except ValueError:
            pass  # a run killed mid-write leaves one truncated line
      _backfill_gave_up(records)
      self.records.extend(records)

    self.calls = [r for r in self.records if r.get("event") == "call"]
    self.decisions = [r for r in self.records if r.get("event") == "decision"]
    self.outcomes = [r for r in self.records if r.get("event") == "outcome"]
    self.rejects = [r for r in self.records if r.get("event") == "reject"]
    self.guards = [r for r in self.records if r.get("event") == "guard"]
    self.manifests = [r for r in self.records if r.get("event") == "manifest"]
    self.backend_down = [r for r in self.records
                         if r.get("event") == "backend_down"]
    # A decision the backend could not answer at all. Counted separately from
    # gave_up, which a healthy backend can also do: these are the ones where
    # nothing was measured because nothing came back.
    self.backend_errors = [r for r in self.decisions if r.get("backend_error")]
    self.steps = sorted(set(r["step"] for r in self.records
                            if r.get("step") is not None))
    backends = set(r.get("backend") for r in self.decisions if r.get("backend"))
    self.backend = "/".join(sorted(backends)) or "-"

  def provenance(self, *path):
    """A field of this condition's run manifest, or None for a trace written
    before manifests existed. A condition is several stage files, so the
    value is only reported when every stage agrees -- a run whose code or
    seed changed halfway through is not one condition."""
    values = []
    for m in self.manifests:
      node = m
      for key in path:
        node = (node or {}).get(key) if isinstance(node, dict) else None
      values.append(json.dumps(node, sort_keys=True, default=str))
    if not values:
      return None
    if len(set(values)) > 1:
      return "MIXED"
    return json.loads(values[0])

  @property
  def decision_calls(self):
    """The requests that served a decision, in either backend. The LLM path
    can spend several on one decision; the typed path spends exactly one."""
    wanted = set(r.get("label") for r in self.decisions)
    return [r for r in self.calls if decision_label(r) in wanted]

  @property
  def decision_requests(self):
    return len(self.decision_calls)

  @property
  def decision_seconds(self):
    return sum(r.get("latency_ms") or 0 for r in self.decision_calls) / 1000.0

  @property
  def seconds(self):
    return sum(r.get("latency_ms") or 0 for r in self.calls) / 1000.0

  @property
  def in_tok(self):
    return sum(r.get("in_tok") or 0 for r in self.calls)

  def cost(self, price_by_path):
    return sum((r.get("in_tok") or 0) * price_of(r, price_by_path)
               for r in self.calls) / 1e6

  def tokens_by_path(self):
    out = collections.Counter()
    for r in self.calls:
      out[r.get("path") or "?"] += r.get("in_tok") or 0
    return out


def gave_up(record):
  """Did this decision actually fall back, as opposed to merely landing on a
  value that equals the fail-safe?

  The distinction matters for the poignancy scores. Their fail-safe is 4,
  which is also a legal answer on a 1-10 scale, so counting "answer == 4" as a
  fallback reports a model that scored something a 4 as having failed. On the
  22 Sep 2026 run that inflated the typed backend's fail-safe rate on
  event_poignancy from 0% to 12.8%.

  Since 22 Sep 2026 the seam records this directly, as "gave_up": the LLM
  branch listens for the retry loop's own report and the typed branch knows it
  has no retries. Older traces have to be guessed at, and the guess misses a
  fallback whose value the model could also have produced -- so a number read
  off an old trace is a floor, not a rate.
  """
  recorded = record.get("gave_up")
  if recorded is not None:
    return bool(recorded)
  return bool(record.get("backend_error")) or str(record.get("answer")) == "False"


def _backfill_gave_up(records):
  """Derive "gave_up" for traces written before the seam recorded it.

  A retry loop that exhausts emits an outcome with failsafe=True, and the
  decision it belongs to is recorded immediately afterwards under the same
  prompt name (the outcome is labelled by stack inspection, so it carries the
  run_gpt_prompt_ prefix). Matching on that name rather than on adjacency
  keeps a generation prompt's fallback -- of which there are plenty in the
  typed condition, where generation is still the LLM's job -- from being
  charged to the next decision that happens to follow it.

  This is only for old traces. Runs from 22 Sep 2026 on carry the field.
  """
  pending = set()
  for r in records:
    event = r.get("event")
    if event == "outcome":
      if r.get("failsafe"):
        pending.add(decision_label(r))
    elif event == "decision":
      if r.get("gave_up") is None:
        label = r.get("label")
        r["gave_up"] = label in pending
        r["gave_up_derived"] = True
      pending.discard(r.get("label"))


def share(count, total):
  return "-" if not total else "%.1f%%" % (100.0 * count / total)


def fmt_code(condition):
  """What ran, named by its commit but judged by its tree.

  A condition is several stage files, and a commit from another session can
  land between two of them: on 22 Sep 2026 two documentation commits split
  cmpD2-llm's eight stages across two shas while the code was byte for byte
  identical. So the commit is only reported as MIXED when the tree moved
  too; when the tree held, the stages ran the same code and the row says so,
  naming the commit the run started on.
  """
  code = condition.provenance("code")
  if code is None:
    return "unrecorded"
  if code != "MIXED":
    sha = (code.get("sha") or "?")[:8]
    return "%s%s" % (sha, "+dirty" if code.get("dirty") else "")

  tree = condition.provenance("code", "tree")
  if tree in (None, "MIXED"):
    return "MIXED"
  # Same code, different commits: report the first stage's commit and say
  # the difference was outside the simulator.
  shas = [(m.get("code") or {}).get("sha") for m in condition.manifests]
  first = (shas[0] or "?")[:8] if shas else "?"
  dirty = any((m.get("code") or {}).get("dirty") for m in condition.manifests)
  return "%s%s (%d commits, same tree)" % (first, "+dirty" if dirty else "",
                                           len(set(shas)))


def fmt_tree(condition):
  tree = condition.provenance("code", "tree")
  if tree is None:
    return "unrecorded"
  return "MIXED" if tree == "MIXED" else str(tree)[:8]


def fmt_seed(condition):
  if not condition.manifests:
    return "unrecorded"
  seed = condition.provenance("seed")
  return "UNSEEDED" if seed is None else str(seed)


def summary(conditions, price_by_backend):
  rows = [
      ("trace files", lambda c: "%d" % len(c.files)),
      ("backend", lambda c: c.backend),
      ("code", fmt_code),
      ("code tree", fmt_tree),
      ("seed", fmt_seed),
      ("config", lambda c: str(c.provenance("config_sha1") or "unrecorded")),
      ("domain guard", lambda c: {True: "on", False: "off", None: "unrecorded"}
       .get(c.provenance("domain_guard"), "MIXED")),
      ("steps with activity", lambda c: (
          "%d..%d (%d)" % (c.steps[0], c.steps[-1], len(c.steps))
          if c.steps else "none")),
      ("model requests", lambda c: "%d" % len(c.calls)),
      ("time in requests", lambda c: "%.1f s" % c.seconds),
      ("input tokens", lambda c: "%d" % c.in_tok),
      ("input tokens by path", lambda c: ", ".join(
          "%s %d" % (k, v) for k, v in sorted(c.tokens_by_path().items()))),
      ("cost", lambda c: "$%.4f" % c.cost(price_by_backend)),
      ("logical prompts", lambda c: "%d" % len(c.outcomes)),
      ("attempts / prompt", lambda c: (
          "%.2f" % (sum(r.get("attempts") or 0 for r in c.outcomes)
                    / len(c.outcomes)) if c.outcomes else "n/a (no retries)")),
      ("discarded responses", lambda c: "%d" % len(c.rejects)),
      ("decisions", lambda c: "%d" % len(c.decisions)),
      ("requests / decision", lambda c: (
          "%.2f" % (float(c.decision_requests) / len(c.decisions))
          if c.decisions else "-")),
      ("seconds / decision", lambda c: (
          "%.2f s" % (c.decision_seconds / len(c.decisions))
          if c.decisions else "-")),
      ("out of domain", lambda c: share(
          sum(1 for r in c.decisions if r.get("in_domain") is False),
          sum(1 for r in c.decisions if r.get("in_domain") is not None))),
      # The rate above is two different findings added together, and they
      # point at different culprits: an answer the model invented, and the
      # paper's own get_fail_safe() literal handed back after the retry loop
      # rejected every attempt. On the 22 Sep window the second was three
      # quarters of the total, so reporting only the sum reads as a model
      # result when most of it is architectural.
      ("  ... the model's answer", lambda c: share(
          sum(1 for r in c.decisions
              if r.get("in_domain") is False and not gave_up(r)),
          sum(1 for r in c.decisions if r.get("in_domain") is not None))),
      ("  ... the code's fail-safe", lambda c: share(
          sum(1 for r in c.decisions
              if r.get("in_domain") is False and gave_up(r)),
          sum(1 for r in c.decisions if r.get("in_domain") is not None))),
      # What the restored guard had to put in their place. Measured apart
      # from the rate above on purpose: one says how often the contract
      # broke, the other how often the simulation had to be steered.
      ("guard substitutions", lambda c: "%d" % len(c.guards)),
      # The row that would have caught cmpD-jev on 22 Sep 2026 at a glance:
      # 209 of its 636 decisions never reached the backend, and every one of
      # them is a fail-safe standing where a measurement should be.
      ("backend errors", lambda c: (
          "%d (%s)" % (len(c.backend_errors),
                       share(len(c.backend_errors), len(c.decisions)))
          if c.backend_errors else "0")),
      ("answer == fail-safe", lambda c: share(
          sum(1 for r in c.decisions if r.get("is_fail_safe")),
          len(c.decisions))),
      ("actually gave up", lambda c: share(
          sum(1 for r in c.decisions if gave_up(r)), len(c.decisions))),
  ]
  width = max(len(name) for name, _ in rows)
  header = "%-*s" % (width, "") + "".join("  %-26s" % c.prefix
                                          for c in conditions)
  print(header)
  print("-" * len(header.rstrip()))
  for name, fn in rows:
    line = "%-*s" % (width, name)
    for c in conditions:
      try:
        line += "  %-26s" % fn(c)
      except Exception:
        line += "  %-26s" % "-"
    print(line)
  print("")


def per_decision(conditions, only=None):
  labels = []
  for c in conditions:
    for r in c.decisions:
      if r.get("label") and r["label"] not in labels:
        labels.append(r["label"])
  labels.sort()
  if only:
    labels = [l for l in labels if l == only]
  if not labels:
    print("no decisions recorded in either condition.\n")
    return

  header = ("%-20s  %-8s %6s %10s %10s %8s %9s %9s %9s"
            % ("decision", "cond", "n", "out of dom", "= failsafe", "gave up",
               "ms/dec", "req/dec", "p50 conf"))
  print("PER DECISION")
  print(header)
  print("-" * len(header))
  for label in labels:
    for c in conditions:
      group = [r for r in c.decisions if r.get("label") == label]
      if not group:
        print("%-20s  %-8s %6s %10s %10s %8s %9s %9s %9s"
              % (label, c.prefix[-8:], "0", "-", "-", "-", "-", "-", "-"))
        continue
      checked = [r for r in group if r.get("in_domain") is not None]
      lat = [r["latency_ms"] for r in c.calls
             if decision_label(r) == label and r.get("latency_ms") is not None]
      conf = [r["confidence"] for r in group if r.get("confidence") is not None]
      # For the LLM path a decision may cost several requests -- a retry is
      # still time the step waits for -- so this is the total request time
      # divided by decisions, not the median of one request.
      per = (sum(lat) / len(group)) if lat and group else None
      print("%-20s  %-8s %6d %10s %10s %8s %9s %9s %9s"
            % (label, c.prefix[-8:], len(group),
               share(sum(1 for r in checked if r["in_domain"] is False),
                     len(checked)),
               share(sum(1 for r in group if r.get("is_fail_safe")),
                     len(group)),
               share(sum(1 for r in group if gave_up(r)), len(group)),
               "-" if per is None else "%.0f" % per,
               "-" if not lat else "%.2f" % (float(len(lat)) / len(group)),
               "-" if not conf else "%.3f" % statistics.median(conf)))
    print("")


def answers(conditions, label):
  print("ANSWER DISTRIBUTION -- %s" % label)
  for c in conditions:
    counts = collections.Counter(str(r.get("answer"))
                                 for r in c.decisions
                                 if r.get("label") == label)
    total = sum(counts.values())
    if not total:
      print("  %-24s (none)" % c.prefix)
      continue
    print("  %s (%d)" % (c.prefix, total))
    for value, n in counts.most_common(10):
      print("      %-32s %5d  %s" % (value[:32], n, share(n, total)))
  print("")


def provenance_warnings(conditions):
  """Everything that would make the table above a comparison of two things
  that are not the same experiment. Printed rather than raised: an old trace
  is still worth reading, as long as nobody reads it as a controlled one."""
  out = []
  for c in conditions:
    if c.backend_errors:
      first = min((r.get("step") for r in c.backend_errors
                   if r.get("step") is not None), default="?")
      out.append("%s: %d of %d decisions never reached the backend (from step "
                 "%s on) -- those rows are fail-safes, not measurements, and "
                 "every rate in this table is diluted by them"
                 % (c.prefix, len(c.backend_errors), len(c.decisions), first))
    if c.backend_down:
      out.append("%s: the run stopped because the backend was down (%s)"
                 % (c.prefix, c.backend_down[0].get("error")))
  if any(not c.manifests for c in conditions):
    out.append("a condition has no run manifest -- trace written before "
               "22 Sep 2026; its code, seed and config are unknown")
  seeds = set(fmt_seed(c) for c in conditions)
  if "UNSEEDED" in seeds:
    out.append("a condition ran unseeded: its trajectory is not reproducible")
  trees = set(fmt_tree(c) for c in conditions if c.manifests)
  if "MIXED" in trees:
    out.append("a condition's stages ran different code: the simulator "
               "changed in the middle of the run")
  elif len(trees) > 1:
    out.append("the conditions ran different code: %s"
               % ", ".join(sorted(trees)))
  if any(c.provenance("code", "dirty") for c in conditions):
    out.append("a condition ran from a dirty tree: the commit does not "
               "describe what produced these numbers")
  configs = set(str(c.provenance("config_sha1")) for c in conditions
                if c.manifests)
  if len(configs) > 1:
    out.append("the conditions ran different configuration (%s) -- expected "
               "when the backend flag lives in the file, not when it is the "
               "environment that differs" % ", ".join(sorted(configs)))
  guards = set(str(c.provenance("domain_guard")) for c in conditions
               if c.manifests)
  if len(guards) > 1:
    out.append("the domain guard was on in one condition and off in the "
               "other: the runs are not comparable")
  return out


def main():
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("conditions", nargs="+",
                      help="simulation-name prefixes, one per condition")
  parser.add_argument("--dir", default=DEFAULT_DIR, help="trace directory")
  parser.add_argument("--decision",
                      help="also break down the answers for this one decision")
  args = parser.parse_args()

  conditions = [Condition(p, args.dir) for p in args.conditions]
  missing = [c.prefix for c in conditions if not c.files]
  if missing:
    sys.exit("no trace file starts with: %s (in %s)"
             % (", ".join(missing), os.path.abspath(args.dir)))

  print("=" * 78)
  print("CONDITIONS: %s" % "  vs  ".join(c.prefix for c in conditions))
  print("=" * 78)
  summary(conditions, prices())
  for warning in provenance_warnings(conditions):
    print("  !! %s" % warning)
  if any(provenance_warnings(conditions)):
    print("")
  per_decision(conditions, args.decision)
  print('  "= failsafe" counts answers equal to the decision\'s fail-safe '
        'value. For a 1-10')
  print('  score whose fail-safe is 4, a genuine 4 is counted there too; '
        '"gave up" is the')
  print('  column that means the backend actually failed.\n')
  if args.decision:
    answers(conditions, args.decision)


if __name__ == "__main__":
  main()
