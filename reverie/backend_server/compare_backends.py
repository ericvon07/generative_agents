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

Three of the per-decision columns are about the retry loop rather than the
model: of the responses the loop threw away, how many already named one of
the options the prompt had offered. On the 22 Sep 2026 window that was 241 of
241 -- the loop was rejecting right answers in the wrong shape and then
writing a literal into the world. Those columns read the full text the trace
keeps for a decision rejection; on a trace written before 23 Sep 2026 they
read the 200-character head instead and the output says the figure is a
floor.
"""
import argparse
import collections
import glob
import json
import os
import re
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
    # A trace written before 006-IP has no finish_reason on any call, so it
    # cannot say whether a rejection was a truncation. That is a question it
    # was never asked, and the answer is "unknown" -- printing zero would be
    # a finding about the model made out of a missing field.
    self.knows_finish_reason = any("finish_reason" in r for r in self.calls)
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


def normalise(text):
  """One spelling for a domain member and for the text that may contain it.

  The same three moves record_decision makes when it asks whether an answer
  is in the set -- strip, drop the stray brace the prompt template invites,
  lower -- plus collapsing whitespace, because a response that arrives on two
  lines names the same place as one that does not.
  """
  text = str(text).strip()
  text = text.strip("{}").strip()
  return re.sub(r"\s+", " ", text).lower()


def members_named(text, domain):
  """Which members of <domain> the text names, by whole-word containment.

  Whole-word, not plain substring: "bed" must not match "bedroom". Containment
  and not equality, because the answers being read are the ones the validator
  refused, and it refused most of them for being a sentence -- "The common
  room in the Dorm for Oak Hill College." is a valid sector inside prose.

  The cost of containment is the overcount 002-ST named as this measurement's
  main risk: a domain whose members nest ("Oak Hill College" inside "Dorm for
  Oak Hill College") matches twice on one name. So the members are returned
  rather than a boolean, the caller counts a multiple match once, and the
  ambiguous total is printed beside the figure instead of being assumed away.
  """
  normal = normalise(text)
  out = []
  for member in domain:
    if member == "...":  # the decision event truncates a domain over 40
      continue
    needle = normalise(member)
    if not needle:
      continue
    if re.search(r"(?<!\w)%s(?!\w)" % re.escape(needle), normal):
      out.append(member)
  return out


class Rejection(object):
  """One discarded response on a closed-set decision, read against the set
  that decision was offered."""

  __slots__ = ("label", "persona", "step", "attempt", "reason", "text",
               "whole", "matched", "verbatim", "truncated")

  def __init__(self, record, domain):
    self.label = decision_label(record)
    self.persona = record.get("persona")
    self.step = record.get("step")
    self.attempt = record.get("attempt")
    self.reason = record.get("reason")
    # The full text since 23 Sep 2026; before that the head is all there is,
    # and <whole> is what tells the reader which of the two was read.
    self.whole = "resp_full" in record
    self.text = record.get("resp_full") or record.get("resp_head") or ""
    self.matched = members_named(self.text, domain)
    self.verbatim = normalise(self.text) in set(
        normalise(m) for m in domain if m != "...")
    # "truncated" from the validate branch, "truncated:JSONDecodeError" and
    # friends from the exception branch; both mean the provider stopped the
    # answer for the token budget.
    self.truncated = str(self.reason or "").startswith("truncated")


def rejections_by_decision(condition):
  """Pair every reject with the decision it belongs to, on (label, persona,
  step) -- the triple 002-ST paired by hand.

  A reject whose label is not one of the closed-set decisions is a generation
  rejection and is not here at all: the daily schedules and the conversations
  are the bulk of the 758 in a window and they answer no closed question.

  A reject that names a decision but matches none is returned under
  "unpaired" and printed, never dropped. It is the only evidence there would
  be that the pairing itself is wrong, and the reader has to see it.
  """
  by_key = {}
  disagreed = set()
  for r in condition.decisions:
    key = (decision_label(r), r.get("persona"), r.get("step"))
    if key in by_key:
      # Several decisions of the same kind in one step (the poignancy scores
      # do this). Harmless while they were offered the same set; if they were
      # not, there is no way to tell which one a reject belongs to.
      if json.dumps(by_key[key].get("domain"), sort_keys=True, default=str) \
         != json.dumps(r.get("domain"), sort_keys=True, default=str):
        disagreed.add(key)
      continue
    by_key[key] = r

  labels = set(decision_label(r) for r in condition.decisions)
  out = collections.defaultdict(list)
  unpaired = collections.Counter()
  scoreless = collections.Counter()
  for r in condition.rejects:
    label = decision_label(r)
    if label not in labels:
      continue  # a generation prompt
    key = (label, r.get("persona"), r.get("step"))
    decision = by_key.get(key)
    if decision is None or key in disagreed:
      unpaired[label] += 1
      continue
    domain = [m for m in (decision.get("domain") or []) if m != "..."]
    if not domain:
      # A score was offered no set, so there is no membership to recover
      # into. Counted apart: the poignancy rejections are real -- 77 in the
      # 22 Sep window -- and reporting them as zero recoverable would read as
      # a finding about the model rather than as a question that does not
      # apply to a 1-10 scale.
      scoreless[label] += 1
      continue
    out[label].append(Rejection(r, domain))
  return out, unpaired, scoreless


def rejection_stats(condition):
  """Per decision label: how many rejections, how many already held a valid
  choice, how many were one verbatim, how many named more than one.

  "recoverable" is None where the question does not apply -- a score, which
  was offered no set -- and that is not the same as zero."""
  paired, unpaired, scoreless = rejections_by_decision(condition)
  stats = {}

  def blank(label):
    return stats.setdefault(label, {"n": 0, "recoverable": 0, "verbatim": 0,
                                    "ambiguous": 0, "floor": 0, "unpaired": 0,
                                    "truncated": 0})

  for label, rejections in paired.items():
    row = blank(label)
    row["n"] = len(rejections)
    row["recoverable"] = sum(1 for r in rejections if r.matched)
    row["verbatim"] = sum(1 for r in rejections if r.verbatim)
    row["ambiguous"] = sum(1 for r in rejections
                           if len(set(normalise(m) for m in r.matched)) > 1)
    row["floor"] = sum(1 for r in rejections if not r.whole)
    row["truncated"] = sum(1 for r in rejections if r.truncated)
  for label, n in unpaired.items():
    blank(label)["unpaired"] = n
  for label, n in scoreless.items():
    row = blank(label)
    row["n"] += n
    row["recoverable"] = None
  return stats


def has_retry_loop(condition, label):
  """Whether this condition's decisions of this kind went through a retry
  loop at all. The typed backend has none, so "of the rejections, how many
  held a valid answer" is not a zero there -- it is a question that does not
  apply, and the table says so with a dash."""
  backends = set(r.get("backend") for r in condition.decisions
                 if decision_label(r) == label)
  return backends != {"jev"}


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
      ("world", world_of),
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

  row_fmt = "%-20s  %-8s %6s %10s %10s %8s %9s %12s %9s %9s %9s %9s %9s"
  header = (row_fmt
            % ("decision", "cond", "n", "out of dom", "= failsafe", "gave up",
               "rejected", "already valid", "verbatim", "truncated",
               "ms/dec", "req/dec", "p50 conf"))
  stats = dict((id(c), rejection_stats(c)) for c in conditions)
  print("PER DECISION")
  print(header)
  print("-" * len(header))
  for label in labels:
    for c in conditions:
      group = [r for r in c.decisions if r.get("label") == label]
      if not group:
        print(row_fmt % (label, c.prefix[-8:], "0", "-", "-", "-",
                         "-", "-", "-", "-", "-", "-", "-"))
        continue
      checked = [r for r in group if r.get("in_domain") is not None]
      lat = [r["latency_ms"] for r in c.calls
             if decision_label(r) == label and r.get("latency_ms") is not None]
      conf = [r["confidence"] for r in group if r.get("confidence") is not None]
      # For the LLM path a decision may cost several requests -- a retry is
      # still time the step waits for -- so this is the total request time
      # divided by decisions, not the median of one request.
      per = (sum(lat) / len(group)) if lat and group else None
      row = stats[id(c)].get(label)
      if not has_retry_loop(c, label):
        # No loop, so nothing was ever discarded: three dashes, not three
        # zeroes. A zero would read as a backend that discarded nothing on
        # its way to an answer; this backend has no way to discard.
        rejected = recoverable = verbatim = truncated = "-"
      elif row is None:
        rejected, recoverable, verbatim, truncated = "0", "-", "-", "-"
      else:
        rejected = "%d" % row["n"]
        if row["recoverable"] is None:
          recoverable = verbatim = "n/a (score)"
        else:
          recoverable = "%d (%s)" % (row["recoverable"],
                                     share(row["recoverable"], row["n"]))
          verbatim = "%d" % row["verbatim"]
        # A trace that never recorded finish_reason cannot answer this, and
        # a zero here would read as "the budget was never the problem" --
        # which is the opposite of what SR-1 found on exactly these rows.
        truncated = ("%d" % row["truncated"] if c.knows_finish_reason
                     else "unknown")
      print(row_fmt
            % (label, c.prefix[-8:], len(group),
               share(sum(1 for r in checked if r["in_domain"] is False),
                     len(checked)),
               share(sum(1 for r in group if r.get("is_fail_safe")),
                     len(group)),
               share(sum(1 for r in group if gave_up(r)), len(group)),
               rejected, recoverable, verbatim, truncated,
               "-" if per is None else "%.0f" % per,
               "-" if not lat else "%.2f" % (float(len(lat)) / len(group)),
               "-" if not conf else "%.3f" % statistics.median(conf)))
    print("")
  for note in rejection_notes(conditions, stats):
    print("  %s" % note)
  if any(rejection_notes(conditions, stats)):
    print("")


def rejection_notes(conditions, stats):
  """Everything the three columns above cannot say inside a cell.

  All three are honesty, not decoration. The ambiguous total is the overcount
  002-ST predicted, reported rather than assumed away. The unpaired total is
  the only sign there would be that pairing on (label, persona, step) is
  wrong. The floor note is the same admission _backfill_gave_up already makes
  for an old trace: a figure read from a 200-character head is a lower bound,
  because the answer may have been past the cut.
  """
  out = []
  for c in conditions:
    rows = stats[id(c)]
    # Only the rejections the question applies to: a score was offered no
    # set, so its rejections belong in neither the numerator nor the
    # denominator of "how many already held a valid choice".
    total = sum(r["n"] for r in rows.values() if r["recoverable"] is not None)
    ambiguous = sum(r["ambiguous"] for r in rows.values())
    unpaired = sum(r["unpaired"] for r in rows.values())
    floor = sum(r["floor"] for r in rows.values())
    if ambiguous:
      out.append("%s: %d of %d rejections named more than one member of the "
                 "set they were offered (a domain whose names nest does "
                 "this). Each counts once under \"already valid\"; "
                 "--decision <label> prints what matched."
                 % (c.prefix, ambiguous, total))
    if unpaired:
      out.append("!! %s: %d rejections carry a decision's name but match no "
                 "decision at the same persona and step -- the pairing is "
                 "missing something and the figures above are short by that "
                 "much" % (c.prefix, unpaired))
    if floor:
      out.append("%s: %d of %d rejections were read from a 200-character "
                 "head (trace written before 23 Sep 2026, which keeps no "
                 "full text) -- \"already valid\" is a floor, not a rate"
                 % (c.prefix, floor, total))
  return out


def rejected_responses(conditions, label, limit=12):
  """What the retry loop actually threw away on one decision, and which
  member of the set each discarded response named.

  This is where RA-D's "which member matched" lives: a count that says 126 of
  126 rejections held a valid sector is only believable next to the sentences
  it counted."""
  print("REJECTED RESPONSES -- %s" % label)
  for c in conditions:
    paired, unpaired, scoreless = rejections_by_decision(c)
    rejections = paired.get(label, [])
    if not rejections:
      why = "no retry loop" if not has_retry_loop(c, label) else "none"
      if scoreless.get(label):
        why = "%d rejections, but a score is offered no set" % scoreless[label]
      print("  %-24s (%s)" % (c.prefix, why))
      continue
    print("  %s (%d rejections, %d already valid)"
          % (c.prefix, len(rejections),
             sum(1 for r in rejections if r.matched)))
    seen = collections.Counter()
    first = {}
    for r in rejections:
      key = normalise(r.text)
      seen[key] += 1
      first.setdefault(key, r)
    for key, n in seen.most_common(limit):
      r = first[key]
      matched = ", ".join(r.matched) if r.matched else "nothing in the set"
      print("      %-44s %4d  %s%s"
            % (r.text.replace("\n", " ")[:44], n, matched,
               " [verbatim]" if r.verbatim else ""))
    if len(seen) > limit:
      print("      ... and %d more distinct responses" % (len(seen) - limit))
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


class CrossWorldPair(Exception):
  """Two conditions from two worlds. Raised, not warned (010-IP LK-G): the
  one provenance check that is not a caveat, because a W1 rate and a W2 rate
  are two experiments, and TR-1 says a table that sets them side by side is a
  bug."""


def world_of(condition):
  """The condition's world. A trace written before the field existed is the
  paper's Ville by construction -- every run before 010-IP was -- so it reads
  W1; a condition whose stages disagree reads MIXED."""
  if not condition.manifests:
    return "W1"
  world = condition.provenance("world")
  return "W1" if world is None else str(world)


def check_worlds(conditions):
  """Raise CrossWorldPair when the conditions do not share one world."""
  worlds = [(c.prefix, world_of(c)) for c in conditions]
  if any(w == "MIXED" for _, w in worlds):
    raise CrossWorldPair("a condition's stages ran in different worlds: %s"
                         % ", ".join("%s=%s" % pw for pw in worlds))
  if len(set(w for _, w in worlds)) > 1:
    raise CrossWorldPair("the conditions are in different worlds (%s): a "
                         "rate from one world is not compared with a rate "
                         "from another (TR-1, 010-IP LK-G)"
                         % ", ".join("%s=%s" % pw for pw in worlds))


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

  try:
    check_worlds(conditions)
  except CrossWorldPair as e:
    sys.exit("compare_backends: refused -- %s" % e)

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
  print('  column that means the backend actually failed.')
  print('  "rejected" is the responses the retry loop discarded on this '
        'decision; "already')
  print('  valid" is how many of them named an option the prompt had '
        'offered, and')
  print('  "verbatim" how many were one and nothing else. A high pair is '
        'the loop')
  print('  rejecting right answers in the wrong shape.')
  print('  "truncated" is how many of those the provider cut for the token '
        'budget:')
  print('  the model was answering and did not get to finish, which is not '
        'the same')
  print('  failure as answering something invalid. A trace written before '
        '006-IP')
  print('  recorded no finish_reason and reads "unknown" rather than 0.\n')
  if args.decision:
    rejected_responses(conditions, args.decision)
    answers(conditions, args.decision)


if __name__ == "__main__":
  main()
