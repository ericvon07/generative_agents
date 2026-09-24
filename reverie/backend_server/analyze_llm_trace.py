"""
File: analyze_llm_trace.py
Description: Aggregate the JSONL traces written by llm_trace.py into the
             baseline tables: where the LLM budget goes, per prompt, per
             cognitive phase, and split between the day bootstrap and a
             regular simulation step.

Usage (from reverie/backend_server, with the project venv):

  python analyze_llm_trace.py                     # newest trace file
  python analyze_llm_trace.py --all               # every trace file
  python analyze_llm_trace.py path/to/trace.jsonl # a specific file
  python analyze_llm_trace.py --persona           # add a per-agent table
  python analyze_llm_trace.py --rejects           # add format-failure detail
  python analyze_llm_trace.py --steps             # add a per-step table

When the trace contains decisions (see decision.py), two more tables appear on
their own: one per closed-set choice and one per rubric score, each split by
the backend that resolved it. Those are the tables the llm/jev comparison is
read out of -- point the script at both runs' traces with --all to see the two
conditions side by side.

Reads only; it never touches a simulation.
"""
import argparse
import collections
import glob
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "logs", "llm_trace")


# --- loading ----------------------------------------------------------------

def find_files(args):
  if args.files:
    return args.files
  pattern = os.path.join(args.dir, "*.jsonl")
  files = sorted(glob.glob(pattern), key=os.path.getmtime)
  if not files:
    sys.exit("no trace file found in %s" % os.path.abspath(args.dir))
  return files if args.all else [files[-1]]


def load(files):
  records = []
  for path in files:
    with open(path, "r", encoding="utf-8") as f:
      for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
          continue
        try:
          records.append(json.loads(line))
        except ValueError:
          # A run killed mid-write leaves one truncated line; skip it.
          print("warning: %s:%d is not valid json, skipped" % (path, line_no))
  return records


# --- stats helpers ----------------------------------------------------------

def percentile(values, q):
  """Nearest-rank percentile. Traces are small; no numpy dependency needed."""
  if not values:
    return None
  ordered = sorted(values)
  index = int(round((q / 100.0) * (len(ordered) - 1)))
  return ordered[index]


def median(values):
  return percentile(values, 50)


def fmt(value, width, digits=0):
  if value is None:
    return "-".rjust(width)
  if digits:
    return ("%.*f" % (digits, value)).rjust(width)
  return ("%.0f" % value).rjust(width)


def table(header, rows, widths):
  line = "  ".join(h.ljust(w) if i == 0 else h.rjust(w)
                   for i, (h, w) in enumerate(zip(header, widths)))
  print(line)
  print("-" * len(line))
  for row in rows:
    print("  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                    for i, (c, w) in enumerate(zip(row, widths))))


# --- aggregation ------------------------------------------------------------

class Bucket(object):
  def __init__(self):
    self.calls = 0
    self.errors = 0
    self.latencies = []
    self.in_tok = []
    self.out_tok = []
    self.outcomes = 0
    self.attempts = 0
    self.failsafes = 0
    self.rejects = collections.Counter()

  def add_call(self, r):
    self.calls += 1
    if not r.get("ok", True):
      self.errors += 1
    if r.get("latency_ms") is not None:
      self.latencies.append(r["latency_ms"])
    if r.get("in_tok") is not None:
      self.in_tok.append(r["in_tok"])
    if r.get("out_tok") is not None:
      self.out_tok.append(r["out_tok"])

  def add_outcome(self, r):
    self.outcomes += 1
    self.attempts += r.get("attempts") or 0
    if r.get("failsafe"):
      self.failsafes += 1

  def add_reject(self, r):
    self.rejects[r.get("reason") or "unknown"] += 1

  @property
  def seconds(self):
    return sum(self.latencies) / 1000.0


def bucket_by(records, key_fn):
  buckets = collections.defaultdict(Bucket)
  for r in records:
    key = key_fn(r)
    if key is None:
      continue
    event = r.get("event")
    if event == "call":
      buckets[key].add_call(r)
    elif event == "outcome":
      buckets[key].add_outcome(r)
    elif event == "reject":
      buckets[key].add_reject(r)
  return buckets


# --- reports ----------------------------------------------------------------

def report_overview(records, files):
  calls = [r for r in records if r.get("event") == "call"]
  outcomes = [r for r in records if r.get("event") == "outcome"]
  rejects = [r for r in records if r.get("event") == "reject"]
  llm_calls = [r for r in calls if r.get("path") != "embedding"]
  emb_calls = [r for r in calls if r.get("path") == "embedding"]
  seconds = sum(r.get("latency_ms") or 0 for r in calls) / 1000.0
  steps = sorted(set(r["step"] for r in records if r.get("step") is not None))
  sims = sorted(set(r.get("sim") for r in records if r.get("sim")))
  models = sorted(set(r.get("model") for r in calls if r.get("model")))

  print("=" * 78)
  print("LLM TRACE -- %s" % ", ".join(os.path.basename(f) for f in files))
  print("=" * 78)
  print("simulation(s) : %s" % (", ".join(sims) or "-"))
  print("model(s)      : %s" % (", ".join(models) or "-"))
  print("steps covered : %s" % ("%s..%s (%d)" % (steps[0], steps[-1], len(steps))
                                if steps else "-"))
  print("requests      : %d (%d completion/chat, %d embedding)"
        % (len(calls), len(llm_calls), len(emb_calls)))
  print("logical prompts: %d resolved, %d fell back to the fail-safe (%.1f%%)"
        % (len(outcomes),
           sum(1 for r in outcomes if r.get("failsafe")),
           100.0 * sum(1 for r in outcomes if r.get("failsafe")) / len(outcomes)
           if outcomes else 0.0))
  if outcomes:
    total_attempts = sum(r.get("attempts") or 0 for r in outcomes)
    print("attempts       : %d for %d prompts (%.2f per prompt, %d discarded "
          "responses)" % (total_attempts, len(outcomes),
                          float(total_attempts) / len(outcomes), len(rejects)))
  print("time in requests: %.1f s" % seconds)
  if steps:
    print("               : %.1f s per step covered" % (seconds / len(steps)))
  errors = sum(1 for r in calls if not r.get("ok", True))
  if errors:
    print("transport errors: %d (see the 'error' field in the trace)" % errors)
  print("")


def report_by_prompt(records, limit):
  buckets = bucket_by(records, lambda r: r.get("label"))
  rows = []
  for label, b in sorted(buckets.items(), key=lambda kv: -kv[1].seconds):
    att_per = (float(b.attempts) / b.outcomes) if b.outcomes else None
    fs_pct = (100.0 * b.failsafes / b.outcomes) if b.outcomes else None
    rows.append([
        label[:44],
        "%d" % b.calls,
        "%d" % b.outcomes,
        fmt(att_per, 7, 2),
        fmt(fs_pct, 5, 1),
        fmt(median(b.latencies), 7),
        fmt(percentile(b.latencies, 95), 7),
        fmt(median(b.in_tok), 6),
        fmt(median(b.out_tok), 6),
        "%.1f" % b.seconds,
    ])
  print("BY PROMPT (sorted by total time in requests)")
  table(["prompt", "calls", "prompts", "att/prm", "fs %", "p50 ms", "p95 ms",
         "in tok", "out tok", "total s"],
        rows[:limit], [44, 5, 7, 7, 5, 7, 7, 6, 7, 8])
  if len(rows) > limit:
    print("... %d more (use --top 0 for all)" % (len(rows) - limit))
  print("")


def report_by_phase(records):
  buckets = bucket_by(records, lambda r: r.get("phase") or "(outside move)")
  total = sum(b.seconds for b in buckets.values()) or 1.0
  rows = []
  for phase, b in sorted(buckets.items(), key=lambda kv: -kv[1].seconds):
    rows.append([
        phase,
        "%d" % b.calls,
        "%d" % b.outcomes,
        "%d" % b.failsafes,
        "%.1f" % b.seconds,
        "%.1f" % (100.0 * b.seconds / total),
    ])
  print("BY PHASE (bootstrap = the once-a-day long term planning)")
  table(["phase", "calls", "prompts", "failsafe", "total s", "% time"],
        rows, [16, 5, 7, 8, 8, 6])
  print("")


def report_by_path(records):
  buckets = bucket_by(records, lambda r: r.get("path"))
  rows = []
  for path, b in sorted(buckets.items(), key=lambda kv: -kv[1].seconds):
    rows.append([
        path,
        "%d" % b.calls,
        "%d" % b.errors,
        fmt(median(b.latencies), 7),
        fmt(percentile(b.latencies, 95), 7),
        "%d" % sum(b.in_tok),
        "%d" % sum(b.out_tok),
        "%.1f" % b.seconds,
    ])
  print("BY CODE PATH (token columns are totals, not medians)")
  table(["path", "calls", "errors", "p50 ms", "p95 ms", "in tok", "out tok",
         "total s"],
        rows, [26, 5, 6, 7, 7, 8, 8, 8])
  print("")


def report_rejects(records):
  by_label = collections.defaultdict(collections.Counter)
  overall = collections.Counter()
  for r in records:
    if r.get("event") != "reject":
      continue
    reason = r.get("reason") or "unknown"
    by_label[r.get("label")][reason] += 1
    overall[reason] += 1
  if not overall:
    print("REJECTED RESPONSES: none -- every call passed validation.\n")
    return
  print("REJECTED RESPONSES (a response arrived and was thrown away)")
  total = sum(overall.values())
  for reason, count in overall.most_common():
    print("  %-34s %5d  (%4.1f%%)" % (reason, count, 100.0 * count / total))
  print("")
  rows = []
  for label, counter in sorted(by_label.items(), key=lambda kv: -sum(kv[1].values())):
    rows.append([
        (label or "unknown")[:44],
        "%d" % sum(counter.values()),
        ", ".join("%s=%d" % (k, v) for k, v in counter.most_common(3)),
    ])
  table(["prompt", "rejects", "reasons"], rows, [44, 7, 40])
  print("")
  print("Sample of the discarded payloads:")
  shown = 0
  for r in records:
    if r.get("event") == "reject" and shown < 5:
      print("  [%s / %s] %s" % (r.get("label"), r.get("reason"),
                                (r.get("resp_head") or "").replace("\n", "\\n")))
      shown += 1
  print("")


# --- decisions --------------------------------------------------------------
# These two tables are the comparison the project exists to make: the same
# decision, resolved by two backends, counted the same way. Everything here
# comes from "decision" events, which the LLM path and the typed path both
# write (llm_trace.record_decision).

JEV_DEFAULT_INPUT_PRICE = 0.042  # $ per 1M input tokens, jev-1.12, Sep 2026


def _jev_input_price():
  try:
    with open(os.path.join(SCRIPT_DIR, "..", "..", "openai_config.json")) as f:
      costs = (json.load(f).get("jev") or {}).get("costs") or {}
    return float(costs.get("input", JEV_DEFAULT_INPUT_PRICE))
  except Exception:
    return JEV_DEFAULT_INPUT_PRICE


def _gave_up(record):
  """Whether a decision actually fell back, as opposed to landing on a value
  that happens to equal the fail-safe. See llm_trace.record_decision."""
  recorded = record.get("gave_up")
  if recorded is not None:
    return bool(recorded)
  return (bool(record.get("backend_error"))
          or str(record.get("answer")) == "False")


def _share(count, total):
  if not total:
    return "-"
  return "%.1f%%" % (100.0 * count / total)


def report_decisions(records):
  """One row per (decision, backend). Out-of-domain is the number the paper's
  code cannot produce: its validation accepts any non-empty string."""
  decisions = [r for r in records if r.get("event") == "decision"]
  if not decisions:
    return

  choices = collections.defaultdict(list)
  scores = collections.defaultdict(list)
  for r in decisions:
    key = (r.get("label"), r.get("backend") or "-")
    if r.get("question_type") == "score" or r.get("raw_score") is not None:
      scores[key].append(r)
    elif r.get("domain_size"):
      choices[key].append(r)
    else:
      scores[key].append(r)

  if choices:
    rows = []
    for (label, backend), group in sorted(choices.items(),
                                          key=lambda kv: -len(kv[1])):
      checked = [r for r in group if r.get("in_domain") is not None]
      outside = [r for r in checked if r["in_domain"] is False]
      failsafe = [r for r in group if r.get("is_fail_safe")]
      broken = [r for r in group if _gave_up(r)]
      confidences = [r["confidence"] for r in group
                     if r.get("confidence") is not None]
      answers = collections.Counter(str(r.get("answer")) for r in group)
      top, top_n = answers.most_common(1)[0]
      rows.append([
          label,
          backend,
          "%d" % len(group),
          "%d" % (group[0].get("domain_size") or 0),
          _share(len(outside), len(checked)),
          _share(len(failsafe), len(group)),
          _share(len(broken), len(group)),
          fmt(median(confidences), 8, 3) if confidences else "-".rjust(8),
          "%s (%s)" % (top[:22], _share(top_n, len(group))),
      ])
    print("DECISIONS -- CLOSED-SET CHOICES")
    table(["decision", "backend", "n", "opts", "out of dom", "= failsafe",
           "gave up", "p50 conf", "most common answer"],
          rows, [20, 8, 5, 5, 10, 10, 7, 8, 30])
    print('  "= failsafe" is the answer matching the fail-safe VALUE; '
          '"gave up" is the')
    print("  decision actually falling back. They differ wherever the "
          "fail-safe is also")
    print("  a legal answer -- action_sector falls back to \"kitchen\".")
    print("")

  if scores:
    rows = []
    for (label, backend), group in sorted(scores.items(),
                                          key=lambda kv: -len(kv[1])):
      values = []
      for r in group:
        try:
          values.append(float(r.get("answer")))
        except (TypeError, ValueError):
          pass
      raws = [r["raw_score"] for r in group if r.get("raw_score") is not None]
      confidences = [r["confidence"] for r in group
                     if r.get("confidence") is not None]
      failsafe = [r for r in group if _gave_up(r)]
      rows.append([
          label,
          backend,
          "%d" % len(group),
          fmt(sum(values) / len(values) if values else None, 6, 2),
          fmt(median(values), 6, 1) if values else "-".rjust(6),
          fmt(median(raws), 7, 3) if raws else "-".rjust(7),
          fmt(median(confidences), 8, 3) if confidences else "-".rjust(8),
          _share(len(failsafe), len(group)),
      ])
    print("DECISIONS -- RUBRIC SCORES")
    table(["decision", "backend", "n", "mean", "p50", "p50 raw", "p50 conf",
           "gave up"],
          rows, [20, 8, 5, 6, 6, 7, 8, 9])
    print("  p50 raw is the float the typed backend returned before it was")
    print("  rounded to the integer the simulation stores; blank for the LLM")
    print("  path, which only ever produced the integer.")
    print("")

  jev_calls = [r for r in records
               if r.get("event") == "call" and r.get("path") == "jev"]
  if jev_calls:
    tokens = sum(r.get("in_tok") or 0 for r in jev_calls)
    price = _jev_input_price()
    errors = sum(1 for r in jev_calls if not r.get("ok", True))
    print("jev: %d requests, %d input tokens, $%.4f at $%.3f/M (%d transport "
          "errors)" % (len(jev_calls), tokens, tokens / 1e6 * price, price,
                       errors))
    print("")


def report_by_persona(records):
  buckets = bucket_by(records, lambda r: r.get("persona"))
  rows = []
  for persona, b in sorted(buckets.items(), key=lambda kv: -kv[1].seconds):
    rows.append([
        persona,
        "%d" % b.calls,
        "%d" % b.outcomes,
        "%d" % b.failsafes,
        "%.1f" % b.seconds,
    ])
  print("BY AGENT")
  table(["agent", "calls", "prompts", "failsafe", "total s"],
        rows, [26, 5, 7, 8, 8])
  print("")


def report_by_step(records):
  buckets = bucket_by(records, lambda r: r.get("step"))
  rows = []
  for step, b in sorted(buckets.items(), key=lambda kv: kv[0]):
    rows.append([
        "%s" % step,
        "%d" % b.calls,
        "%d" % b.outcomes,
        "%d" % b.failsafes,
        "%.1f" % b.seconds,
    ])
  print("BY STEP")
  table(["step", "calls", "prompts", "failsafe", "total s"],
        rows, [8, 5, 7, 8, 8])
  print("")


# --- the social decisions ---------------------------------------------------
# Everything here reads fields the trace has carried since 008-IP SV-B --
# `subject`, `other` and `episode` on a social decision -- and which no reader
# knew about. Read by hand until now, which is how the n9win readings of
# 2026-09-23 were made.
#
# Two traps this report exists to keep a reader out of:
#
#   is_fail_safe is not a failure rate on decide_to_talk. Its fail-safe value
#   is "yes", so EVERY genuine "yes" reads is_fail_safe: true. `gave_up` is
#   the field that says a decision actually fell back, and the header below
#   prints both so they can never be confused again.
#
#   the ask is not the episode. The same question is put again every step two
#   agents can see each other, so an agent who stands still all afternoon
#   collects asks without collecting encounters -- Isabella Rodriguez ran at
#   4.3 asks per episode in n9win while Carmen Ortiz ran at 1.0. UN-1 in the
#   lab is the decision this serves: report the distribution, and where a rate
#   is shown at all show both units.

SOCIAL_LABELS = ("decide_to_talk", "decide_to_react")


def _p_yes(record):
  """The probability the Choice put on the affirmative option, or None.

  "yes" on decide_to_talk, "1" on decide_to_react -- the option that means
  the agent engages rather than walks past. A Noul answer has no option, so
  its recorded probability is read instead.
  """
  probabilities = record.get("probabilities")
  if isinstance(probabilities, dict):
    for option in ("yes", "1"):
      if option in probabilities:
        return probabilities[option]
  return record.get("noul")


def _social(records, label):
  out = []
  for r in records:
    if r.get("event") != "decision" or r.get("label") != label:
      continue
    if r.get("subject") and r.get("other") and _p_yes(r) is not None:
      out.append(r)
  out.sort(key=lambda r: (r.get("step") or 0))
  return out


def _episodes(rows):
  buckets = {}
  for r in rows:
    buckets.setdefault(r.get("episode"), []).append(r)
  return buckets


def report_social(records, tail=0.30):
  # A trace directory holds every run the machine ever made, and --all reads
  # all of them. Adding two runs together makes a rate out of two populations
  # and law 3 of the lab forbids it, so say so before printing a number that
  # looks like one run's.
  sims = sorted(set(str(r.get("sim") or "").split("-s-")[0]
                    for r in records if r.get("sim")))
  if len(sims) > 1:
    print("!! THESE RECORDS SPAN %d RUNS: %s" % (len(sims), ", ".join(sims)))
    print("!! Every figure below adds them together. Pass the one run's files "
          "explicitly, or --dir a directory that holds only it.")
    print("")

  for label in SOCIAL_LABELS:
    rows = _social(records, label)
    if not rows:
      continue
    values = [_p_yes(r) for r in rows]
    eps = _episodes(rows)
    engaged = [r for r in rows if _p_yes(r) > 0.5]
    gave_up = sum(1 for r in rows if r.get("gave_up"))
    equals_fs = sum(1 for r in rows if r.get("is_fail_safe"))

    print("SOCIAL: %s" % label)
    print("  %d asks in %d episodes (%.1f per episode); p median %.3f, max %.3f"
          % (len(rows), len(eps), float(len(rows)) / len(eps),
             median(values), max(values)))
    print("  above 0.50: %d of %d asks (%.1f%%), %d of %d episodes (%.1f%%)"
          % (len(engaged), len(rows), 100.0 * len(engaged) / len(rows),
             len(set(r.get("episode") for r in engaged)), len(eps),
             100.0 * len(set(r.get("episode") for r in engaged)) / len(eps)))
    print("  gave_up: %d  <- the failure rate. "
          "is_fail_safe: %d  <- NOT a failure rate: \"yes\" IS the fail-safe "
          "value, so every genuine yes reads true." % (gave_up, equals_fs))
    print("")

    # --- by subject: who is asked, how often, and how concentrated ---------
    by_subject = {}
    for r in rows:
      by_subject.setdefault(r["subject"], []).append(r)
    subject_rows = []
    for name, group in sorted(by_subject.items(), key=lambda kv: -len(kv[1])):
      vals = [_p_yes(r) for r in group]
      n_eps = len(set(r.get("episode") for r in group))
      subject_rows.append([
          name, "%d" % len(group), "%d" % n_eps,
          "%.1f" % (float(len(group)) / n_eps),
          "%.3f" % median(vals), "%.3f" % max(vals),
          "%d" % sum(1 for v in vals if v >= tail),
          "%d" % sum(1 for v in vals if v > 0.5),
      ])
    print("  BY SUBJECT")
    table(["subject", "asks", "episodes", "asks/ep", "median", "max",
           ">=%.2f" % tail, ">0.50"],
          subject_rows, [22, 5, 9, 8, 7, 7, 7, 6])
    print("")

    # --- the reciprocal gap -----------------------------------------------
    # The strongest thing this report can say. A->B and B->A asked at the
    # SAME step is the same encounter, the same place, the same second: what
    # differs is only which persona the question is about, and which
    # persona's own retrieved memories are in `context`. A wide gap there is
    # the backend discriminating a DIRECTION rather than scoring a name --
    # and it is the reading that says whether it reads the state at all.
    at_step = {}
    for r in rows:
      at_step.setdefault(r.get("step"), {})[(r["subject"], r["other"])] = r
    seen, recip = set(), []
    for step, pairs in at_step.items():
      for (a, b), r in pairs.items():
        mirror = pairs.get((b, a))
        if not mirror:
          continue
        key = (step, tuple(sorted((a, b))))
        if key in seen:
          continue
        seen.add(key)
        recip.append((step, r, mirror))
    if recip:
      gaps = [abs(_p_yes(r) - _p_yes(m)) for _, r, m in recip]
      print("  RECIPROCAL ASKS AT THE SAME STEP: %d pairs, "
            "|p(A->B) - p(B->A)| median %.3f, max %.3f"
            % (len(recip), median(gaps), max(gaps)))
      widest = sorted(recip, key=lambda t: -abs(_p_yes(t[1]) - _p_yes(t[2])))
      gap_rows = []
      for step, r, m in widest[:8]:
        gap_rows.append([
            "%s" % step,
            "%s -> %s" % (r["subject"], r["other"]), "%.3f" % _p_yes(r),
            "%.3f" % _p_yes(m),
            "%.3f" % abs(_p_yes(r) - _p_yes(m)),
        ])
      table(["step", "direction", "p", "p back", "gap"],
            gap_rows, [8, 40, 7, 8, 7])
      print("")

    # --- within an episode -------------------------------------------------
    multi = [v for v in eps.values() if len(v) > 1]
    if multi:
      spreads = [max(_p_yes(x) for x in v) - min(_p_yes(x) for x in v)
                 for v in multi]
      frozen = sum(1 for s in spreads if s < 1e-9)
      print("  WITHIN AN EPISODE: %d episodes hold more than one ask; spread "
            "median %.3f, max %.3f; %d never moved at all"
            % (len(multi), median(spreads), max(spreads), frozen))
      widest = max(multi, key=lambda v: max(_p_yes(x) for x in v)
                   - min(_p_yes(x) for x in v))
      print("    widest: %s (n=%d) %s"
            % (widest[0].get("episode"), len(widest),
               " ".join("%.3f" % _p_yes(x) for x in widest)))
      print("")

    # --- who was ever in a room with whom ----------------------------------
    # An ask exists only where two agents could see each other, so this IS
    # the encounter graph. It belongs beside any diffusion number: an agent
    # who never meets five of the other eight cannot pass anything to them,
    # and a reach figure computed without it measures the map.
    met = {}
    pair_asks = {}
    for r in rows:
      a, b = r["subject"], r["other"]
      met.setdefault(a, set()).add(b)
      met.setdefault(b, set()).add(a)
      key = tuple(sorted((a, b)))
      pair_asks[key] = pair_asks.get(key, 0) + 1
    names = sorted(met)
    print("  THE ENCOUNTER GRAPH (%d agents ever asked about)" % len(names))
    reach_rows = []
    for name in sorted(names, key=lambda n: -len(met[n])):
      missing = [m for m in names if m != name and m not in met[name]]
      reach_rows.append([
          name, "%d of %d" % (len(met[name]), len(names) - 1),
          ", ".join(missing) if missing else "-",
      ])
    table(["agent", "met", "never in a room with"],
          reach_rows, [22, 8, 62])
    print("")


def main():
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("files", nargs="*", help="trace files (default: newest)")
  parser.add_argument("--dir", default=DEFAULT_DIR, help="trace directory")
  parser.add_argument("--all", action="store_true",
                      help="read every trace file in the directory")
  parser.add_argument("--top", type=int, default=25,
                      help="prompts to show in the per-prompt table (0 = all)")
  parser.add_argument("--phase", help="keep only one phase (e.g. bootstrap)")
  parser.add_argument("--rejects", action="store_true",
                      help="detail the responses that failed validation")
  parser.add_argument("--persona", action="store_true", help="per-agent table")
  parser.add_argument("--steps", action="store_true", help="per-step table")
  parser.add_argument("--social", action="store_true",
                      help="the social decisions: episodes, reciprocal "
                           "gaps and the encounter graph")
  args = parser.parse_args()

  files = find_files(args)
  records = load(files)
  if not records:
    sys.exit("trace is empty: %s" % ", ".join(files))
  if args.phase:
    records = [r for r in records if r.get("phase") == args.phase]
    if not records:
      sys.exit("no record with phase=%s" % args.phase)

  report_overview(records, files)
  report_by_phase(records)
  report_by_path(records)
  report_by_prompt(records, args.top if args.top > 0 else len(records))
  report_decisions(records)
  if args.rejects:
    report_rejects(records)
  if args.persona:
    report_by_persona(records)
  if args.steps:
    report_by_step(records)
  if args.social:
    report_social(records)


if __name__ == "__main__":
  main()
