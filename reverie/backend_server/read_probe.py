#!/usr/bin/env python3
"""
File: read_probe.py
Description: 011-IP step 4's instrument -- what the probe's rows say, and
             which form TS-A's criterion picks.

`probe_questions.py` asks one state in every form that applies and writes one
JSONL row per answer. This reads those rows. It decides nothing on its own:
TS-A fixed the criterion BEFORE the data, and this applies it and shows its
working, including when the answer is "no form beat the dictionary" -- which
is a result and not a failure, and which fires SC-D's reopening clause in
007-ST.

THE CRITERION, as TS-A wrote it:

  a form is ADEQUATE if it puts the anchors on the right side;
  among adequate forms the one with the WIDER SPREAD across the probe run's
  real states wins;
  a tie within 0.05 of spread goes to the CHEAPER mechanism.

Cheapness is an order this file hard-codes because TS-A named it: the
dictionary costs nothing, descriptive criteria cost two sentences, the
composed state replaces the payload. Spread is reported two ways -- the
literal range TS-A's words ask for, and the 10th-to-90th percentile band,
which one outlier cannot move. The criterion runs on the range by default;
`--spread p10p90` runs it on the robust one, and the readout says which.

FOUR READINGS

  anchors      per question per form, did each hand-written state land on the
               side a reader does not doubt?
  spread       per question per form, over the real states only.
  reciprocal   the one reading that can falsify something. A->B and B->A
               asked at the SAME step is one encounter: same place, same
               second, and only the direction differs. n9win opened gaps up
               to 0.720 there. If a form COLLAPSES that gap, the asymmetry
               was living in `context` -- a key the instruction already names
               -- and the orphan keys were never the cause.
  four-way     decide_to_talk as a Noul and as a Choice, on the dictionary
               and on the composed state. A diagnostic, not a decision: it
               says whether the facts move the two TYPES differently, which
               is the half of D5's evidence a window cannot give.

USAGE
  cd reverie/backend_server
  ../../../.venv/bin/python read_probe.py probe_questions.jsonl
"""
import sys
import json
import argparse
import collections

# TS-A's own order, cheapest first. A tie goes left. "structured" (2026-09-24,
# structured_state.py) is last: it replaces the payload as "composed" does and
# also needs its own builder at every call site.
FORM_COST = ("dictionary", "descriptive", "composed", "structured",
             "sentences", "question-out")
TIE = 0.05


def load(path):
  rows = []
  with open(path) as handle:
    for line in handle:
      line = line.strip()
      if line:
        try:
          rows.append(json.loads(line))
        except ValueError:
          pass
  return [r for r in rows if not r.get("dry_run")]


def median(values):
  values = sorted(values)
  if not values:
    return float("nan")
  mid = len(values) // 2
  if len(values) % 2:
    return values[mid]
  return (values[mid - 1] + values[mid]) / 2.0


def percentile(values, q):
  values = sorted(values)
  if not values:
    return float("nan")
  if len(values) == 1:
    return values[0]
  pos = (len(values) - 1) * q / 100.0
  low = int(pos)
  high = min(low + 1, len(values) - 1)
  return values[low] + (values[high] - values[low]) * (pos - low)


def table(header, rows, widths):
  line = " ".join(h.rjust(w) if i else h.ljust(w)
                  for i, (h, w) in enumerate(zip(header, widths)))
  print(line)
  print("-" * len(line))
  for row in rows:
    print(" ".join(str(c).rjust(w) if i else str(c).ljust(w)
                   for i, (c, w) in enumerate(zip(row, widths))))


def _p(row):
  """The scalar this row is compared on.

  A BINARY question has p(yes) -- the probability on the option that means
  the agent engages -- and that is the number every reading here is about.
  A question over seven sectors or three arenas has no such option, and
  p_yes comes back None: the scalar there is the CONFIDENCE, the probability
  the backend put on whichever option it picked. The two are not the same
  quantity and this file never mixes them in one table; what they share is
  the property TS-A's criterion actually tests, which is whether a form
  DISCRIMINATES. A form that answers every state at 1.000 confidence has no
  spread and is telling the reader nothing, exactly as a form that puts
  every p(yes) at 0.30 is.
  """
  value = row.get("p_yes")
  if value is not None:
    return value
  return row.get("confidence")


def _scale_of(label):
  """What _p returns for <label>, for the header that says so."""
  return "p(yes)" if label in ("decide_to_talk", "decide_to_react") \
      else "confidence"


def _choice_rows(rows, label=None):
  out = [r for r in rows
         if r.get("question_type") == "choice" and not r.get("error")]
  if label:
    out = [r for r in out if r.get("label") == label]
  return out


# --- 1. the anchors ---------------------------------------------------------

def report_anchors(rows):
  print("=" * 78)
  print("ANCHORS -- states whose right answer a reader does not doubt.")
  print("A form that cannot separate them is not measuring anything (TS-A).")
  print("")
  anchors = [r for r in _choice_rows(rows) if r.get("expect")]
  if not anchors:
    print("  no hand-written state carried an expected answer.")
    return {}
  verdict = collections.defaultdict(dict)
  by_case = collections.defaultdict(dict)
  for r in anchors:
    by_case[(r["label"], r["case"], r["expect"], bool(r.get("anchor")))][r["form"]] = r
  forms = [f for f in FORM_COST if any(f in v for v in by_case.values())]
  body = []
  for (label, case, expect, is_anchor), per_form in sorted(by_case.items()):
    cells = []
    for f in forms:
      r = per_form.get(f)
      if not r:
        cells.append("-")
        continue
      ok = str(r.get("answer")) == str(expect)
      p = _p(r)
      cells.append("%s %s" % ("OK " if ok else "MISS",
                              "    " if p is None else "%.3f" % p))
      verdict[(label, f)][case] = ok
    body.append(["%s%s" % ("* " if is_anchor else "  ", case), label,
                 str(expect)] + cells)
  table(["case (* = anchor)", "question", "want"] + list(forms), body,
        [24, 16, 12] + [12] * len(forms))
  print("")
  print("  * the two anchors are the ones TS-A named; the rest are the four")
  print("    other hand-written states. ADEQUACY below counts the anchors.")
  print("")
  adequate = {}
  for (label, form), cases in verdict.items():
    anchor_cases = [c for c in cases
                    if any(k[1] == c and k[3] for k in by_case)]
    checked = anchor_cases or list(cases)
    adequate[(label, form)] = all(cases[c] for c in checked)
  return adequate


# --- 2. the spread over real states ------------------------------------------

def report_spread(rows, adequate, spread_kind):
  print("=" * 78)
  print("SPREAD over the probe run's REAL states (hand-written ones excluded).")
  print("")
  real = [r for r in _choice_rows(rows) if r.get("source") == "trace"]
  chosen = {}
  by_label = collections.defaultdict(lambda: collections.defaultdict(list))
  for r in real:
    if _p(r) is not None:
      by_label[r["label"]][r["form"]].append(_p(r))
  for label in sorted(by_label):
    body = []
    scores = {}
    for form in FORM_COST:
      vals = by_label[label].get(form)
      if not vals:
        continue
      rng = max(vals) - min(vals)
      band = percentile(vals, 90) - percentile(vals, 10)
      scores[form] = rng if spread_kind == "range" else band
      ok = adequate.get((label, form))
      body.append([form, "%d" % len(vals), "%.3f" % median(vals),
                   "%.3f" % min(vals), "%.3f" % max(vals), "%.3f" % rng,
                   "%.3f" % band,
                   "%d" % sum(1 for v in vals
                                if v > (0.5 if _scale_of(label) == "p(yes)"
                                        else 0.99)),
                   "yes" if ok else ("no" if ok is not None else "?")])
    if not body:
      continue
    print("  %s  --  the column below is %s"
          % (label, _scale_of(label)))
    last = ">0.50" if _scale_of(label) == "p(yes)" else ">0.99"
    table(["form", "n", "median", "min", "max", "range", "p10-90", last,
           "adequate"], body, [14, 5, 7, 7, 7, 7, 8, 6, 9])
    eligible = {f: s for f, s in scores.items() if adequate.get((label, f))}
    base = scores.get("dictionary")
    if not eligible:
      print("    -> NO FORM IS ADEQUATE on this question: the anchors were")
      print("       missed by every form, so spread decides nothing here.")
    else:
      best = max(eligible.values())
      near = [f for f in FORM_COST
              if f in eligible and best - eligible[f] <= TIE]
      pick = near[0]
      print("    -> criterion picks: %s (spread %.3f; ties within %.2f went "
            "to the cheaper form)" % (pick, eligible[pick], TIE))
      if base is not None and pick != "dictionary" and eligible[pick] <= base:
        print("       BUT it does not beat the dictionary (%.3f): D7 is still"
              % base)
        print("       entered, and the readout says the orphan keys were not")
        print("       the cause -- 007-ST's SC-D reopening clause fires.")
      chosen[label] = pick
    print("")
  return chosen


# --- 3. the reciprocal gap ---------------------------------------------------

def report_reciprocal(rows):
  print("=" * 78)
  print("RECIPROCAL ASKS -- A->B and B->A at the SAME step: one encounter,")
  print("one place, one second, and only the direction differs.")
  print("")
  real = [r for r in _choice_rows(rows)
          if r.get("step") is not None and r.get("subject") and r.get("other")]
  index = collections.defaultdict(dict)
  for r in real:
    index[(r["label"], r["form"], r["step"])][(r["subject"], r["other"])] = r
  gaps = collections.defaultdict(list)
  examples = collections.defaultdict(list)
  for (label, form, step), pairs in index.items():
    seen = set()
    for (a, b), r in pairs.items():
      mirror = pairs.get((b, a))
      if not mirror:
        continue
      key = tuple(sorted((a, b)))
      if key in seen:
        continue
      seen.add(key)
      if _p(r) is None or _p(mirror) is None:
        continue
      gap = abs(_p(r) - _p(mirror))
      gaps[(label, form)].append(gap)
      examples[(label, form)].append((gap, step, a, b, _p(r), _p(mirror)))
  if not gaps:
    print("  the probe rows hold no reciprocal pair. Either the probe run")
    print("  produced none, or the rows predate subject/other being written.")
    return
  for label in sorted(set(k[0] for k in gaps)):
    body = []
    for form in FORM_COST:
      vals = gaps.get((label, form))
      if not vals:
        continue
      body.append([form, "%d" % len(vals), "%.3f" % median(vals),
                   "%.3f" % max(vals),
                   "%d" % sum(1 for v in vals if v >= 0.30)])
    if not body:
      continue
    print("  %s" % label)
    table(["form", "pairs", "median gap", "max gap", ">=0.30"], body,
          [14, 6, 11, 9, 7])
    dict_med = median(gaps[(label, "dictionary")]) if gaps.get((label, "dictionary")) else None
    for form in FORM_COST:
      if form == "dictionary" or not gaps.get((label, form)):
        continue
      med = median(gaps[(label, form)])
      if dict_med is None:
        continue
      if med < dict_med * 0.5:
        print("    -> %s COLLAPSES the gap the dictionary opens (%.3f -> "
              "%.3f)." % (form, dict_med, med))
        print("       The asymmetry was not in the orphan keys. It was in")
        print("       `context`, which the instruction already names.")
      elif med > dict_med * 1.5:
        print("    -> %s WIDENS the gap (%.3f -> %.3f): naming the facts"
              % (form, dict_med, med))
        print("       makes the two directions further apart, not closer.")
    top = sorted(examples[(label, "dictionary")], reverse=True)[:5] \
        if examples.get((label, "dictionary")) else []
    if top:
      print("    widest, on the dictionary form:")
      for gap, step, a, b, pa, pb in top:
        print("      step %-6s %.3f  %s -> %s   vs %.3f back   gap %.3f"
              % (step, pa, a, b, pb, gap))
    print("")


# --- 4. the four-way ask ------------------------------------------------------

def report_four_way(rows):
  print("=" * 78)
  print("THE FOUR-WAY ASK on decide_to_talk -- Noul and Choice, on the")
  print("dictionary state and on the composed state. A diagnostic: does")
  print("naming the facts move the two TYPES differently?")
  print("")
  four = [r for r in rows
          if r.get("label") == "decide_to_talk" and not r.get("error")
          and r.get("form") in ("dictionary", "composed")]
  by_case = collections.defaultdict(dict)
  for r in four:
    by_case[(r.get("case"), bool(r.get("anchor")), r.get("expect"))][
        (r["form"], r.get("question_type"))] = r
  cols = [("dictionary", "choice"), ("composed", "choice"),
          ("dictionary", "noul"), ("composed", "noul")]
  body = []
  for (case, is_anchor, expect), cells in sorted(by_case.items()):
    if not is_anchor and expect is None:
      continue
    row = ["%s%s" % ("* " if is_anchor else "  ", case), str(expect)]
    for key in cols:
      r = cells.get(key)
      row.append("    " if not r or _p(r) is None else "%.3f" % _p(r))
    body.append(row)
  if not body:
    print("  no hand-written decide_to_talk case carried all four asks.")
  else:
    table(["case", "want", "dict/choice", "comp/choice", "dict/noul",
           "comp/noul"], body, [24, 6, 11, 11, 10, 10])
    print("")
    print("  A Choice that rises where a reader would say yes is the half of")
    print("  D5's evidence a window cannot give (TS-C).")
    print("")
    # SEPARATION: the distance between the anchor a reader calls yes and the
    # anchor a reader calls no, per type per state form. It is the number D5
    # is actually about -- not whether an answer is high, but whether the two
    # situations land apart. A Noul is scored by its probability against the
    # 0.50 cut, because a Noul has no option to return.
    want = {}
    for (case, is_anchor, expect), cells in by_case.items():
      if is_anchor and expect in ("yes", "no"):
        want[case] = expect
    yes_cases = [c for c, w in want.items() if w == "yes"]
    no_cases = [c for c, w in want.items() if w == "no"]
    if yes_cases and no_cases:
      sep = []
      for form, qtype in cols:
        highs, lows, right = [], [], True
        for (case, is_anchor, expect), cells in by_case.items():
          r = cells.get((form, qtype))
          if not is_anchor or not r or _p(r) is None:
            continue
          (highs if expect == "yes" else lows).append(_p(r))
          # a Choice is right when it RETURNS the option; a Noul when its
          # probability falls on the right side of the cut.
          if qtype == "choice":
            right = right and str(r.get("answer")) == str(expect)
          else:
            right = right and ((_p(r) > 0.5) == (expect == "yes"))
        if highs and lows:
          sep.append([("%s / %s" % (qtype, form)),
                      "%.3f" % min(lows), "%.3f" % max(highs),
                      "%.3f" % (max(highs) - min(lows)),
                      "yes" if right else "NO"])
      if sep:
        print("  SEPARATION between the two anchors -- the distance D5 is")
        print("  actually about: not whether an answer is high, but whether")
        print("  the situation a reader calls no and the one a reader calls")
        print("  yes land APART.")
        table(["type / state", "the 'no'", "the 'yes'", "apart",
               "both right"], sep, [24, 9, 10, 7, 11])
  print("")
  # and the real states a reader would call a crossing
  crossings = [r for r in _choice_rows(rows, "decide_to_talk")
               if r.get("source") == "trace" and (_p(r) or 0) > 0.5]
  if crossings:
    print("  REAL states that crossed 0.50, by form:")
    seen = collections.Counter((r["form"]) for r in crossings)
    for form in FORM_COST:
      if seen.get(form):
        print("    %-14s %d" % (form, seen[form]))
  print("")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
  parser.add_argument("rows", help="the probe's JSONL")
  parser.add_argument("--spread", choices=("range", "p10p90"),
                      default="range",
                      help="which spread the criterion runs on "
                           "(default: range, which is TS-A's literal words)")
  args = parser.parse_args(argv)

  rows = load(args.rows)
  errors = [r for r in rows if r.get("error")]
  print("read %d answers from %s%s"
        % (len(rows), args.rows,
           " (%d carried an error and are excluded)" % len(errors)
           if errors else ""))
  if errors:
    for r in errors[:5]:
      print("   error: %s %s %s -- %s"
            % (r.get("label"), r.get("form"), r.get("case"), r["error"]))
  print("")
  adequate = report_anchors(rows)
  chosen = report_spread(rows, adequate, args.spread)
  report_reciprocal(rows)
  report_four_way(rows)
  print("=" * 78)
  print("WHAT THE CRITERION PICKED, per question (spread on: %s)" % args.spread)
  if not chosen:
    print("  nothing: no form was adequate on any question.")
  for label in sorted(chosen):
    print("  %-18s %s" % (label, chosen[label]))
  print("")
  print("This file applies TS-A's criterion; it does not decide D7. Step 4")
  print("is a READING GATE: the author reads this beside n9win's readout")
  print("before step 5 moves anything onto the run path.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
