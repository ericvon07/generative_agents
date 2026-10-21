# Runs

The two runs behind the Smallville results in version 20 of the System One essay (October
2026): the Valentine's Day party on the village of nine, over the paper's two game days, once
with the paper's prompts deciding (`vp9-llm`) and once with Jev deciding (`vp9-jev`). Gemma 4 31B
wrote the plans, reflections and dialogue in both. Both used `openai_config.example.vp9.json`;
the decision backend was set by `DECISION_BACKEND`.

Each run folder holds:

- `manifest.json`: the condition, the code, the seed, the window, the providers that answered,
  and every stage with its trace and the trace's sha1. `code.sha` is the commit the run was
  made on in the private repository where this fork was developed, and the one the traces
  record; `code.public_sha` is the same code in this repository. `code.tree`, the tree of
  `reverie/backend_server`, is the same in both.
- `trace/`: the JSONL trace of every model call and decision, one file per stage, gzipped.
- `interviews/`: the paper's end-of-run interview (`-facts`, the paper's two questions;
  `-relations`, `Do you know of <name>?`) at the last checkpoint, at the first one
  (`-s-0-0-200`) and at the base town, with each answer's label.
- `checkpoints.tar.gz`: the part of the run's checkpoints the readers use: every stage's
  `reverie/meta.json` and each character's `scratch.json`, the memory nodes of the first and
  last stages, and the last stage's `movement/`, which holds every step of the run. The full
  checkpoints, about 11 GB per run, are not published.

## Recomputing the essay's numbers

From the repository root, in the Python environment the main README sets up:

```sh
runs/unpack.sh                 # traces to logs/llm_trace/, interviews to logs/interviews/,
                               # checkpoints to environment/frontend_server/storage/
cd reverie/backend_server
python read_valentine.py vp9-llm --start base_the_ville_n9
python read_valentine.py vp9-jev --start base_the_ville_n9
python read_vp9_window.py
```

- **Who knew about the party, and who showed up** (7 and 8 of 9; 4 and 6 of the 7 who heard
  before it): section 1 of `read_valentine.py`.
- **Conversations, reflection cycles, decisions, cost and time** (44 and 50 conversations;
  98 and 172 reflection cycles; 17,099 and 20,040 decisions; $1.82 and $1.48 in total;
  $0.0634 and $0.0287 per thousand decisions): `read_vp9_window.py`, in the column of the
  steps before 17,280, the paper's two days. It reads the traces under `runs/` directly and
  refuses one whose sha1 is not the manifest's.
- **Object choices outside the offered list** (43 of 1,565 in `vp9-llm`):
  `read_vp9_window.py`, the line `objects out of the offered set`.

Each approach ran once, so another run could change these numbers. The two runs stop visiting
the same states at their first differing decision, so each is read on its own, side by side,
never paired.
