# Generative Agents, with a decision model for the quick decisions

**This is a fork of [drudilorenzo/generative_agents](https://github.com/drudilorenzo/generative_agents)**
(branch `fix-and-improve`), a maintained version of
[joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents), the
code published with Park et al., *Generative Agents: Interactive Simulacra of Human Behavior*
(UIST 2023, [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)). The earlier README follows,
unchanged, below the line.

The fork asks what changes when the simulation's quick decisions are made by a decision model
instead of the language model. With `DECISION_BACKEND=jev`, [Jev](https://typesafe.ai),
TypeSafe's decision model, chooses where a character goes (sector, arena and object), decides
whether to talk to or react to someone, and scores how important a memory is, always over the
options the paper's code already offers. With `DECISION_BACKEND=llm`, the paper's prompts make
those decisions. In both settings the language model writes the plans, reflections and dialogue.

## What changed

All new code is in `reverie/backend_server/`. Changes to the paper's behaviour are numbered
D0 to D13 in the code comments, each with its reason, and apply to both settings unless the
comment says otherwise.

| Files | What |
|---|---|
| `persona/prompt_template/decision.py`, `jev.py`, `jev_questions.py` | The decision seam and the Jev backend. Each decision declares its options, and both settings receive the same information. |
| `persona/prompt_template/llm_trace.py`, `run_manifest.py` | A JSONL trace of every model call and decision, with the run's seed, code version and configuration. |
| `persona/prompt_template/gpt_structure.py`, `run_gpt_prompt.py`, `domain_guard.py`, `lenient_parser.py`, `seed_history.py` | The numbered changes. Among them: the authors' guard for a location answer outside the list, which they had left commented out, is switched back on (D4); the social prompts get enough tokens to reach their answer, which is read where they state it (D9, D10); and the paper's seed memories are loaded at the start (D13). |
| `automatic_execution.py`, `env_file.py`, `run_ollama.sh` | The runner: stages and checkpoints, keys from the environment, the language model through OpenRouter and embeddings on Ollama. |
| `interview_ville.py`, `read_valentine.py`, `compare_backends.py`, `analyze_llm_trace.py` | The paper's end-of-run interview, and readers that turn traces into results without touching a simulation. |
| `probe_*.py`, `world.py`, `build_w2.py`, `build_w3.py` | Questions asked offline on constructed states, and two additional worlds. |
| `environment/frontend_server/storage/base_the_ville_n7`, `base_the_ville_n9` | Smaller versions of the paper's town, with 7 and 9 residents. |

## Running it

Set up the environment and the frontend as described in the README below. Then:

```sh
cp openai_config.example.vp9.json openai_config.json
export OPEN_ROUTER_API=...       # the language model, Gemma 4 31B through OpenRouter
export TYPESAFE_API_KEY=...      # only for DECISION_BACKEND=jev
export DECISION_BACKEND=jev      # or llm
./run_ollama.sh                  # embeddings
./run_backend_automatic.sh --origin base_the_ville_n9 --target my-run --steps 17280
```

`openai_config.example.vp9.json` is the nine-resident Valentine's Day party: two simulated days,
17,280 steps, from `base_the_ville_n9`. Checkpoints are written to
`environment/frontend_server/storage/`, and the trace of each stage to `logs/llm_trace/`.

This fork was developed in a private repository and published here on 2026-10-20 with its full
commit history. Commit hashes differ from that private copy.

---



# Generative Agents: Interactive Simulacra of Human Behavior 

<p align="center" width="100%">
<img src="cover.png" alt="Smallville" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

This repository contains fixes and improvements for the repository "[generative_agents](https://github.com/joonspk-research/generative_agents)" that accompanies the paper "[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)."

Since the project is no longer officially supported, I decided to develop some new features:
- [x] Easy configuration + **Azure support**
- [x] **Cost tracking** using [openai-cost-logger](https://github.com/drudilorenzo/openai-cost-logger)
- [x] Set **cost upperbound** and stop the experiment when it is reached
- [x] New models and OpenAI API support
- [x] Added [skip-morning-s-14](https://github.com/drudilorenzo/generative_agents/tree/fix-and-improve/environment/frontend_server/storage/skip-morning-s-14): a simulation based on `base_the_ville_n25` that starts after 3000 steps (~8:00am). That permits us to save time and see interactions and actions earlier.
- [x] **Zoom in**/**Zoom out** using Z and X
- [x] [Powerful automated script](#step-3-automatic-execution) for enhanced simulation performance.

_______________________________________
## Index:
1. [Setup](#setting-up-the-environment)
2. [Execution](#running-a-simulation)
3. [Cost-Tracking](#cost-tracking)
_______________________________________

## Setting Up The Environment

### Step 1. Conda Env

Do not change the env name to be able to use the bash scripts later.
```bash
    conda create -n simulacra python=3.9.12 pip
    conda activate simulacra
    pip install -r requirements.txt
```


### Step 2. OpenAI Config

Create a file called `openai_config.json` in the root directory.\
Azure example:
```json
{
    "client": "azure", 
    "model": "gpt-35-turbo-0125",
    "model-key": "<MODEL-KEY>",
    "model-endpoint": "<MODEL-ENDPOINT>",
    "model-api-version": "<API-VERSION>",
    "model-costs": {
        "input":  0.5,
        "output": 1.5
    },
    "embeddings-client": "azure",
    "embeddings": "text-embedding-3-small",
    "embeddings-key": "<EMBEDDING-KEY>",
    "embeddings-endpoint": "<EMBEDDING-MODEL-ENDPOINT>",
    "embeddings-api-version": "<API-VERSION>",
    "embeddings-costs": {
        "input": 0.02,
        "output": 0.0
    },
    "experiment-name": "simulacra-test",
    "cost-upperbound": 10
}
```
OpenAI example:
```json
{
    "client": "openai", 
    "model": "gpt-3.5-turbo-0125",
    "model-key": "<MODEL-KEY>",
    "model-costs": {
        "input":  0.5,
        "output": 1.5
    },
    "embeddings-client": "openai",
    "embeddings": "text-embedding-3-small",
    "embeddings-key": "<EMBEDDING-KEY>",
    "embeddings-costs": {
        "input": 0.02,
        "output": 0.0
    },
    "experiment-name": "simulacra-test",
    "cost-upperbound": 10
}
```

Feel free to change and test also other models (and change accordingly the input and output costs).\
Be aware that the only supported clients are **azure** and **openai**.\
The generation and the embedding models are configured separately to be able to use different clients.\
Change also the `cost-upperbound` according to your needs (the cost computation is done using "[openai-cost-logger](https://github.com/drudilorenzo/openai-cost-logger)" and the costs are specified per million tokens).


## Running a simulation

> All the following scripts automatically activate a conda environment called `simulacra` using a conda installation at the following path: `/home/${USER}/anaconda3/bin/activate`.\
> You may want to change this line in case you are using a different conda installation/conda environment.

### Step 1. Starting the Environment Server
```bash
    ./run_frontend.sh <PORT-NUMBER>
```
 >Note: omit the port number to use the default 8000.

### Step 2. Starting the Simulation Server
```bash
    ./run_backend.sh <ORIGIN> <TARGET>
```
Example:
```bash
    ./run_backend.sh base_the_ville_isabella_maria_klaus simulation-test
```

### Step 3. Automatic Execution
The following script offer a range of enhanced features:
- `Automatic Saving`: The simulation automatically saves progress every 200 steps, ensuring you never lose data.
- `Error Recovery`: In the event of an error, the simulation automatically resumes by stepping back and restarting from the last successful point. This is crucial as the model relies on formatted answers, which can sometimes cause exceptions.
- `Automatic Tab Opening`: A new browser tab will automatically open when necessary.
- `Headless Mode`: The scripts support running simulations in Chrome's headless mode, enabling execution on a server without a UI (it needs [headless-chrome](https://developer.chrome.com/blog/headless-chrome) installed.
- `Configurable Port Number`: You can configure the port number as needed.

For more details, refer to: [run_backend_automatic.sh](https://github.com/drudilorenzo/generative_agents/blob/fix-and-improve/run_backend_automatic.sh) and [automatic_execution.py](https://github.com/drudilorenzo/generative_agents/blob/fix-and-improve/reverie/backend_server/automatic_execution.py).
```bash
    ./run_backend_automatic.sh -o <ORIGIN> -t <TARGET> -s <STEPS> --ui <True|False> -p <PORT> --browser_path <BROWSER-PATH>
```
Example:
```bash
    ./run_backend_automatic.sh -o base_the_ville_isabella_maria_klaus -t test_1 -s 4 --ui False
```

### Endpoint list
- [http://localhost:8000/](http://localhost:8000/) - check if the server is running
- [http://localhost:8000/simulator_home](http://localhost:8000/simulator_home) - watch the live simulation
- `http://localhost:8000/replay/<simulation-name>/<starting-time-step>` - replay a simulation

For a more detailed explanation see the [original readme](README_origin.md).


## Cost Tracking

For the cost tracking is used the package "[openai-cost-logger](https://github.com/drudilorenzo/openai-cost-logger)". Given the possible high cost of a simulation,  you can set a cost upperbound in the config file to be able to raise an exception and stop the execution when it is reached.

See all the details of your expenses using the notebook "[cost_viz.ipynb](https://github.com/drudilorenzo/generative_agents/blob/main/cost_viz.ipynb)."

## Cost Assessment

### 1. base_the_ville_isabella_maria_klaus

- **Model**: "gpt-3.5-turbo-0125"
- **Embeddings**: "text-embedding-3-small"
- **N. Agents**: 3
- **Steps**: ~5000
- **Final Cost**: ~0.31 USD

### 2. base_the_ville_n25

- See the simulation saved: [skip-morning-s-14](https://github.com/drudilorenzo/generative_agents/tree/fix-and-improve/environment/frontend_server/storage/skip-morning-s-14)
- **Model**: "gpt-3.5-turbo-0125"
- **Embeddings**: "text-embedding-3-small"
- **N. Agents**: 25
- **Steps**: ~3000 (until ~8 a.m.)
- **Final Cost**: ~1.3 USD

### 3. base_the_ville_n25

- **Model**: "gpt-3.5-turbo-0125"
- **Embeddings**: "text-embedding-3-small"
- **N. Agents**: 25
- **Steps**: ~8650 (full day)
- **Final Cost**: ~18.5 USD
