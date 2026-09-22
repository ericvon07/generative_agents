#!/bin/bash
#
# Start Ollama with the settings this project needs. Everything here is server
# configuration: no simulation code changes, no change in agent behaviour
# (see the caveat on the KV cache below).
#
#   ./run_ollama.sh          # foreground, like `ollama serve`
#
# Stop any server already running first -- these variables are only read at
# startup, so a `serve` that is already up ignores them:
#
#   pkill -f "ollama serve"

# The default context is 4096 tokens and the server TRUNCATES A LONGER PROMPT
# SILENTLY. Measured on the 3-agent bootstrap (2026-09-21): the largest prompt
# was 2354 tokens (generate_hourly_schedule), so 4096 was not yet biting --
# but the hourly schedule grows through the simulated day, and 25 agents with
# conversations go well past it. 16K is cheap insurance against a truncation
# that would never show up in a log.
export OLLAMA_CONTEXT_LENGTH=16384

# Keep both models (the chat model and the embedder) resident. The default
# unloads after 5 minutes idle, which cost 5-16 s of reload in the middle of a
# step -- and `ollama ps` came back empty right after a run.
export OLLAMA_KEEP_ALIVE=-1

# The simulation loop is serial today (one persona after another), so one slot
# is all it can use. Pinning it to 1 stops the server from reserving a KV cache
# for 4 parallel slots at 16K each. Raise this together with the parallel
# persona loop (item A2 of OTIMIZACOES.md).
export OLLAMA_NUM_PARALLEL=1

# Flash attention is the prerequisite for quantizing the K/V cache.
export OLLAMA_FLASH_ATTENTION=1

# q8_0 halves the K/V cache against the f16 default -- roughly 2.1 GB -> 1.1 GB
# for llama3.1:8b at 16K context. The docs call the precision loss "very small"
# and usually not noticeable, but it IS a numeric change: on a model with high
# GQA it can move a token. If a measurement run has to be bit-comparable with
# an earlier one, comment this line out instead of changing it mid-experiment.
export OLLAMA_KV_CACHE_TYPE=q8_0

echo "(Bash-Script-Ollama): context=${OLLAMA_CONTEXT_LENGTH} keep_alive=${OLLAMA_KEEP_ALIVE} parallel=${OLLAMA_NUM_PARALLEL} kv_cache=${OLLAMA_KV_CACHE_TYPE}"
exec ollama serve
