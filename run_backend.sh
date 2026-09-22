#!/bin/bash

# Resolve the repo dir BEFORE any cd so the venv lookup works with relative invocation.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND_SCRIPT_PATH="reverie/backend_server"
BACKEND_SCRIPT_FILE="reverie.py"
CONDA_ENV="simulacra"
LOGS_PATH="../../logs"

echo "Running backend server at: http://127.0.0.1:8000/simulator_home"
cd ${BACKEND_SCRIPT_PATH}
# Prefer the project-level .venv (../.venv relative to this repo); fall back to conda.
if [ -f "${REPO_DIR}/../.venv/bin/activate" ]; then
    source "${REPO_DIR}/../.venv/bin/activate"
elif [ -f "${REPO_DIR}/.venv/bin/activate" ]; then
    source "${REPO_DIR}/.venv/bin/activate"
elif [ -f "/home/${USER}/anaconda3/bin/activate" ]; then
    source /home/${USER}/anaconda3/bin/activate ${CONDA_ENV}
fi

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Timestamp: ${timestamp}"
mkdir -p ${LOGS_PATH}
python3 ${BACKEND_SCRIPT_FILE} --origin ${1} --target ${2} | tee  ${LOGS_PATH}/${2}_${timestamp}.txt