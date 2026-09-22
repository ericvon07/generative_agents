#!/bin/bash

# Resolve the repo dir BEFORE any cd so the venv lookup works with relative invocation.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FRONTEND_SCRIPT_PATH="environment/frontend_server"
FRONTEND_SCRIPT_FILE="manage.py"
CONDA_ENV="simulacra"

FILE_NAME="Bash-Script-Frontend"
echo "(${FILE_NAME}): Running frontend server"
cd ${FRONTEND_SCRIPT_PATH}
# Prefer the project-level .venv (../.venv relative to this repo); fall back to conda.
if [ -f "${REPO_DIR}/../.venv/bin/activate" ]; then
    source "${REPO_DIR}/../.venv/bin/activate"
elif [ -f "${REPO_DIR}/.venv/bin/activate" ]; then
    source "${REPO_DIR}/.venv/bin/activate"
elif [ -f "/home/${USER}/anaconda3/bin/activate" ]; then
    source /home/${USER}/anaconda3/bin/activate ${CONDA_ENV}
fi

PORT=8000
if [ -z "$1" ]
then
    echo "(${FILE_NAME}): No port provided. Using default port: ${PORT}"
else
    PORT=$1
fi

python3 ${FRONTEND_SCRIPT_FILE} runserver ${PORT}