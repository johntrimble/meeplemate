#!/usr/bin/env bash

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MODEL_NAME="TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
MODEL_MAX_LENGTH=8096

function cuda_version() {
    nvcc --version | grep "release" | awk '{print $6}' | cut -c2-
}

function python_version() {
    python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
}

# Assert python version is a 3.10.x release
if [[ "$(python_version)" != "3.10"* ]]; then
    echo "Python 3.10.x is required"
    exit 1
fi

# Assert CUDA version is 11.8.x
if [[ "$(cuda_version)" != "11.8"* ]]; then
    echo "CUDA 11.8.x is required"
    exit 1
fi

if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
    python -m venv "$SCRIPT_DIR/.venv"
fi

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Install requirements
"$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

# Run OpenAI compatible API server
# max_model_len
"$SCRIPT_DIR/.venv/bin/python" -m vllm.entrypoints.openai.api_server --max-model-len $MODEL_MAX_LENGTH --model "$MODEL_NAME" --dtype float16 --quantization awq --trust-remote-code --port 8000