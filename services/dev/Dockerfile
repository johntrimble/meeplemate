# --- Stage 1: Build virtual environment and install dependencies ---
# ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

FROM ${BASE_IMAGE} AS builder

ARG VENV_PATH=/opt/venv

# Minimal build tools for native wheels; add git if your deps use VCS URLs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential pkg-config git \
 && rm -rf /var/lib/apt/lists/*

# Install Python
# Add Deadsnakes and install Python 3.12 + venv/dev headers
ARG PYTHON_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg software-properties-common \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
  && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_TOOL_BIN_DIR=/opt/uv-bin/ \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_CACHE_DIR=/opt/uv-cache/ \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-system
RUN mkdir -p ${UV_TOOL_BIN_DIR} ${UV_PYTHON_CACHE_DIR}

# Install project dependencies
WORKDIR /workspace
COPY --from=workspace pyproject.toml ./
COPY --from=workspace uv.lock* ./
RUN uv sync --python /usr/bin/python${PYTHON_VERSION} --no-install-project
WORKDIR /

# --- Stage 2: Main image with user, tools, and venv from builder ---
FROM ${BASE_IMAGE}
# Install Python
# Add Deadsnakes and install Python 3.12 + venv/dev headers
ARG PYTHON_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg software-properties-common \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
  && rm -rf /var/lib/apt/lists/*

# Install useful packages
RUN apt-get update && \
    apt-get install -y \
        git \
        sudo \
        fzf \
        jq \
        tree \
        bat \
        fd-find \
        ripgrep \
        lsb-release \
        emacs-nox \
        vim \
        nano \
        gh && \
    ln -s /usr/bin/fdfind /usr/local/bin/fd && \
    ln -s /usr/bin/batcat /usr/local/bin/bat && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js (for claude copilot VSCode extension)
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# User and group IDs for the non-root user
# These should match the host user to avoid permission issues with mounted volumes.
ARG USERNAME=vscode
ARG UID=1000
ARG GID=1000
ENV DOCKER_USER=${USERNAME}

RUN set -eux; \
    EXISTING_GROUP=$(getent group $GID | cut -d: -f1 || true); \
    if [ -z "$EXISTING_GROUP" ]; then \
        groupadd -g $GID $USERNAME; \
        GROUPNAME=$USERNAME; \
    else \
        # Rename the group to $USERNAME
        groupmod -n $USERNAME $EXISTING_GROUP; \
        GROUPNAME=$USERNAME; \
    fi; \
    EXISTING_USER=$(getent passwd $UID | cut -d: -f1 || true); \
    if [ -z "$EXISTING_USER" ]; then \
        useradd -u $UID -g $GID -m -s /bin/bash $USERNAME; \
        USERNAME_FINAL=$USERNAME; \
    else \
        # Rename the existing user to $USERNAME and update home/shell
        usermod -l $USERNAME -d /home/$USERNAME -m -s /bin/bash $EXISTING_USER; \
        USERNAME_FINAL=$USERNAME; \
    fi; \
    chsh -s /bin/bash $USERNAME_FINAL

# Allow the user to use sudo without a password
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && chmod 0440 /etc/sudoers.d/${USERNAME}

# Persist bash history
ENV HISTFILE=/commandhistory/.bash_history
ENV HISTSIZE=10000
ENV HISTFILESIZE=20000
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME:$USERNAME /commandhistory

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_TOOL_BIN_DIR=/opt/uv-bin/ \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_CACHE_DIR=/opt/uv-cache/ \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-system

USER ${USERNAME}
RUN ["/bin/bash", "-c", "mkdir /home/${USERNAME}/.claude && set -o pipefail && curl -fsSL https://claude.ai/install.sh | bash"]
USER root

# Copy over virtualenv
COPY --from=builder --chown=${USERNAME}:${USERNAME} /opt/venv /opt/venv

# Copy over other misc uv files
COPY --from=builder --chown=${USERNAME}:${USERNAME} /opt/uv-bin /opt/uv-bin
COPY --from=builder --chown=${USERNAME}:${USERNAME} /opt/uv-cache /opt/uv-cache

# Create /workspace and /data directories
RUN mkdir -p /workspace /data && chown -R $USERNAME:$USERNAME /workspace /data

USER root
# Add ENTRYPOINT script
COPY . /docker/
RUN chmod +x /docker/entrypoint.sh && chown $USERNAME:$USERNAME /docker/entrypoint.sh
ENTRYPOINT ["/docker/entrypoint.sh"]

# Set the environment variable to use the virtual environment
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:/home/${USERNAME}/.local/bin:$PATH"
ENV DEVCONTAINER=true
ENV EDITOR=emacs

USER $USERNAME

# Set the working directory
WORKDIR /workspace
