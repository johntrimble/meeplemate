# --- Stage 1: Build virtual environment and install dependencies ---
ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE} as builder

ARG VENV_PATH=/opt/venv

RUN python -m venv --system-site-packages ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:$PATH"
ENV VENV_PATH="${VENV_PATH}"

# Install the requirements using BuildKit mount from the workspace context
RUN --mount=type=bind,from=workspace,source=requirements.txt,target=/tmp/requirements.txt \
    grep -v "^[-]e" /tmp/requirements.txt > /tmp/requirements_no_editable.txt && \
    ${VENV_PATH}/bin/python -m pip install --upgrade pip && \
    ${VENV_PATH}/bin/python -m pip install --no-cache-dir -r /tmp/requirements_no_editable.txt

# --- Stage 2: Main image with user, tools, and venv from builder ---
FROM ${BASE_IMAGE}

# User and group IDs for the non-root user
# These should match the host user to avoid permission issues with mounted volumes.
ARG UID=1000
ARG GID=1000
ARG USERNAME=vscode
ARG VENV_PATH=/opt/venv

USER root

# Install useful packages
RUN apt-get update && \
    apt-get install -y \
        git \
        wget \
        curl \
        sudo \
        exa \
        fzf \
        jq \
        tree \
        bat \
        fd-find \
        ripgrep \
        ca-certificates \
        lsb-release \
        emacs-nox \
        vim \
        nano \
        gh && \
    ln -s /usr/bin/fdfind /usr/local/bin/fd && \
    ln -s /usr/bin/batcat /usr/local/bin/bat && \
    rm -rf /var/lib/apt/lists/*

# Install project system dependencies
RUN apt-get update && \
    apt-get install -y \
        ffmpeg libsm6 libxext6 \
        tesseract-ocr \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Add non-root user with the same UID and GID as the host user
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

# Create /workspace directory
RUN mkdir -p /workspace && chown $USERNAME:$USERNAME /workspace

# Add ENTRYPOINT script
COPY . /docker/
RUN chmod +x /docker/entrypoint.sh && chown $USERNAME:$USERNAME /docker/entrypoint.sh
ENTRYPOINT ["/docker/entrypoint.sh"]

# Copy the pre-built virtual environment from builder stage and set ownership
COPY --from=builder --chown=${USERNAME}:${USERNAME} ${VENV_PATH} ${VENV_PATH}

# Set the environment variable to use the virtual environment
ENV PATH="${VENV_PATH}/bin:$PATH"
ENV VENV_PATH="${VENV_PATH}"

USER $USERNAME

# Set the working directory
WORKDIR /workspace
