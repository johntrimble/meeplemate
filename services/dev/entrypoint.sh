#!/usr/bin/env bash
set -e

# Setup the .cache directory for the docker user
DOCKER_USER_CACHE_DIR="/home/$DOCKER_USER/.cache"
DATA_CACHE_DIR="/data/.cache"

# Symlink DOCKER_USER_CACHE_DIR to DATA_CACHE_DIR

# Create the .cache directory if it does not exist
if [ ! -d "$DATA_CACHE_DIR" ]; then
    echo "Creating $DATA_CACHE_DIR directory."
    mkdir -p "$DATA_CACHE_DIR"
fi

# Remove symlink if it exists and points to a different location
if [ -L "$DOCKER_USER_CACHE_DIR" ] && [ "$(readlink "$DOCKER_USER_CACHE_DIR")" != "$DATA_CACHE_DIR" ]; then
    echo "Symbolic link $DOCKER_USER_CACHE_DIR already exists but points to a different location. Removing it."
    rm "$DOCKER_USER_CACHE_DIR"
fi

# Remove DOCKER_USER_CACHE_DIR if it exists and is not a symlink
if [ -e "$DOCKER_USER_CACHE_DIR" ] && [ ! -L "$DOCKER_USER_CACHE_DIR" ]; then
    echo "Removing existing $DOCKER_USER_CACHE_DIR as it is not a symlink."
    rm -rf "$DOCKER_USER_CACHE_DIR"
fi

# If the symlink does not exist, create it
if [ ! -L "$DOCKER_USER_CACHE_DIR" ]; then
    echo "Creating symbolic link $DOCKER_USER_CACHE_DIR to $DATA_CACHE_DIR."

    # Create a new symlink
    ln -s "$DATA_CACHE_DIR" "$DOCKER_USER_CACHE_DIR"
fi

# Run any additional commands passed and replace the shell with the command
if [ $# -gt 0 ]; then
    exec "$@"
fi
