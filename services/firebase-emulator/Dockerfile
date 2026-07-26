FROM node:20-slim
RUN apt-get update && \
    apt-get install -y default-jre-headless && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g firebase-tools
COPY firebase.json /workspace/firebase.json
WORKDIR /workspace

# Persist the auth database across restarts. Without this the emulator is purely
# in-memory, so every `docker compose up` wiped the users and the next sign-in
# minted a *new* Firebase uid for the same person — which, since the uid is the
# account key, silently created a second account and stranded the old chats.
# One dev address had accumulated 18 accounts that way.
#
# --import only accepts a directory that already holds an export, so skip it on
# the very first run.
CMD ["sh", "-c", "\
  if [ -f /data/firebase-export-metadata.json ]; then IMPORT=--import=/data; fi; \
  exec firebase emulators:start --only auth --project boardbarian \
       $IMPORT --export-on-exit=/data"]
