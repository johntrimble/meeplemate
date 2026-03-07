FROM node:20-slim
RUN apt-get update && \
    apt-get install -y default-jre-headless && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g firebase-tools
COPY firebase.json /workspace/firebase.json
WORKDIR /workspace
CMD ["firebase", "emulators:start", "--only", "auth", "--project", "boardbarian"]
