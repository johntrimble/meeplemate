# Board Game Rules Assistant

This project is for experimenting with various RAG techniques with respect to answering board game questions. This repository is not intended to be a finished product, but rather a place to try out different ideas.

## Notebooks

### Self-Consistency with Board Game Questions

Notebook: [code/notebooks/self_consistency_board_game_questions.ipynb](code/notebooks/self_consistency.ipynb)

This notebook was part of a presentation I gave exploring the use of self-consistency in answering board game questions. In particular, it compares techniques for applying self-consistency to open-ended questions where it is harder to define a "consensus" answer.

## Running API and custom frontend

Backend:

```bash
uvicorn meeplemate.server.api:create_app --factory --host 0.0.0.0 --port 8000 --reload --reload-dir meeplemate
```

Frontend:

```bash
cd frontend && npm run dev
```

## Running the code

1. Open in VS Code Dev Container

   - Open the project in VS Code.
   - If prompted, reopen the project in a dev container.
   - Or use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and select:
     ```
     Dev Containers: Rebuild and Reopen in Container
     ```
2. See [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on running the backend and frontend.

