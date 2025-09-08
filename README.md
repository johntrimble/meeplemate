# Board Game Rules Assistant

This project is for experimenting with various RAG techniques with respect to answering board game questions. This repository is not intended to be a finished product, but rather a place to try out different ideas.

## Notebooks

### Self-Consistency with Board Game Questions

Notebook: [code/notebooks/self_consistency_board_game_questions.ipynb](code/notebooks/self_consistency.ipynb)

This notebook was part of a presentation I gave exploring the use of self-consistency in answering board game questions. In particular, it compares techniques for applying self-consistency to open-ended questions where it is harder to define a "consensus" answer.

## Running the code

1. Download the Munchkin rule PDFs from https://munchkin.game/gameplay/rules/ and place them in the code/munchkin_rules directory
5. Open in VS Code Dev Container

   - Open the project in VS Code.
   - If prompted, reopen the project in a dev container.
   - Or use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and select:
     ```
     Dev Containers: Rebuild and Reopen in Container
     ```
6. Run chainlit app in the dev container terminal:
   ```
   chainlit run main.py
   ```
7. Open the URL printed in the terminal (usually http://localhost:8000)

