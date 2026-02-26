# Boardbarian Frontend

Boardbarian is the external name for meeplemate. It is an AI assistant for answer board game questions. The goal is to help speed up gameplay for games with complex rules by providing an AI assistant that can answer questions and provide clarifications about the rules faster than looking something up online or in a physical rulebook. The frontend will be the user interface for interacting with the AI assistant, and it will eventually implement the Vercel AI stream protocol to communicate with the backend.

The intent is for the frontend to be largely static in terms of its assets (no server-side rendering) and to only leverage the backend in for logged in users to provide lists of games, start new chats, access chat histories, etc. This will allow us to scale the backend to zero when not in use and to have a frontend that can be easily deployed on a CDN.

This document outlines the tasks for the frontend project. Each task includes a checklist of steps to complete and a test plan to verify that the task is done correctly. As items on the checklist are completed, they should be checked off. Once all the items are checked off for a task, and the test plan is executed successfully, the task can be considered complete and the changes committed using git.

Mobile is expected to be the primary platform for this application, so we should optimize the frontend for mobile devices, but it should also be responsive and work well on desktop devices as well. We can use a mobile-first design approach to ensure that the application looks and works great on mobile devices, while still providing a good experience on desktop.

## Tech Stack

- React
- TypeScript
- Vercel AI SDK (for future integration with the backend)
- Tailwind CSS (for styling)
- shadcn/ui + Elements AI components (for pre-built UI components and AI-specific UI elements)
- React Router (for navigation between screens)


## Tasks

### Task: Create frontend project

We need to make a new frontend project. Eventually, we will implement the Vercel AI stream protocol (https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol) on the backend to be used by the frontend, but for now we will focus on just getting the frontend project set up and creating the main home screen page. The frontend should not depend on server-side rendering.

- [ ] Create frontend project in `frontend` directory
- [ ] Add README.md (in the frontend directory) with instructions on how to run the frontend project locally
- [ ] Use React and TypeScript
- [ ] Set up basic project structure
- [ ] Create the main home screen page (consult the frontend/tasks/boardbarian_screens.png)
- [ ] Add Elements AI (https://elements.ai-sdk.dev/)
- [ ] Ensure we have no dependency on a backend for now
- [ ] Create a test plan for ensuring the frontend build and the home screen page works correctly
    * This could be a test you run manually
    * Fill out the test plan below
- [ ] Execute the test plan to verify this task is done

#### Test Plan for Task

TBD


### Task: Create Select Game Screen

- [ ] Add mock user to frontend project (we are not adding a dependency on the backend yet)
- [ ] Create a new screen for selecting a game (consult the frontend/tasks/boardbarian_screens.png)
  * Just pull some images from a dozen or so popular games and use those as mock data for now
  * Include games from ./meeplemate/eval/test_cases.yaml
- [ ] For now, have clicking on the "Sign In" button take the user to the select game screen from the home screen
- [ ] Designate some games as "recently used" by the mock user
- [ ] Create a test plan for ensuring the frontend build and the home screen page works correctly
    * This could be a test you run manually
    * Fill out the test plan below
- [ ] Execute the test plan to verify this task is done

#### Test Plan for Task

TBD


### Task: Create New Chat Screen

- [ ] Create a new screen for the chat interface (consult the frontend/tasks/boardbarian_screens.png)
- [ ] Add sidebar navigation to the chat screen
- [ ] "Select Game" should navigate back to the "Select Game" screen
- [ ] Create a test plan for ensuring the frontend build and the home screen page works correctly
    * This could be a test you run manually
    * Fill out the test plan below
- [ ] Execute the test plan to verify this task is done

#### Test Plan for Task

TBD


### Task: Create chat conversation view

- [ ] Implement the view for chats with existing messages
- [ ] Create mock conversations with messages for testing
  * Use the test cases in ./meeplemate/eval/test_cases.yaml as a guide for creating mock conversations
  * Also you can look at the runs under data/evals/generation_runs
- [ ] Add chain-of-though components to AI assistant responses that support nested thoughts (since the AI Agent will be multi-agent)
- [ ] Add buttons under AI assistant resposne for "Copy", "Thumbs Up", "Thumbs Down", and "Regenerate"
- [ ] Create a test plan for ensuring the frontend build and the home screen page works correctly
    * This could be a test you run manually
    * Fill out the test plan below
- [ ] Execute the test plan to verify this task is done

#### Test Plan for Task

TBD

