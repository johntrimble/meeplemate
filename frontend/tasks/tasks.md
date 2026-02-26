# Boardbarian Frontend

Boardbarian is the external name for meeplemate. It is an AI assistant for answer board game questions. The goal is to help speed up gameplay for games with complex rules by providing an AI assistant that can answer questions and provide clarifications about the rules faster than looking something up online or in a physical rulebook. The frontend will be the user interface for interacting with the AI assistant, and it will eventually implement the Vercel AI stream protocol to communicate with the backend.

The intent is for the frontend to be largely static in terms of its assets (no server-side rendering) and to only leverage the backend in for logged in users to provide lists of games, start new chats, access chat histories, etc. This will allow us to scale the backend to zero when not in use and to have a frontend that can be easily deployed on a CDN.

This document outlines the tasks for the frontend project. Each task includes a checklist of steps to complete and a test plan to verify that the task is done correctly. As items on the checklist are completed, they should be checked off. Once all the items are checked off for a task, and the test plan is executed successfully, the task can be considered complete. Commit only the files created or modified as part of that task using git — do not include unrelated files. Each task should result in one git commit.

Mobile is expected to be the primary platform for this application, so we should optimize the frontend for mobile devices, but it should also be responsive and work well on desktop devices as well. We can use a mobile-first design approach to ensure that the application looks and works great on mobile devices, while still providing a good experience on desktop.

## Tech Stack

- React
- TypeScript
- Vercel AI SDK (for future integration with the backend)
- Tailwind CSS (for styling)
- shadcn/ui + Elements AI components (for pre-built UI components and AI-specific UI elements)
- React Router (for navigation between screens)

## Style

Dark theme by default. The `dark` class is set on the `<html>` element in `index.html`, activating shadcn/ui's dark mode CSS variables globally. All components and screens should use semantic Tailwind color classes (`bg-background`, `text-foreground`, `text-muted-foreground`, `bg-card`, `border-border`, etc.) rather than hardcoded colors, so they automatically respect the theme.


## Tasks

### Task: Create frontend project

We need to make a new frontend project. Eventually, we will implement the Vercel AI stream protocol (https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol) on the backend to be used by the frontend, but for now we will focus on just getting the frontend project set up and creating the main home screen page. The frontend should not depend on server-side rendering.

- [x] Create frontend project in `frontend` directory
- [x] Add README.md (in the frontend directory) with instructions on how to run the frontend project locally
- [x] Use React and TypeScript
- [x] Set up basic project structure
- [x] Create the main home screen page (consult the frontend/tasks/boardbarian_screens.png)
- [x] Add Elements AI (https://elements.ai-sdk.dev/)
- [x] Ensure we have no dependency on a backend for now
- [x] Create a test plan for ensuring the frontend build and the home screen page works correctly
    * This could be a test you run manually
    * Fill out the test plan below
- [x] Execute the test plan to verify this task is done

#### Test Plan for Task

1. **Production build passes** — run `npm run build` from `frontend/`. Expected: exits 0, creates `dist/` with hashed assets, no TypeScript errors. ✅ Verified.
2. **Dev server starts** — run `npm run dev`. Expected: Vite starts at `http://localhost:5173` with no errors.
3. **Home screen renders correctly** — open `http://localhost:5173`. Verify:
   - Page loads with no console errors
   - White background, clean layout
   - "Sign In" button visible in the top-right corner
   - Rounded logo card (🎲 Boardbarian) visible on the left
   - Marketing copy ("Rules questions, answered instantly.") visible to the right of the logo
   - Layout looks correct at mobile viewport (390px wide in DevTools)
   - Layout looks correct at desktop viewport (full-width)
4. **Preview build works** — run `npm run build && npm run preview`. Repeat step 3 at `http://localhost:4173`.


### Task: Create Select Game Screen

- [x] Add mock user to frontend project (we are not adding a dependency on the backend yet)
- [x] Create a new screen for selecting a game (consult the frontend/tasks/boardbarian_screens.png)
  * Just pull some images from a dozen or so popular games and use those as mock data for now
  * Include games from ./meeplemate/eval/test_cases.yaml
- [x] For now, have clicking on the "Sign In" button take the user to the select game screen from the home screen
- [x] Designate some games as "recently used" by the mock user
- [x] Create a test plan for ensuring the frontend build and the home screen page works correctly
    * This could be a test you run manually
    * Fill out the test plan below
- [x] Execute the test plan to verify this task is done

#### Test Plan for Task

1. **Production build passes** — run `npm run build` from `frontend/`. Expected: exits 0, no TypeScript errors. ✅ Verified.
2. **Dev server starts** — run `npm run dev`. Expected: Vite starts at `http://localhost:5173` with no errors.
3. **Sign In navigates to Select Game** — on the home screen, click "Sign In". Expected: navigates to `/select-game` and the Select Game screen renders.
4. **Select Game screen layout** — verify at mobile (390px) and desktop viewports:
   - "Select a Game" title visible top-left
   - User avatar ("AJ") visible top-right
   - "Recently Used" section with 4 game cards in a 4-column grid (Munchkin, Warhammer, Catan, Codenames)
   - All 13 games displayed in a 3-column grid below
   - Each card shows an emoji on a colored background with the game name below
   - Cards have hover and active (tap) scale animations
5. **Game cards are tappable** — clicking any game card navigates to `/chat` (route not yet implemented; a blank/404 page is acceptable until Task 3).
6. **No console errors** — open DevTools and confirm no errors in the console on either screen.


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

