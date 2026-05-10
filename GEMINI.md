# Project Mandates: Neuro-Randki

## General Guidelines
- **Project Role:** This is a "Love Tester" style biometric application (Neuro-Randki) using mock EEG data and mock "neuroguard" biometrics models for initial development.
- **Tone & Style:** The UI should be fun, engaging, and arcade-like, suitable for events, integrations, and ice-breakers.
- **Tech Stack:** Stick to Python, Flask, and SQLite. Do not introduce heavy frontend frameworks (like React or Vue) unless explicitly requested; standard HTML/CSS/JS with Jinja templates is preferred to keep the integration simple.

## Core Directives
1. **Mock Data First:** 
   - All actual EEG hardware communication is currently a TODO. You MUST use placeholder/mock data generators for any brainwave streaming.
   - All "neuroguard" model inferences are currently a TODO. You MUST use a mock function that returns a simulated similarity score (e.g., random percentage or basic algorithm based on the mock data).
2. **Kiosk UI & Synchronization:** 
   - The application runs on a single shared screen for two users simultaneously.
   - Task synchronization is driven by the frontend UI. The frontend must display the task and explicitly signal the backend (via API) when to start and stop the mocked data collection.
3. **Database Simplicity:**
   - Use SQLite for data storage. Store user profiles (Nickname, Age, Gender - optional) and their test results/scores.
   - Keep the database schema flat and simple.
4. **Data Handling:**
   - Ensure the mock data collection simulates up to 5 minutes of data streaming (even if accelerated for testing).
   - Ensure user profiles and similarity results are permanently logged to SQLite after each session.
