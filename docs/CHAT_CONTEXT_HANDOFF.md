# StreamSage Workspace Handoff & Chat Context

This document serves as a persistent memory state for moving development between devices. Reading this file immediately brings you (and your AI agent) up to speed on current progress, problems, and immediate next steps.

## 1. Recently Completed Tasks
- **Dockerization & Orchestration**: Created a robust `docker-compose.yml` supporting 7 services (Gateway, Oracle, Binge, Sentiment, Movie Assistant, Redis, Qdrant).
- **PowerShell Automation**: Implemented `scripts/deploy/docker_stack.ps1`, `verify_stack_health.ps1`, and `run_smoke_tests.ps1` to manage the stack. Fixed `Invoke-WebRequest` Explorer UI blocking issues.
- **Git Authentication**: Solved GitHub 403 permissions out of VS Code by rewriting commit authors specifically to `aryankadam1256 <aryankadam1256@users.noreply.github.com>`.
- **Comprehensive Docs**: Added 10+ markdown files in `docs/docker/` explaining the entire architecture, how to handle the HF cache, and operations playbooks. All code successfully pushed to `main`.

## 2. Immediate Next Step: Oracle API Upgrade
**Problem**: The `oracle-rag-service` relies on local `llama3:8b` (via Groq/Ollama), which throws aggressive "Token Limit Exceeded" errors when ingesting large subtitle chunks from ChromaDB.

**Agreed Solution Plan**: 
Switch the LLM client to **OpenAI `gpt-4o-mini`** (or **Gemini 1.5 Flash**) to leverage massive 128k+ token context windows and much faster, more reliable streaming.

*Action Items for Next Session:*
1. Create new branch: `git checkout -b feature/oracle-api-upgrade`.
2. Add `OPENAI_API_KEY` (or `GEMINI_API_KEY`) to `.env` on the new device.
3. Refactor `services/oracle-rag-service/main.py` to replace the custom `OllamaClient` with an API client that supports Server-Sent Events (SSE) streaming (matching the existing pipeline).
4. Update `docker-compose.yml` to inject the new API key into the Oracle container.
5. Validate streaming response via Gateway.

## 3. Secondary Context: PDF Image Extractor
- A standalone standalone `pdf-image-extractor.html` was briefly discussed and provided in the editor context.
- It uses `pdf.js` for canvas rendering and threshold-based flood filling to extract images locally without external APIs.
- We clarified that the *Oracle chatbot token limit* issue is completely separate from this PDF vision tool. 

## 4. How to Resume on New Device
1. `git clone https://github.com/aryankadam1256/StreamSage.git`
2. `cd StreamSage`
3. Manually create your local `.env` and `.env.deploy` files (since these are `.gitignore`d for security).
4. Run `.\scripts\deploy\docker_stack.ps1 up`.
5. Begin the Oracle API Upgrade.
