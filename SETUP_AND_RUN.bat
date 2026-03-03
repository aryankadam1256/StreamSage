@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  SETUP_AND_RUN.bat  -  StreamSage Full Setup and Launcher
::
::  USAGE:
::    SETUP_AND_RUN.bat           -- First-time setup, then run all
::    SETUP_AND_RUN.bat setup     -- Install all dependencies only
::    SETUP_AND_RUN.bat run       -- Start all services (skip install)
::    SETUP_AND_RUN.bat stop      -- Kill all service processes
::    SETUP_AND_RUN.bat help      -- Show this help
:: ============================================================

set BASE=%~dp0
:: Remove trailing backslash
if "%BASE:~-1%"=="\" set BASE=%BASE:~0,-1%

if /i "%1"=="setup" goto :SETUP
if /i "%1"=="run"   goto :RUN
if /i "%1"=="stop"  goto :STOP
if /i "%1"=="help"  goto :HELP
if "%1"==""         goto :SETUP_THEN_RUN

:HELP
echo.
echo  Usage: SETUP_AND_RUN.bat [setup ^| run ^| stop ^| help]
echo.
echo    (no args) - First-time setup + run all services
echo    setup     - Install all Python and Node dependencies
echo    run       - Start all services (assumes deps are installed)
echo    stop      - Kill all running service ports
echo    help      - Show this message
echo.
goto :EOF

:: ============================================================
:SETUP_THEN_RUN
call :do_setup
if errorlevel 1 exit /b 1
call :do_run
goto :EOF

:SETUP
call :do_setup
goto :EOF

:RUN
call :do_run
goto :EOF

:STOP
call :do_stop
goto :EOF

:: ============================================================
:: SETUP FUNCTION
:: ============================================================
:do_setup

echo.
echo ============================================================
echo   StreamSage  ^|  DEPENDENCY SETUP
echo ============================================================
echo.

:: ── 1. Prerequisite checks ──────────────────────────────────
echo [STEP 0]  Checking prerequisites...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Python not found.
    echo          Download from https://www.python.org/downloads/
    exit /b 1
) else (
    for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo   Python  : %%v
)

node --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Node.js not found.
    echo          Download from https://nodejs.org
    exit /b 1
) else (
    for /f "tokens=*" %%v in ('node --version 2^>^&1') do echo   Node.js : %%v
)

npm --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: npm not found (should come with Node.js).
    exit /b 1
) else (
    for /f "tokens=*" %%v in ('npm --version 2^>^&1') do echo   npm     : v%%v
)

ollama --version >nul 2>&1
if errorlevel 1 (
    echo   Ollama  : NOT FOUND  ^(Oracle RAG service will be unavailable^)
    echo             Download from https://ollama.ai
    set OLLAMA_MISSING=1
) else (
    for /f "tokens=*" %%v in ('ollama --version 2^>^&1') do echo   Ollama  : %%v
    set OLLAMA_MISSING=0
)
echo.

:: ── 2. Python  - Gateway ─────────────────────────────────────
echo [STEP 1/6]  Installing Gateway dependencies...
pip install -r "%BASE%\gateway\requirements.txt" --quiet --disable-pip-version-check
echo             Done.
echo.

:: ── 3. Python  - Oracle RAG ──────────────────────────────────
echo [STEP 2/6]  Installing Oracle RAG Service dependencies...
echo             (includes torch, sentence-transformers, chromadb - may take a few minutes)
pip install -r "%BASE%\services\oracle-rag-service\requirements.txt" --quiet --disable-pip-version-check
echo             Done.
echo.

:: ── 4. Python  - Binge Predictor ─────────────────────────────
echo [STEP 3/6]  Installing Binge Predictor Service dependencies...
echo             (includes tensorflow - large download on first run)
pip install -r "%BASE%\services\binge-service\requirements.txt" --quiet --disable-pip-version-check
echo             Done.
echo.

:: ── 5. Python  - Sentiment Engine ────────────────────────────
echo [STEP 4/6]  Installing Sentiment Engine dependencies...
echo             (includes transformers, torch)
pip install -r "%BASE%\services\sentiment-service\requirements.txt" --quiet --disable-pip-version-check
echo             Done.
echo.

:: ── 6. Python  - Movie Assistant ─────────────────────────────
echo [STEP 5/6]  Installing Movie Assistant Service dependencies...
pip install -r "%BASE%\services\movie-assistant-service\requirements.txt" --quiet --disable-pip-version-check
echo             Done.
echo.

:: ── 7. Node.js - Frontend ────────────────────────────────────
echo [STEP 6/6]  Installing Frontend (npm) dependencies...
cd /d "%BASE%\frontend"
npm install --silent
cd /d "%BASE%"
echo             Done.
echo.

:: ── 8. Ollama model pull ──────────────────────────────────────
if "%OLLAMA_MISSING%"=="0" (
    echo [STEP 7/7]  Pulling Ollama model: llama3:8b
    echo             This downloads ~4.7 GB on first run. Please wait...
    echo.
    start /B "" ollama serve
    timeout /t 5 /nobreak >nul
    ollama pull llama3:8b
    echo.
    echo             llama3:8b ready.
) else (
    echo [STEP 7/7]  SKIPPED - Ollama not installed.
    echo             Oracle RAG questions-about-specific-movies feature will not work.
)

echo.
echo ============================================================
echo   SETUP COMPLETE!
echo ============================================================
echo   Run:  SETUP_AND_RUN.bat run     to start all services.
echo ============================================================
echo.
exit /b 0


:: ============================================================
:: RUN FUNCTION — starts all 7 services in separate windows
:: ============================================================
:do_run

echo.
echo ============================================================
echo   StreamSage  ^|  STARTING ALL SERVICES
echo ============================================================
echo.

:: ── Environment variables inherited by all child processes ──
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG_LEVEL=INFO

:: Ollama
set OLLAMA_BASE_URL=http://localhost:11434
set LLM_MODEL=llama3:8b

:: Oracle RAG
set ORACLE_SERVICE_URL=http://localhost:8001

:: Binge & Sentiment
set BINGE_SERVICE_URL=http://localhost:8002
set SENTIMENT_SERVICE_URL=http://localhost:8003

:: Movie Assistant
set MOVIE_ASSISTANT_SERVICE_URL=http://localhost:8004
set CHROMA_DB_PATH=%BASE%\services\movie-assistant-service\data\chroma_db
set INFERENCE_BACKEND=api
set LOCAL_MODEL_PATH=%BASE%\services\movie-assistant-service\fine_tuning\outputs\final_model

:: HF token (if set in environment)
if "%HF_API_TOKEN%"=="" (
    echo   WARNING: HF_API_TOKEN is not set.
    echo            Movie Assistant will work in retrieval-only mode (no LLM answers).
    echo            To enable LLM: set HF_API_TOKEN=your_hf_token  then re-run.
    echo.
)

:: ── STEP 1: Ollama ───────────────────────────────────────────
echo [Step 1/7]  Ollama LLM Server  (port 11434)
ollama --version >nul 2>&1
if errorlevel 1 (
    echo             SKIPPED - Ollama not installed
) else (
    start "StreamSage ^ Ollama :11434" cmd /k "title StreamSage - Ollama LLM :11434 && ollama serve"
    timeout /t 4 /nobreak >nul
    echo             Started (window: StreamSage ^ Ollama :11434)
)
echo.

:: ── STEP 2: Oracle RAG ──────────────────────────────────────
echo [Step 2/7]  Oracle RAG Service  (port 8001)
start "StreamSage ^ Oracle RAG :8001" cmd /k "title StreamSage - Oracle RAG :8001 && cd /d %BASE%\services\oracle-rag-service && python main.py"
echo             Started (window: StreamSage ^ Oracle RAG :8001)
echo.

:: ── STEP 3: Binge Predictor ──────────────────────────────────
echo [Step 3/7]  Binge Predictor Service  (port 8002)
start "StreamSage ^ Binge :8002" cmd /k "title StreamSage - Binge Predictor :8002 && cd /d %BASE%\services\binge-service && python main.py"
echo             Started (window: StreamSage ^ Binge :8002)
echo.

:: ── STEP 4: Sentiment Engine ─────────────────────────────────
echo [Step 4/7]  Sentiment Analysis Service  (port 8003)
start "StreamSage ^ Sentiment :8003" cmd /k "title StreamSage - Sentiment Engine :8003 && cd /d %BASE%\services\sentiment-service && python main.py"
echo             Started (window: StreamSage ^ Sentiment :8003)
echo.

:: ── STEP 5: Movie Assistant ───────────────────────────────────
echo [Step 5/7]  Movie Assistant Service  (port 8004)
start "StreamSage ^ Movie Assistant :8004" cmd /k "title StreamSage - Movie Assistant :8004 && set PYTHONPATH=%BASE%\services\movie-assistant-service && cd /d %BASE%\services\movie-assistant-service && python -m uvicorn main:app --host 0.0.0.0 --port 8004"
echo             Started (window: StreamSage ^ Movie Assistant :8004)
echo.

:: ── Wait for backend services to be ready ────────────────────
echo   [Waiting 18 seconds for backend services to initialize...]
timeout /t 18 /nobreak
echo.

:: ── STEP 6: API Gateway ──────────────────────────────────────
echo [Step 6/7]  API Gateway  (port 8000)
start "StreamSage ^ Gateway :8000" cmd /k "title StreamSage - API Gateway :8000 && cd /d %BASE%\gateway && python main.py"
echo             Started (window: StreamSage ^ Gateway :8000)
echo.

:: ── Wait for gateway ─────────────────────────────────────────
timeout /t 5 /nobreak >nul

:: ── STEP 7: Frontend ─────────────────────────────────────────
echo [Step 7/7]  Frontend Dev Server  (port 5174)
start "StreamSage ^ Frontend :5174" cmd /k "title StreamSage - Frontend :5174 && cd /d %BASE%\frontend && npm run dev"
echo             Started (window: StreamSage ^ Frontend :5174)
echo.

:: ── Summary ──────────────────────────────────────────────────
echo ============================================================
echo   ALL SERVICES STARTED  -  7 terminal windows opened
echo ============================================================
echo.
echo   Service               Port     URL
echo   ---------------------------------------------------
echo   Ollama LLM Server     11434    http://localhost:11434
echo   Oracle RAG (docs)      8001    http://localhost:8001/docs
echo   Binge Predictor        8002    http://localhost:8002/docs
echo   Sentiment Engine       8003    http://localhost:8003/docs
echo   Movie Assistant        8004    http://localhost:8004/docs
echo   API Gateway (docs)     8000    http://localhost:8000/docs
echo   Frontend App (**main**)5174    http://localhost:5174
echo   ---------------------------------------------------
echo.
echo   OPEN APP:  http://localhost:5174
echo   API DOCS:  http://localhost:8000/docs
echo.
echo   Opening browser in 12 seconds...
echo   (Close any service window to stop that service)
echo.
timeout /t 12 /nobreak >nul
start http://localhost:5174
goto :EOF


:: ============================================================
:: STOP FUNCTION — kills processes on all service ports
:: ============================================================
:do_stop
echo.
echo [STOP]  Killing all StreamSage service ports...
echo.

for %%P in (8000 8001 8002 8003 8004 11434) do (
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr ":%%P " ^| findstr "LISTENING"') do (
        echo   Killing PID %%A on port %%P
        taskkill /F /PID %%A >nul 2>&1
    )
)

echo.
echo   Done. All services stopped.
echo.
goto :EOF
