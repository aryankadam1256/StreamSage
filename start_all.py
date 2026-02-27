"""
StreamSage - Full Stack Startup Script
Starts all services in the correct order with health checks.
Run: python start_all.py
"""
import subprocess
import sys
import time
import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORACLE_SVC_DIR = os.path.join(BASE_DIR, "services", "oracle-rag-service")
MOVIE_SVC_DIR  = os.path.join(BASE_DIR, "services", "movie-assistant-service")
GATEWAY_DIR    = os.path.join(BASE_DIR, "gateway")
FRONTEND_DIR   = os.path.join(BASE_DIR, "frontend")

# ── Config ────────────────────────────────────────────────────────────────────
ORACLE_PORT          = 8001
MOVIE_ASSISTANT_PORT = 8004
GATEWAY_PORT         = 8000
FRONTEND_PORT        = 5174

ENV = {
    **os.environ,
    "CHROMA_DB_PATH":        os.path.join(MOVIE_SVC_DIR, "data", "chroma_db"),
    "INFERENCE_BACKEND":     "auto",
    "LOCAL_MODEL_PATH":      os.path.join(MOVIE_SVC_DIR, "fine_tuning", "outputs", "final_model"),
    "HF_API_TOKEN":          os.environ.get("HF_API_TOKEN", ""),
    "PYTHONIOENCODING":      "utf-8",
    "PYTHONPATH":            MOVIE_SVC_DIR,
    "ORACLE_SERVICE_URL":    f"http://localhost:{ORACLE_PORT}",
    "MOVIE_ASSISTANT_SERVICE_URL": f"http://localhost:{MOVIE_ASSISTANT_PORT}",
}


def kill_port(port: int):
    """Kill any process listening on the given port."""
    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if f":{port} " in line and "LISTEN" in line:
            pid = line.strip().split()[-1]
            subprocess.run(["taskkill", "/F", "/PID", pid],
                           capture_output=True, text=True)
            print(f"  Killed existing process {pid} on port {port}")
            time.sleep(2)
            return
    print(f"  Port {port} was free")


def wait_for_http(url: str, timeout: int = 40, label: str = "") -> bool:
    """Poll an HTTP endpoint until it returns 200 or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_oracle():
    print("\n[1/4] Starting Oracle RAG Service (port 8001)...")
    kill_port(ORACLE_PORT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", str(ORACLE_PORT)],
        cwd=ORACLE_SVC_DIR,
        env={**ENV, "PYTHONPATH": ORACLE_SVC_DIR},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("  Waiting for Oracle to be healthy...")
    if wait_for_http(f"http://localhost:{ORACLE_PORT}/health", timeout=40,
                     label="oracle"):
        info = requests.get(f"http://localhost:{ORACLE_PORT}/health", timeout=5).json()
        print(f"  Oracle READY — {info.get('collection_count', '?')} chunks in ChromaDB")
        return proc
    else:
        print("  ERROR: Oracle failed to start in time")
        proc.kill()
        sys.exit(1)


def start_movie_assistant():
    print("\n[2/4] Starting Movie Assistant Service (port 8004)...")
    kill_port(MOVIE_ASSISTANT_PORT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", str(MOVIE_ASSISTANT_PORT)],
        cwd=MOVIE_SVC_DIR,
        env=ENV,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("  Waiting for service to be healthy...")
    if wait_for_http(f"http://localhost:{MOVIE_ASSISTANT_PORT}/health", timeout=40,
                     label="movie-assistant"):
        info = requests.get(f"http://localhost:{MOVIE_ASSISTANT_PORT}/health", timeout=5).json()
        print(f"  Movie assistant READY — {info.get('total_movies', '?')} movies loaded")
        return proc
    else:
        print("  ERROR: Movie assistant failed to start in time")
        proc.kill()
        sys.exit(1)


def start_gateway():
    print("\n[3/4] Starting Gateway Service (port 8000)...")
    kill_port(GATEWAY_PORT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", str(GATEWAY_PORT)],
        cwd=GATEWAY_DIR,
        env=ENV,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Gateway /health calls other services which don't exist — just wait for port
    time.sleep(4)
    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
    listening = any(
        f":{GATEWAY_PORT} " in line and "LISTEN" in line
        for line in result.stdout.splitlines()
    )
    if listening:
        print("  Gateway READY")
        return proc
    else:
        print("  ERROR: Gateway failed to start")
        proc.kill()
        sys.exit(1)


def start_frontend():
    print("\n[4/4] Starting Frontend Dev Server (port 3000)...")
    kill_port(FRONTEND_PORT)

    # Clear stale Vite cache so first browser load is fast
    vite_cache = os.path.join(FRONTEND_DIR, "node_modules", ".vite")
    if os.path.exists(vite_cache):
        import shutil
        shutil.rmtree(vite_cache, ignore_errors=True)
        print("  Cleared stale Vite cache")

    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    proc = subprocess.Popen(
        [npm, "run", "dev", "--", "--force"],
        cwd=FRONTEND_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("  Waiting for Vite to finish pre-bundling...")
    if wait_for_http(f"http://localhost:{FRONTEND_PORT}", timeout=60,
                     label="frontend"):
        print("  Frontend READY")
        return proc
    else:
        print("  ERROR: Frontend failed to start in time")
        proc.kill()
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 55)
    print("  StreamSage — Starting All Services")
    print("=" * 55)

    procs = []
    procs.append(start_oracle())
    procs.append(start_movie_assistant())
    procs.append(start_gateway())
    procs.append(start_frontend())

    print("\n" + "=" * 55)
    print("  All services are UP!")
    print("=" * 55)
    print(f"  Frontend:        http://localhost:{FRONTEND_PORT}")
    print(f"  Gateway API:     http://localhost:{GATEWAY_PORT}")
    print(f"  Oracle RAG:      http://localhost:{ORACLE_PORT}")
    print(f"  Movie Assistant: http://localhost:{MOVIE_ASSISTANT_PORT}")
    print("\n  Press Ctrl+C to stop all services.")
    print("=" * 55)

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nShutting down all services...")
        for p in procs:
            p.terminate()
        print("Done.")
