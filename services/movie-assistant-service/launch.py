"""Launcher script for movie assistant service with proper env vars."""
import os
import subprocess
import sys

os.environ["INFERENCE_BACKEND"] = "local"
os.environ["LOCAL_MODEL_PATH"] = r"c:\Users\mailp\StreamSage\services\movie-assistant-service\fine_tuning\outputs\final_model"
os.environ["HF_API_TOKEN"] = "hf_your_token_here"
os.environ["PYTHONIOENCODING"] = "utf-8"

os.chdir(r"c:\Users\mailp\StreamSage\services\movie-assistant-service")
subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"])
