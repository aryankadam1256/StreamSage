@echo off
SET CHROMA_DB_PATH=c:\Users\mailp\StreamSage\services\movie-assistant-service\data\chroma_db
SET INFERENCE_BACKEND=local
SET LOCAL_MODEL_PATH=c:\Users\mailp\StreamSage\services\movie-assistant-service\fine_tuning\outputs\final_model
SET HF_API_TOKEN=%HF_API_TOKEN%
SET PYTHONIOENCODING=utf-8
SET PYTHONPATH=c:\Users\mailp\StreamSage\services\movie-assistant-service
cd /d "c:\Users\mailp\StreamSage\services\movie-assistant-service"
python -m uvicorn main:app --host 0.0.0.0 --port 8004 >> "c:\Users\mailp\StreamSage\movie_assistant.log" 2>> "c:\Users\mailp\StreamSage\movie_assistant_err.log"
