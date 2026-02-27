@echo off
SET MOVIE_ASSISTANT_SERVICE_URL=http://localhost:8004
SET ORACLE_SERVICE_URL=http://localhost:8001
SET BINGE_SERVICE_URL=http://localhost:8002
SET SENTIMENT_SERVICE_URL=http://localhost:8003
SET PYTHONIOENCODING=utf-8
cd /d "c:\Users\mailp\StreamSage\gateway"
python main.py >> "c:\Users\mailp\StreamSage\gateway.log" 2>> "c:\Users\mailp\StreamSage\gateway_err.log"
