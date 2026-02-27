@echo off
SET VITE_API_URL=http://localhost:8000/api/v1
cd /d "c:\Users\mailp\StreamSage\frontend"
npm run dev >> "c:\Users\mailp\StreamSage\frontend.log" 2>&1
