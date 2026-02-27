@echo off
REM Start Ollama with GPU-optimized settings
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_GPU_OVERHEAD=0
set OLLAMA_CONTEXT_LENGTH=2048
set OLLAMA_DEBUG=1
set OLLAMA_LLM_LIBRARY=cuda_v13
echo Starting Ollama with GPU settings...
echo FLASH_ATTENTION=%OLLAMA_FLASH_ATTENTION%
echo GPU_OVERHEAD=%OLLAMA_GPU_OVERHEAD%
echo CONTEXT_LENGTH=%OLLAMA_CONTEXT_LENGTH%
echo LLM_LIBRARY=%OLLAMA_LLM_LIBRARY%
"C:\Users\mailp\AppData\Local\Programs\Ollama\ollama.exe" serve
