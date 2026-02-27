# Start Ollama with GPU-optimized environment variables
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_GPU_OVERHEAD = "0"
$env:OLLAMA_CONTEXT_LENGTH = "2048"
$env:OLLAMA_DEBUG = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"

Write-Host "Starting Ollama with GPU settings:"
Write-Host "  FLASH_ATTENTION = $env:OLLAMA_FLASH_ATTENTION"
Write-Host "  GPU_OVERHEAD = $env:OLLAMA_GPU_OVERHEAD"
Write-Host "  CONTEXT_LENGTH = $env:OLLAMA_CONTEXT_LENGTH"
Write-Host "  KV_CACHE_TYPE = $env:OLLAMA_KV_CACHE_TYPE"

& "C:\Users\mailp\AppData\Local\Programs\Ollama\ollama.exe" serve
