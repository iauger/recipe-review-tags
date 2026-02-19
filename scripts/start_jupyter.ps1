Set-Location (Split-Path $PSScriptRoot -Parent)
.\.venv\Scripts\activate

jupyter server --config scripts\jupyter_server_config.py --debug