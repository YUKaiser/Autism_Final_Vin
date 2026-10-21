@echo off
echo Setting up the project environment...
if not exist venv (
    echo Creating fresh virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo Installing required libraries...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)
echo Starting your project...
python main.py
pause
