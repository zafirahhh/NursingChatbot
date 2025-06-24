@echo off
echo Setting up KKH Nursing Chatbot...

:: Check if virtual environment exists
if not exist "eb-env" (
    echo Creating virtual environment...
    python -m venv eb-env
)

:: Activate virtual environment
echo Activating virtual environment...
call eb-env\Scripts\activate.bat

:: Install requirements
echo Installing dependencies...
pip install -r requirements.txt

:: Start the backend server
echo Starting backend server...
python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload

pause
