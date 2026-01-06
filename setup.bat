@echo off
echo Setting up Amazon Ads Optimization Pipeline...

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

REM Create directory structure
mkdir amazon_reports 2>nul
mkdir output\logs 2>nul
mkdir output\csv_output 2>nul
mkdir output\aggregated_data 2>nul

REM Create .env file template
if not exist .env (
    echo # OpenAI API Configuration > .env
    echo OPENAI_API_KEY=your-openai-api-key-here >> .env
    echo. >> .env
    echo # Directory Configuration >> .env
    echo DATA_DIRECTORY=./amazon_reports >> .env
    echo OUTPUT_DIRECTORY=./output >> .env
    echo.
    echo Created .env file template
    echo Please edit .env and add your OpenAI API key
) else (
    echo .env file already exists
)

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Edit .env and add your OpenAI API key
echo 2. Place your Amazon Ads reports in .\amazon_reports\
echo 3. Run: python run_pipeline.py

pause