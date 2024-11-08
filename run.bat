@echo off
setlocal

rem Define Python version and virtual environment directory
set "python_version=3.11"
set "venv_dir=Batch_correction_from_MZmine"

rem Get the directory path where the batch file is located
set "script_dir=%~dp0"

rem Check if the virtual environment already exists in the script's directory
if exist "%script_dir%\%venv_dir%\Scripts\activate" (
    echo %venv_dir% already exists. Activating...
    call "%script_dir%\%venv_dir%\Scripts\activate"
) else (
    echo Creating virtual environment in "%script_dir%\%venv_dir%"...
    py -%python_version% -m venv "%script_dir%\%venv_dir%"
    call "%script_dir%\%venv_dir%\Scripts\activate"
)

rem Check if Jupyter is installed
pip show jupyter >nul 2>&1
if errorlevel 1 (
    echo Jupyter is not installed. Installing now...
    echo Installing the requirements...
    python -m pip install --upgrade pip
    python -m pip install -r "%script_dir%\requirements.txt"
    echo Installing Jupyter Notebook...
    python -m pip install jupyter notebook
    python -m pip install traitlets
) else (
    echo Jupyter is already installed.
)

rem Start Jupyter Notebook
call jupyter notebook

pause
