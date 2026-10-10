@echo off
rem Launch the race strategy simulator with the project environment.
cd /d "%~dp0"
"%USERPROFILE%\.venvs\f1sim\Scripts\python.exe" -m streamlit run 05_app.py --browser.gatherUsageStats false
