@echo off
rem Launch the interactive simulator with the project environment.
cd /d "%~dp0"
"%USERPROFILE%\.venvs\f1sim\Scripts\python.exe" -m streamlit run 09_interactive_simulator.py --browser.gatherUsageStats false
