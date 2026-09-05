@echo off
start "fw2job" /min cmd /c "cd /d C:\Users\User\Desktop\trans && python resume_fineweb2.py > data\interim\resume_run.log 2>&1 & echo SCRIPT_EXIT_CODE:%errorlevel% >> data\interim\resume_run.log"
