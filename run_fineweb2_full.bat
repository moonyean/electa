@echo off
start "fw2splittv" /min cmd /c "cd /d C:\Users\User\Desktop\trans && python split_train_val.py > data\interim\split_tv_run.log 2>&1 & echo SCRIPT_EXIT_CODE:%errorlevel% >> data\interim\split_tv_run.log"
