@echo off
start "fw2merge" /min cmd /c "cd /d C:\Users\User\Desktop\trans && python merge_all_shards.py > data\interim\merge_run.log 2>&1 & echo SCRIPT_EXIT_CODE:%errorlevel% >> data\interim\merge_run.log"
