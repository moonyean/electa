@echo off
setlocal
cd /d "%~dp0"
if not exist "data\interim" mkdir "data\interim"
python -X utf8 src\data\split_train_val.py > data\interim\split_tv_run.log 2>&1
set "result=%errorlevel%"
echo SCRIPT_EXIT_CODE:%result% >> data\interim\split_tv_run.log
exit /b %result%
