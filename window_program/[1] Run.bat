@echo off

set root=%UserProfile%\anaconda3
call %root%\Scripts\activate.bat %root%
call python ./sources/main.py
pause