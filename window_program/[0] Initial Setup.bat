@echo off

set root=%UserProfile%\anaconda3
call %root%\Scripts\activate.bat %root%
call pip install --no-index --find-links="./sources/requirements" -r .\sources\requirements.txt
pause