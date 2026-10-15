@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem Выбор интерпретатора: .venv, затем py-launcher, затем python
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if defined PY goto check_python
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY set "PY=python"

:check_python
%PY% -c "import sys" >nul 2>nul
if not errorlevel 1 goto check_deps
echo Python не найден. Установите Python 3.9+ с https://www.python.org/
goto fail

:check_deps
%PY% -c "import tkinter" >nul 2>nul
if not errorlevel 1 goto check_ultralytics
echo В установленном Python нет модуля tkinter. Переустановите Python с компонентом "tcl/tk and IDLE".
goto fail

:check_ultralytics
%PY% -c "import ultralytics, yaml" >nul 2>nul
if not errorlevel 1 goto start
echo Не установлены зависимости: ultralytics.
choice /C YN /M "Установить сейчас через pip"
if errorlevel 2 goto start
%PY% -m pip install -r requirements.txt
if not errorlevel 1 goto start
echo Не удалось установить зависимости.
goto fail

:start
%PY% yolo_gui.py
if errorlevel 1 goto fail
endlocal
exit /b 0

:fail
pause
endlocal
exit /b 1
