@echo off
::chcp 65001 > nul

::set PROJEYOLU=C:\Users\onurz\OneDrive\Masaüstü\PPE_PYTHON

::if not exist "%PROJEYOLU%" (
::    echo KLASOR BULUNAMADI:
::    echo %PROJEYOLU%
::    pause
::    exit
::)

::pushd "%PROJEYOLU%"
python -m streamlit run app.py
pause