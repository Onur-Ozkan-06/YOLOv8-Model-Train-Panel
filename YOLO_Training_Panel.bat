@echo off
:: Türkçe karakterleri düzelt
chcp 65001 > nul

:: Çalışma dizinini, bu .bat dosyasının bulunduğu klasör yap
cd /d "%~dp0"

:: Sanal ortam varsa aktif et (Opsiyonel ama iyi olur)
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

:: Başlık
title YOLOv8 Egitim Paneli

echo.
echo ==========================================
echo   YOLOv8 Egitim Paneli Baslatiliyor...
echo ==========================================
echo.

:: Uygulamayı çalıştır
python -m streamlit run app.py

pause