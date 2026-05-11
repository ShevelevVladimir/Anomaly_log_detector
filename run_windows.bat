@echo off
chcp 65001 >nul
echo Запуск прототипа интеллектуальной системы обнаружения аномалий...
python -m streamlit run app.py
pause
