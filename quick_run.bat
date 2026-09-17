@echo off

echo ---------- Generando datos mediante simulacion FVM ... ----------
python src/handle_data/generate.py
if %errorlevel% neq 0 (
    echo Error en la generacion de los datos
    exit /b %errorlevel%
)

echo ---------- Entrenando modelo mediante datos generados ... ----------
python src/models/train.py
if %errorlevel% neq 0 (
    echo Error en el entrenamiento del modelo
    exit /b %errorlevel%
)

echo ---------- Prediciendo solucion mediante modelo entrenado ... ----------
python src/models/predict.py
if %errorlevel% neq 0 (
    echo Error en la predicción de la solución
    exit /b %errorlevel%
)