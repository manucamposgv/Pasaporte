@echo off
rem Arranque con doble clic en Windows: instala si hace falta, hace la ingesta
rem la primera vez, abre el navegador y lanza la web en http://localhost:8000
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo [1/3] Creando el entorno virtual e instalando dependencias...
  python -m venv .venv || goto :error
  .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
)

if not exist .env (
  copy .env.example .env >nul
  echo Se ha creado .env: abre el fichero y pon tu GROQ_API_KEY.
)

if not exist chroma_db (
  echo [2/3] Ingesta de los documentos en ChromaDB...
  .venv\Scripts\python.exe -m app.ingesta || goto :error
)

echo [3/3] Web en http://localhost:8000  (Ctrl+C para parar)
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8000"
.venv\Scripts\python.exe -m uvicorn server:app
goto :eof

:error
echo Algo ha fallado. Revisa el mensaje de arriba y el README.
pause
