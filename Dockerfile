# Imagen para publicar Pasaporte en Hugging Face Spaces (SDK Docker).
# También sirve en local:  docker build -t pasaporte .
#                          docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... pasaporte
FROM python:3.12-slim

# Hugging Face ejecuta el contenedor con el usuario 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/user/fastembed_cache
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Ingesta al construir: descarga el modelo de embeddings y crea chroma_db/
# dentro de la imagen, así el arranque es rápido y no depende de la red.
RUN python -m app.ingesta

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
