"""Configuración común: rutas, modelo de lenguaje y modelo de embeddings."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings

# Lee las variables del fichero .env (por ejemplo, GROQ_API_KEY)
RAIZ = Path(__file__).parent.parent
load_dotenv(RAIZ / ".env")

CARPETA_DOCS = RAIZ / "docs"          # documentos de los que aprende el RAG
CARPETA_CHROMA = RAIZ / "chroma_db"   # donde ChromaDB guarda la base vectorial
COLECCION = "pasaporte"               # nombre de la "tabla" dentro de ChromaDB

# Modelo de embeddings multilingüe que se ejecuta en tu ordenador (gratis, sin clave)
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Solo para pruebas sin clave ni red (ver app/simulado.py y el README):
#   MODO_SIMULADO=1        -> LLM simulado en lugar de Groq
#   EMBEDDINGS_SIMULADOS=1 -> embeddings simulados en lugar de fastembed
MODO_SIMULADO = os.getenv("MODO_SIMULADO") == "1"
EMBEDDINGS_SIMULADOS = os.getenv("EMBEDDINGS_SIMULADOS") == "1"


def crear_modelo():
    """Modelo de lenguaje (LLM). Por defecto, gpt-oss-120b en el plan gratuito de Groq
    (Llama 3.3, el del manual, ya no está disponible en Groq)."""
    if MODO_SIMULADO:
        from app.simulado import ModeloSimulado
        return ModeloSimulado()
    return init_chat_model(os.getenv("MODELO", "groq:openai/gpt-oss-120b"))


class EmbeddingsLocales(Embeddings):
    """Convierte textos en vectores con la librería fastembed.

    Hereda de la clase Embeddings de LangChain para que ChromaDB sepa usarla.
    """

    def __init__(self):
        from fastembed import TextEmbedding
        # La primera vez descarga el modelo (unos 220 MB) y lo guarda en caché
        self.modelo = TextEmbedding(MODELO_EMBEDDINGS)

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.modelo.embed(textos)]

    def embed_query(self, texto: str) -> list[float]:
        return self.embed_documents([texto])[0]


def crear_embeddings() -> Embeddings:
    if EMBEDDINGS_SIMULADOS:
        from app.simulado import EmbeddingsSimulados
        return EmbeddingsSimulados()
    return EmbeddingsLocales()
