"""Web de Pasaporte: FastAPI sirve la página y expone el grafo con streaming SSE.

Arranque (desde la raíz del proyecto, tras la ingesta):
    uvicorn server:app
y abre http://localhost:8000

Eventos SSE de POST /api/chat (cada uno con data en JSON):
    decision   {"siguiente", "tarea"}         lo que decide el supervisor
    hallazgo   {"nodo", "texto"}              lo que aporta un especialista
    fuentes    {"nodo", "fuentes": [...]}     ficheros .md citados
    nodo       {"nodo"}                       un nodo del grafo ha terminado (va tras los anteriores)
    token      {"texto"}                      trozo de la respuesta final (nodo responder)
    respuesta  {"texto", "fuentes": [...]}    respuesta final completa
    error      {"mensaje"}
    fin        {}
"""
import json
import logging
import os
import re
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agentes import abrir_herramientas_mcp
from app.config import CARPETA_CHROMA, EMBEDDINGS_SIMULADOS, MODO_SIMULADO, RAIZ
from app.grafo import PATRON_FUENTE, construir_grafo

log = logging.getLogger("pasaporte")
MAX_HISTORIAL = 6  # turnos previos que se mandan al grafo como contexto


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Crea el grafo UNA vez al arrancar y mantiene abierta la sesión MCP."""
    app.state.grafo, app.state.error, app.state.herramientas = None, None, []
    async with AsyncExitStack() as pila:
        try:
            if not CARPETA_CHROMA.exists():
                raise RuntimeError("No existe chroma_db/: ejecuta antes python -m app.ingesta")
            herramientas = await abrir_herramientas_mcp(pila)
            app.state.herramientas = [h.name for h in herramientas]
            app.state.grafo = await construir_grafo(herramientas_mcp=herramientas)
            log.info("Grafo listo. Herramientas MCP: %s", app.state.herramientas)
        except Exception as e:  # la web arranca igual y explica el problema
            app.state.error = _explicar(e)
            log.error("No se pudo crear el grafo: %s", app.state.error)
        yield  # aquí la aplicación atiende peticiones; al salir se cierra el MCP


app = FastAPI(title="Pasaporte", lifespan=lifespan)


def _explicar(e: Exception) -> str:
    texto = f"{type(e).__name__}: {e}"
    if "api_key" in texto.lower() or "GROQ_API_KEY" in texto:
        return "Falta GROQ_API_KEY en .env (o arranca con MODO_SIMULADO=1 para probar)."
    if "401" in texto or "invalid api key" in texto.lower():
        return "La clave de Groq no es válida: revisa GROQ_API_KEY en .env."
    if "429" in texto or "rate limit" in texto.lower():
        return "Límite por minuto del plan gratuito de Groq (429). Espera unos segundos y repite."
    return texto[:500]


def _sse(tipo: str, datos: dict) -> str:
    return f"event: {tipo}\ndata: {json.dumps(datos, ensure_ascii=False)}\n\n"


class Turno(BaseModel):
    rol: str  # "usuario" o "asistente"
    texto: str


class Peticion(BaseModel):
    mensaje: str = Field(min_length=1, max_length=2000)
    historial: list[Turno] = []


@app.get("/")
async def inicio():
    return FileResponse(RAIZ / "web" / "index.html")


@app.get("/api/salud")
async def salud(request: Request):
    estado = request.app.state
    return {
        "ok": estado.grafo is not None,
        "modelo": "simulado" if MODO_SIMULADO else os.getenv("MODELO", "groq:openai/gpt-oss-120b"),
        "embeddings": "simulados" if EMBEDDINGS_SIMULADOS else "fastembed",
        "herramientas_mcp": estado.herramientas,
        "error": estado.error,
    }


@app.post("/api/chat")
async def chat(peticion: Peticion, request: Request):
    grafo = request.app.state.grafo
    error = request.app.state.error

    mensajes = []
    for turno in peticion.historial[-MAX_HISTORIAL:]:
        clase = HumanMessage if turno.rol == "usuario" else AIMessage
        mensajes.append(clase(turno.texto))
    mensajes.append(HumanMessage(peticion.mensaje))

    async def eventos():
        if grafo is None:
            yield _sse("error", {"mensaje": error or "El grafo no está listo."})
            yield _sse("fin", {})
            return
        fuentes: list[str] = []
        texto_final = ""
        try:
            # "updates": qué cambia cada nodo; "messages": tokens de los LLM
            async for modo, datos in grafo.astream(
                {"messages": mensajes}, stream_mode=["updates", "messages"]
            ):
                if modo == "messages":
                    trozo, meta = datos
                    if meta.get("langgraph_node") == "responder" and trozo.text:
                        yield _sse("token", {"texto": trozo.text})
                    continue
                for nodo, cambios in datos.items():
                    cambios = cambios or {}
                    if nodo == "supervisor":
                        yield _sse("decision", {"siguiente": cambios.get("siguiente", "FIN"),
                                                "tarea": cambios.get("tarea", "")})
                    elif nodo == "responder":
                        texto_final = cambios["messages"][-1].text
                    else:
                        texto = cambios.get("hallazgos", [""])[-1]
                        yield _sse("hallazgo", {"nodo": nodo, "texto": re.sub(r"^\[\w+\] ", "", texto)})
                        nuevas = cambios.get("fuentes", [])
                        if nuevas:
                            fuentes += [f for f in nuevas if f not in fuentes]
                            yield _sse("fuentes", {"nodo": nodo, "fuentes": nuevas})
                    yield _sse("nodo", {"nodo": nodo})  # el nodo ha terminado
            # Fuentes de la respuesta: las que cita el texto y, además, las de los hallazgos
            citadas = list(dict.fromkeys(PATRON_FUENTE.findall(texto_final) + fuentes))
            yield _sse("respuesta", {"texto": texto_final, "fuentes": citadas})
        except Exception as e:
            log.exception("Error en el grafo")
            yield _sse("error", {"mensaje": _explicar(e)})
        yield _sse("fin", {})

    return StreamingResponse(
        eventos(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
