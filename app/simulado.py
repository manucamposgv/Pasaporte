"""SOLO PARA PRUEBAS: modelo de lenguaje y embeddings simulados.

No lo usa el código de producción salvo que lo pidas con variables de entorno:
    MODO_SIMULADO=1         -> crear_modelo() devuelve ModeloSimulado (sin Groq)
    EMBEDDINGS_SIMULADOS=1  -> crear_embeddings() devuelve EmbeddingsSimulados (sin fastembed)

ModeloSimulado imita con reglas sencillas a los cuatro papeles del grafo
(supervisor, documentalista, calculadora y responder) para poder probar de
punta a punta la arquitectura real: LangGraph, create_agent, llamadas a
herramientas, servidor MCP, ChromaDB, SSE y streaming de tokens.
Sus respuestas son mecánicas; no sirve para evaluar la calidad del asistente.
"""
import asyncio
import hashlib
import json
import math
import re
import time
import uuid
from typing import Any, Iterator, AsyncIterator

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

# ---------------------------------------------------------------- embeddings


class EmbeddingsSimulados(Embeddings):
    """Bolsa de trigramas de caracteres con hashing, normalizada (384 dimensiones).

    Recupera por coincidencia léxica; basta para probar el RAG sin descargar nada.
    """

    DIM = 384

    def _vector(self, texto: str) -> list[float]:
        v = [0.0] * self.DIM
        palabras = re.findall(r"\w+", texto.lower())
        for palabra in palabras:
            p = f" {palabra} "
            for i in range(len(p) - 2):
                h = int(hashlib.md5(p[i:i + 3].encode()).hexdigest(), 16)
                v[h % self.DIM] += 1.0
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in textos]

    def embed_query(self, texto: str) -> list[float]:
        return self._vector(texto)


# ------------------------------------------------------------- utilidades

TEMA = ("ects", "crédito", "credito", "nota", "gpa", "residencia", "alojamiento", "piso",
        "alquiler", "seguro", "visado", "llegada", "banco", "cuenta", "móvil", "movil",
        "presupuesto", "cost", "cuesta", "semestre", "lakeshore", "learning", "convalid",
        "idioma", "toefl", "ielts", "plazo", "fianza", "barrio", "transporte", "comida",
        "asignatura", "orientación", "welcome", "port alder", "intercambio")


def _num(texto: str) -> float:
    return float(texto.replace(".", "").replace(",", ".")) if "," in texto else float(texto)


def _palabras(texto: str) -> set[str]:
    return {p for p in re.findall(r"\w+", texto.lower()) if len(p) > 3}


def _ultimo_humano(mensajes: list[BaseMessage]) -> str:
    for m in reversed(mensajes):
        if isinstance(m, HumanMessage):
            return m.text
    return ""


def _sistema(mensajes: list[BaseMessage]) -> str:
    return next((m.text for m in mensajes if isinstance(m, SystemMessage)), "")


def _hallazgos(sistema: str) -> str:
    m = re.search(r"(?:hasta ahora|esta información):\n(.*?)\n(?:Si ya hay|Cita SIEMPRE)", sistema, re.S)
    return m.group(1) if m else ""


# ------------------------------------------------------- papeles simulados


def _decidir(mensajes: list[BaseMessage]) -> dict:
    """Supervisor: documentalista -> calculadora -> FIN, según la pregunta."""
    pregunta = _ultimo_humano(mensajes)
    q = pregunta.lower()
    hallazgos = _hallazgos(_sistema(mensajes))
    if not any(t in q for t in TEMA):
        return {"siguiente": "FIN", "tarea": ""}

    presupuesto = any(k in q for k in ("presupuesto", "costar", "cuesta", "gastar", "gasto"))
    creditos = re.search(r"\d", q) and any(k in q for k in ("ects", "crédito", "credito"))
    nota = re.search(r"\d", q) and any(k in q for k in ("nota", "gpa", "sobre 10", "/10"))
    calculo = bool(presupuesto or creditos or nota)
    docs = not calculo or presupuesto or any(
        k in q for k in ("mínimo", "minimo", "máximo", "llega", "residencia", "seguro", "requisit"))

    if docs and "[documentalista]" not in hallazgos:
        partes = [p.strip() for p in re.split(r"[?¿;]", pregunta) if len(p.strip()) > 3]
        if creditos:
            partes.append("créditos mínimos por semestre y factor de conversión ECTS US")
        if presupuesto:
            partes += ["alquiler mensual de las residencias", "seguro médico LakeCare coste al mes",
                       "presupuesto orientativo de comida y transporte al mes"]
        return {"siguiente": "documentalista", "tarea": "; ".join(partes)}

    if calculo and "[calculadora]" not in hallazgos:
        return {"siguiente": "calculadora", "tarea": _encargo_calculo(q, hallazgos)}
    return {"siguiente": "FIN", "tarea": ""}


def _encargo_calculo(q: str, hallazgos: str) -> str:
    llamadas = []
    m = re.search(r"(\d+)\s*asignaturas?\s*de\s*(\d+[.,]?\d*)\s*créditos?\s*(us|ects)", q)
    if m:
        total, origen = int(m.group(1)) * _num(m.group(2)), m.group(3).upper()
    else:
        m = re.search(r"(\d+[.,]?\d*)\s*(?:créditos?\s*)?(us|ects)\b", q)
        total, origen = (_num(m.group(1)), m.group(2).upper()) if m else (None, None)
    if total is not None:
        destino = "ECTS" if origen == "US" else "US"
        llamadas.append(f"convertir_creditos(creditos={total:g}, origen={origen}, destino={destino})")

    m = re.search(r"(\d+[.,]?\d*)\s*(?:sobre 10|/10)", q) or re.search(r"nota\D{0,20}(\d+[.,]?\d*)", q)
    if m and ("nota" in q or "gpa" in q):
        llamadas.append(f"convertir_nota(nota={_num(m.group(1)):g}, escala_origen=ES)")

    if any(k in q for k in ("presupuesto", "costar", "cuesta", "gastar", "gasto")):
        def buscar(patron: str, defecto: float) -> float:
            encontrados = [_num(x) for x in re.findall(patron, hallazgos)]
            return min(encontrados) if encontrados else defecto
        alquiler = buscar(r"\|\s*([\d.]+)\s*\$\s*\|", 780)
        comida = buscar(r"[Cc]omida: unos (\d+)", 350)
        transporte = buscar(r"(\d+) \$ al mes con el Alder Pass", 45)
        seguro = buscar(r"equivalen a (\d+) \$ al mes", 250)
        meses = int(buscar(r"cuenta como (\d+) meses", 5))
        llamadas.append(f"presupuesto_mensual(alquiler={alquiler:g}, comida={comida:g}, "
                        f"transporte={transporte:g}, seguro={seguro:g}, otros=0, meses={meses})")
    return "Calcula: " + "; ".join(llamadas)


def _llamadas_desde_encargo(encargo: str) -> list[dict]:
    llamadas = []
    for nombre, args in re.findall(r"(\w+)\(([^)]*)\)", encargo):
        argumentos = {}
        for clave, valor in re.findall(r"(\w+)=([^,]+)", args):
            valor = valor.strip().strip("'\"")
            try:
                argumentos[clave] = float(valor) if "." in valor else int(valor)
            except ValueError:
                argumentos[clave] = valor
        llamadas.append({"name": nombre, "args": argumentos, "id": f"call_{uuid.uuid4().hex[:8]}",
                         "type": "tool_call"})
    return llamadas


def _resumir_busqueda(consulta: str, resultado: str) -> list[str]:
    """Frases de los fragmentos que más palabras comparten con la consulta, con su fuente."""
    claves = _palabras(consulta)
    candidatas = []
    for bloque in resultado.split("[Fuente: ")[1:]:
        fuente, _, texto = bloque.partition("]\n")
        for frase in re.split(r"(?<=[.:])\s+|\n", texto):
            frase = frase.strip(" -#")
            if len(frase) > 25 and not frase.startswith("|") and frase[-1] in ".:)":
                candidatas.append((len(claves & _palabras(frase)), f"{frase} ({fuente})"))
    candidatas.sort(key=lambda c: -c[0])
    return [c[1] for c in candidatas[:2] if c[0] > 0]


def _responder_rol(mensajes: list[BaseMessage], herramientas: list[str]) -> AIMessage:
    sistema = _sistema(mensajes)
    ultimo = mensajes[-1]

    if "buscar_documentos" in herramientas:  # documentalista
        if isinstance(ultimo, ToolMessage):
            lineas = []
            llamadas = {c["id"]: c["args"].get("consulta", "") for m in mensajes
                        if isinstance(m, AIMessage) for c in m.tool_calls}
            for m in mensajes:
                if isinstance(m, ToolMessage):
                    lineas += _resumir_busqueda(llamadas.get(m.tool_call_id, ""), m.text)
            lineas = list(dict.fromkeys(lineas)) or ["No lo encuentro en los documentos."]
            return AIMessage("Según los documentos: " + " ".join(lineas))
        consultas = [c.strip() for c in _ultimo_humano(mensajes).split(";") if c.strip()]
        return AIMessage("", tool_calls=[
            {"name": "buscar_documentos", "args": {"consulta": c}, "id": f"call_{uuid.uuid4().hex[:8]}",
             "type": "tool_call"} for c in consultas])

    if herramientas:  # calculadora
        if isinstance(ultimo, ToolMessage):
            resultados = [m.text for m in mensajes if isinstance(m, ToolMessage)]
            return AIMessage("Resultados: " + " | ".join(resultados))
        llamadas = _llamadas_desde_encargo(_ultimo_humano(mensajes))
        if not llamadas:
            return AIMessage("No hay datos suficientes para calcular.")
        return AIMessage("", tool_calls=llamadas)

    # responder: redacta con los hallazgos
    hallazgos = [h for h in _hallazgos(sistema).splitlines() if h.strip()]
    if not hallazgos or hallazgos == ["Ninguna todavía."]:
        return AIMessage("Solo puedo ayudarte con el intercambio en Lakeshore University "
                         "(normativa, alojamiento, llegada y cuentas). *(modo simulado)*")
    fuentes = list(dict.fromkeys(re.findall(r"\b[\w\-]+\.md\b", "\n".join(hallazgos))))
    def viñeta(h: str) -> str:
        return "- " + re.sub(r"^\[(\w+)\] ", lambda m: f"**{m.group(1).capitalize()}:** ", h)
    cuerpo = "\n".join(viñeta(h) for h in hallazgos)
    texto = f"Esto es lo que he encontrado *(modo simulado)*:\n\n{cuerpo}"
    if fuentes:
        texto += "\n\nFuentes: " + ", ".join(f"({f})" for f in fuentes)
    return AIMessage(texto)


# ------------------------------------------------------------------ modelo


class ModeloSimulado(BaseChatModel):
    """Chat model de reglas, compatible con create_agent, with_structured_output y streaming."""

    herramientas: list[str] = []
    pausa: float = 0.015  # segundos entre tokens al hacer streaming

    @property
    def _llm_type(self) -> str:
        return "pasaporte-simulado"

    def bind_tools(self, tools: list, **kwargs: Any):
        nombres = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"herramientas": nombres})

    def with_structured_output(self, schema, **kwargs: Any):
        return RunnableLambda(lambda entrada: schema(**_decidir(_a_mensajes(entrada))))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=_responder_rol(messages, self.herramientas))])

    def _trozos(self, messages) -> Iterator[AIMessageChunk]:
        mensaje = _responder_rol(messages, self.herramientas)
        if mensaje.tool_calls:
            yield AIMessageChunk(content="", tool_call_chunks=[
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(mensaje.tool_calls)])
            return
        for token in re.findall(r"\S+\s*|\s+", mensaje.text):
            yield AIMessageChunk(content=token)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        for trozo in self._trozos(messages):
            if run_manager and trozo.content:
                run_manager.on_llm_new_token(trozo.content, chunk=ChatGenerationChunk(message=trozo))
            time.sleep(self.pausa)
            yield ChatGenerationChunk(message=trozo)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs) -> AsyncIterator[ChatGenerationChunk]:
        for trozo in self._trozos(messages):
            if run_manager and trozo.content:
                await run_manager.on_llm_new_token(trozo.content, chunk=ChatGenerationChunk(message=trozo))
            await asyncio.sleep(self.pausa)
            yield ChatGenerationChunk(message=trozo)


def _a_mensajes(entrada) -> list[BaseMessage]:
    if hasattr(entrada, "to_messages"):
        return entrada.to_messages()
    return list(entrada)
