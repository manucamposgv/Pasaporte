"""Orquestación con LangGraph: un supervisor reparte el trabajo entre especialistas.

INICIO -> supervisor -> (documentalista | calculadora) -> supervisor -> ...
       -> responder -> FIN
"""
import operator
import re
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from app.agentes import crear_especialistas
from app.config import crear_modelo

MAX_PASOS = 5  # límite de seguridad para que el supervisor no entre en bucle

PATRON_FUENTE = re.compile(r"\b[\w\-]+\.md\b")


class Estado(MessagesState):
    """Estado compartido del grafo. La clave "messages" ya viene de MessagesState."""

    siguiente: str  # a quién le toca actuar
    tarea: str  # encargo para ese especialista
    hallazgos: Annotated[list[str], operator.add]  # se acumulan, no se sustituyen
    fuentes: Annotated[list[str], operator.add]  # ficheros .md citados


class Decision(BaseModel):
    """Respuesta estructurada del supervisor."""

    siguiente: Literal["documentalista", "calculadora", "FIN"] = Field(
        description="Quién actúa ahora, o FIN si ya hay información suficiente."
    )
    tarea: str = Field(description="Encargo completo, con los datos. Vacío si FIN.")


PROMPT_SUPERVISOR = """Eres el supervisor de Pasaporte, el copiloto de un intercambio
universitario en Lakeshore University (universidad ficticia).
No respondes al estudiante: decides qué especialista actúa.
- documentalista: busca en la normativa de intercambio (créditos mínimos y máximos,
  factor ECTS/US, tabla de notas, learning agreement, plazos, idioma), la guía de
  alojamiento (residencias, precios, fianzas, barrios, transporte) y las preguntas
  frecuentes de llegada (visado, seguro médico, banco, móvil, comida).
- calculadora: convierte créditos ECTS <-> US, convierte notas españolas a GPA/letra
  y calcula presupuestos mensuales y del semestre.
Si un cálculo necesita datos de los documentos (factor de conversión, créditos mínimos,
precios de alojamiento, coste del seguro, comida, transporte, meses del semestre),
pide primero TODOS esos datos al documentalista en un solo encargo y luego pasa los
números concretos a la calculadora en un solo encargo.
No repitas un encargo que ya aparece en la información recogida.
Información recogida hasta ahora:
{hallazgos}
Si ya hay información suficiente, o la pregunta no trata del intercambio, responde FIN."""

PROMPT_RESPUESTA = """Eres Pasaporte, el copiloto del intercambio en Lakeshore University.
Responde al estudiante en español, con claridad y brevedad, usando SOLO esta información:
{hallazgos}
Cita SIEMPRE las fuentes por el nombre del fichero entre paréntesis, por ejemplo
(normativa_intercambio.md), justo detrás del dato que sale de cada una.
Si no hay información suficiente, o la pregunta no trata del intercambio, dilo."""


def _texto(estado: Estado) -> str:
    return "\n".join(estado.get("hallazgos", [])) or "Ninguna todavía."


def _fuentes(resultado: dict) -> list[str]:
    """Ficheros citados por el especialista; si no cita ninguno, los que devolvió la búsqueda."""
    citadas = PATRON_FUENTE.findall(resultado["messages"][-1].text)
    if not citadas:
        for mensaje in resultado["messages"]:
            if isinstance(mensaje, ToolMessage):
                citadas += PATRON_FUENTE.findall(mensaje.text)
    return list(dict.fromkeys(citadas))  # sin repetidos, en orden


async def construir_grafo(modelo=None, herramientas_mcp=None):
    modelo = modelo or crear_modelo()
    especialistas = await crear_especialistas(modelo, herramientas_mcp)
    decisor = modelo.with_structured_output(Decision)  # responde con el esquema Decision

    # --- Nodos: reciben el estado y devuelven solo lo que cambian ---
    async def supervisor(estado: Estado) -> dict:
        if len(estado.get("hallazgos", [])) >= MAX_PASOS:
            return {"siguiente": "FIN", "tarea": ""}
        prompt = PROMPT_SUPERVISOR.format(hallazgos=_texto(estado))
        decision = await decisor.ainvoke([SystemMessage(prompt), *estado["messages"]])
        return {"siguiente": decision.siguiente, "tarea": decision.tarea}

    def crear_nodo(nombre: str):
        async def nodo(estado: Estado) -> dict:
            # El especialista solo recibe su encargo, no toda la conversación
            entrada = {"messages": [HumanMessage(estado["tarea"])]}
            try:
                resultado = await especialistas[nombre].ainvoke(entrada)
            except Exception as e:  # p. ej., el LLM pide una herramienta con datos mal formados
                return {"hallazgos": [f"[{nombre}] No pude completar el encargo ({type(e).__name__})."],
                        "fuentes": []}
            texto = resultado["messages"][-1].text
            fuentes = _fuentes(resultado) if nombre == "documentalista" else []
            if fuentes and not PATRON_FUENTE.search(texto):
                texto += f" (fuentes: {', '.join(fuentes)})"
            return {"hallazgos": [f"[{nombre}] {texto}"], "fuentes": fuentes}

        return nodo

    async def responder(estado: Estado) -> dict:
        prompt = PROMPT_RESPUESTA.format(hallazgos=_texto(estado))
        respuesta = await modelo.ainvoke([SystemMessage(prompt), *estado["messages"]])
        return {"messages": [respuesta]}

    # --- Grafo: nodos y aristas ---
    grafo = StateGraph(Estado)
    grafo.add_node("supervisor", supervisor)
    grafo.add_node("responder", responder)
    for nombre in especialistas:
        grafo.add_node(nombre, crear_nodo(nombre))
        grafo.add_edge(nombre, "supervisor")  # cada especialista vuelve al supervisor
    grafo.add_edge(START, "supervisor")

    # Arista condicional: el destino depende de estado["siguiente"]
    rutas = {"documentalista": "documentalista", "calculadora": "calculadora",
             "FIN": "responder"}
    grafo.add_conditional_edges("supervisor", lambda estado: estado["siguiente"], rutas)
    grafo.add_edge("responder", END)
    return grafo.compile()
