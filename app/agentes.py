"""Agentes especialistas.

Agente = modelo + instrucciones (system prompt) + herramientas.
create_agent monta el bucle: pensar -> usar herramienta -> leer resultado -> responder.
"""
import sys
from contextlib import AsyncExitStack

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from app.config import RAIZ
from app.herramientas import convertir_creditos, convertir_nota, presupuesto_mensual


def crear_cliente_mcp() -> MultiServerMCPClient:
    """Cliente MCP que sabe arrancar nuestro servidor FastMCP como proceso."""
    return MultiServerMCPClient(
        {
            "pasaporte": {
                "transport": "stdio",
                "command": sys.executable,  # el mismo Python que ejecuta este programa
                "args": ["-m", "app.servidor_mcp"],
                "cwd": str(RAIZ),
            }
        }
    )


async def cargar_herramientas_mcp() -> list:
    """Arranca nuestro servidor MCP y convierte sus herramientas al formato LangChain.

    Como en el manual: cada llamada a la herramienta abre una sesión nueva
    (y lanza otra vez el servidor). Vale para main.py, que hace una sola pregunta.
    """
    return await crear_cliente_mcp().get_tools()


async def abrir_herramientas_mcp(pila: AsyncExitStack) -> list:
    """Igual que cargar_herramientas_mcp, pero con UNA sesión que queda abierta.

    La usa server.py: el servidor MCP se lanza una sola vez al arrancar la web
    y se cierra al pararla (la pila de contextos se cierra en el lifespan).
    """
    cliente = crear_cliente_mcp()
    sesion = await pila.enter_async_context(cliente.session("pasaporte"))
    return await load_mcp_tools(sesion)


async def crear_especialistas(modelo, herramientas_mcp=None) -> dict:
    """Devuelve un diccionario {nombre: agente}."""
    if herramientas_mcp is None:
        herramientas_mcp = await cargar_herramientas_mcp()

    documentalista = create_agent(
        modelo,
        tools=herramientas_mcp,  # buscar_documentos, que llega desde el servidor MCP
        system_prompt="Eres el documentalista del intercambio en Lakeshore University. "
        "Busca SIEMPRE con tu herramienta buscar_documentos (puedes hacer varias "
        "búsquedas si el encargo pide varias cosas) y responde solo con lo que digan "
        "los documentos, copiando las cifras exactas y citando entre paréntesis el "
        "fichero de la fuente, por ejemplo (guia_alojamiento.md). "
        "Si no está en los documentos, dilo.",
        name="documentalista",
    )

    calculadora = create_agent(
        modelo,
        tools=[convertir_creditos, convertir_nota, presupuesto_mensual],
        system_prompt="Eres la calculadora del intercambio. Haz las cuentas SIEMPRE con "
        "tus herramientas (convertir_creditos, convertir_nota, presupuesto_mensual), "
        "nunca de cabeza, y explica cada resultado en una frase con sus números.",
        name="calculadora",
    )

    return {"documentalista": documentalista, "calculadora": calculadora}
