"""Prueba el servidor MCP sin agentes ni LLM: lista sus herramientas y llama a una.

Uso:
    python probar_mcp.py "¿Cuánto cuesta la residencia más barata?"
"""
import asyncio
import sys

from fastmcp import Client

from app.servidor_mcp import servidor


async def main() -> None:
    consulta = " ".join(sys.argv[1:]) or "créditos mínimos por semestre"
    # Client(servidor) conecta con el servidor en memoria, sin lanzar otro proceso
    async with Client(servidor) as cliente:
        herramientas = await cliente.list_tools()
        print("Herramientas:", [h.name for h in herramientas])
        resultado = await cliente.call_tool("buscar_documentos", {"consulta": consulta})
        print(resultado.content[0].text)


if __name__ == "__main__":
    if sys.platform == "win32":  # la consola de Windows no siempre usa UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
