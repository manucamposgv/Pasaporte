"""Pregunta a Pasaporte desde la consola.

Uso:
    python main.py "¿Cuántos ECTS son 12 créditos US?"
"""
import asyncio
import sys

from langchain_core.messages import HumanMessage

from app.grafo import construir_grafo


async def main() -> None:
    pregunta = " ".join(sys.argv[1:]) or input("Pregunta: ")
    grafo = await construir_grafo()
    entrada = {"messages": [HumanMessage(pregunta)]}

    # stream_mode="updates" devuelve, paso a paso, lo que hace cada nodo
    async for paso in grafo.astream(entrada, stream_mode="updates"):
        for nodo, cambios in paso.items():
            if nodo == "supervisor" and cambios["siguiente"] == "FIN":
                print("  [supervisor] Ya hay información suficiente.")
            elif nodo == "supervisor":
                print(f"  [supervisor] -> {cambios['siguiente']}: {cambios['tarea']}")
            elif nodo == "responder":
                print(f"\nPasaporte: {cambios['messages'][-1].text}")
            else:
                print(f"  {cambios['hallazgos'][-1][:200]}")


if __name__ == "__main__":
    if sys.platform == "win32":  # la consola de Windows no siempre usa UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
