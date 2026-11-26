"""Herramientas locales del agente calculadora.

Una herramienta es una función normal con el decorador @tool. El modelo lee
el docstring para saber cuándo usarla y los tipos para saber qué pasarle.
Las cifras fijas (factor de créditos y tabla de notas) son las de
docs/normativa_intercambio.md; si cambian los documentos, cámbialas aquí.
"""
from langchain_core.tools import tool

# Factor de la normativa: 1 crédito US = 2 ECTS
ECTS_POR_CREDITO_US = 2.0

# Tabla de la normativa: (nota española mínima del tramo, letra, GPA, punto medio)
TABLA_NOTAS = [
    (9.0, "A", 4.0, 9.5),
    (8.0, "A-", 3.7, 8.5),
    (7.0, "B+", 3.3, 7.5),
    (6.0, "B", 3.0, 6.5),
    (5.0, "C", 2.0, 5.5),
    (0.0, "F", 0.0, 2.5),
]


@tool
def convertir_creditos(
    creditos: float, origen: str, destino: str, factor: float = ECTS_POR_CREDITO_US
) -> str:
    """Convierte créditos entre ECTS y créditos US de Lakeshore University.
    origen y destino: "ECTS" o "US". factor: ECTS que vale 1 crédito US
    (la normativa fija 2.0; cámbialo solo si los documentos dan otro)."""
    origen, destino = origen.strip().upper(), destino.strip().upper()
    if {origen, destino} != {"ECTS", "US"}:
        return "origen y destino deben ser 'ECTS' y 'US' (uno de cada)."
    if origen == "US":
        resultado = creditos * factor
    else:
        resultado = creditos / factor
    return f"{creditos:g} créditos {origen} = {round(resultado, 2):g} {destino} (factor 1 US = {factor:g} ECTS)"


@tool
def convertir_nota(nota: float, escala_origen: str = "ES") -> str:
    """Convierte notas con la tabla de la normativa de Lakeshore University.
    escala_origen="ES": nota española sobre 10 -> letra y GPA de Lakeshore.
    escala_origen="GPA": GPA de Lakeshore (0 a 4) -> nota española (punto medio del tramo)."""
    escala = escala_origen.strip().upper()
    if escala in ("ES", "ESPAÑOLA", "ESPANOLA", "10"):
        if not 0 <= nota <= 10:
            return "La nota española debe estar entre 0 y 10."
        for minimo, letra, gpa, _ in TABLA_NOTAS:
            if nota >= minimo:
                return f"{nota:g}/10 = {letra} (GPA {gpa:.1f}) en Lakeshore"
    if escala in ("GPA", "US", "LAKESHORE"):
        if not 0 <= nota <= 4:
            return "El GPA debe estar entre 0 y 4."
        # El tramo con el GPA más alto que no supere la nota dada
        for _, letra, gpa, medio in TABLA_NOTAS:
            if nota >= gpa:
                return f"GPA {nota:g} = {letra} = {medio:g}/10 en la universidad de origen"
    return "escala_origen debe ser 'ES' (sobre 10) o 'GPA' (sobre 4)."


@tool
def presupuesto_mensual(
    alquiler: float,
    comida: float = 0,
    transporte: float = 0,
    seguro: float = 0,
    otros: float = 0,
    meses: int = 5,
) -> str:
    """Suma los gastos mensuales (en $) y calcula el total del periodo.
    Pasa todos los gastos que conozcas; los que no se sepan valen 0.
    seguro: cuota MENSUAL (si el documento da el coste por semestre, divídelo
    entre los meses). meses: duración del periodo (un semestre cuenta como 5)."""
    mensual = alquiler + comida + transporte + seguro + otros
    total = mensual * meses
    return (
        f"Mensual: {mensual:,.2f} $ (alquiler {alquiler:g} + comida {comida:g} + "
        f"transporte {transporte:g} + seguro {seguro:g} + otros {otros:g}). "
        f"Total {meses} meses: {total:,.2f} $"
    )
