import pandas as pd


def run_part7_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame | str]:
    total_mesas = int(df["MESA_DE_VOTACION"].nunique())
    total_ubigeos = int(df["UBIGEO"].nunique())
    total_departamentos = int(df["DEPARTAMENTO"].nunique())

    region_df = (
        df.groupby("DEPARTAMENTO")[["VOTOS_P1", "VOTOS_P2", "N_CVAS", "N_ELEC_HABIL"]]
        .sum()
        .sort_values(["N_CVAS"], ascending=False)
    )
    region_df["PARTICIPACION"] = (
        region_df["N_CVAS"] / region_df["N_ELEC_HABIL"].replace(0, 1)
    ).round(4)
    region_df["TOTAL_VALIDOS"] = region_df["VOTOS_P1"] + region_df["VOTOS_P2"]

    concepts_df = pd.DataFrame(
        [
            {
                "Concepto": "Qubits",
                "Descripcion": "Son las unidades basicas de informacion cuantica. A diferencia de un bit clasico, un qubit puede representar mas de un estado de manera simultanea.",
            },
            {
                "Concepto": "Superposicion",
                "Descripcion": "Permite que un qubit exista en una combinacion de estados al mismo tiempo, lo que abre la posibilidad de explorar muchas alternativas en paralelo.",
            },
            {
                "Concepto": "Paralelismo cuantico",
                "Descripcion": "Es la capacidad teorica de ciertos algoritmos cuanticos para procesar gran cantidad de combinaciones de forma mucho mas eficiente que los metodos clasicos en problemas especificos.",
            },
        ]
    )

    top_regions_df = (
        region_df[["TOTAL_VALIDOS", "PARTICIPACION"]]
        .head(8)
        .reset_index()
        .rename(
            columns={
                "DEPARTAMENTO": "Departamento",
                "TOTAL_VALIDOS": "Votos validos",
                "PARTICIPACION": "Participacion",
            }
        )
    )
    top_regions_df["Participacion"] = (top_regions_df["Participacion"] * 100).round(2)

    applications_df = pd.DataFrame(
        [
            {
                "Aplicacion potencial": "Busqueda de patrones masivos",
                "Analisis": (
                    f"Con {total_mesas} mesas y {total_ubigeos} ubigeos, una aproximacion cuantica podria "
                    "acelerar la exploracion de combinaciones y patrones territoriales complejos."
                ),
            },
            {
                "Aplicacion potencial": "Optimizacion de simulaciones electorales",
                "Analisis": (
                    "Algoritmos cuanticos de optimizacion podrian ser utiles para evaluar escenarios "
                    "de redistribucion, simulacion o asignacion de recursos en procesos a gran escala."
                ),
            },
            {
                "Aplicacion potencial": "Segmentacion avanzada de regiones",
                "Analisis": (
                    f"El analisis de {total_departamentos} departamentos con volumenes muy distintos de votos "
                    "podria beneficiarse en el futuro de metodos cuanticos para clustering y deteccion de anomalias."
                ),
            },
        ]
    )

    interpretation_df = pd.DataFrame(
        [
            {
                "Hallazgo": "Escala del problema",
                "Interpretacion": (
                    f"El dataset electoral ya maneja decenas de miles de mesas, por lo que el analisis masivo "
                    "es un escenario razonable para explorar tecnologias emergentes."
                ),
            },
            {
                "Hallazgo": "Estado actual",
                "Interpretacion": (
                    "Hoy en dia, la computacion cuantica no reemplaza el analisis clasico usado en la aplicacion, "
                    "pero sirve como marco conceptual para futuros sistemas de alto rendimiento."
                ),
            },
            {
                "Hallazgo": "Valor practico futuro",
                "Interpretacion": (
                    "Su mayor aporte potencial estaria en optimizacion, simulacion y deteccion de patrones complejos "
                    "sobre grandes volumenes de informacion electoral."
                ),
            },
        ]
    )

    summary = {
        "mesas": f"{total_mesas:,}".replace(",", "."),
        "ubigeos": f"{total_ubigeos:,}".replace(",", "."),
        "departamentos": str(total_departamentos),
    }

    return {
        "concepts_df": concepts_df,
        "top_regions_df": top_regions_df,
        "applications_df": applications_df,
        "interpretation_df": interpretation_df,
        "summary": summary,
    }
