import pandas as pd


CANDIDATE_COLUMN_MAP = {
    "Candidato 1": "VOTOS_P1",
    "Candidato 2": "VOTOS_P2",
}


def get_region_options(df: pd.DataFrame) -> list[str]:
    return ["Todo el pais"] + sorted(df["DEPARTAMENTO"].dropna().unique().tolist())


def build_interface_results(
    df: pd.DataFrame, selected_region: str, selected_candidate: str
) -> dict[str, object]:
    filtered_df = df if selected_region == "Todo el pais" else df[df["DEPARTAMENTO"] == selected_region]

    group_field = "DEPARTAMENTO" if selected_region == "Todo el pais" else "PROVINCIA"
    grouped = (
        filtered_df.groupby(group_field)[
            ["VOTOS_P1", "VOTOS_P2", "VOTOS_VB", "VOTOS_VN", "N_CVAS", "N_ELEC_HABIL"]
        ]
        .sum()
        .sort_values("N_CVAS", ascending=False)
    )
    grouped["TOTAL_VALIDOS"] = grouped["VOTOS_P1"] + grouped["VOTOS_P2"]
    grouped["PARTICIPACION"] = (
        grouped["N_CVAS"] / grouped["N_ELEC_HABIL"].replace(0, 1)
    ).round(4)

    total_mesas = int(filtered_df["MESA_DE_VOTACION"].nunique())
    total_validos = int(grouped["TOTAL_VALIDOS"].sum()) if not grouped.empty else 0
    total_blancos = int(grouped["VOTOS_VB"].sum()) if not grouped.empty else 0
    total_nulos = int(grouped["VOTOS_VN"].sum()) if not grouped.empty else 0
    total_p1 = int(grouped["VOTOS_P1"].sum()) if not grouped.empty else 0
    total_p2 = int(grouped["VOTOS_P2"].sum()) if not grouped.empty else 0
    top_area = grouped.index[0] if not grouped.empty else "Sin datos"
    top_participation = (
        f"{grouped['PARTICIPACION'].max() * 100:.2f}%" if not grouped.empty else "0.00%"
    )

    if selected_candidate == "Todos":
        chart_df = grouped[["VOTOS_P1", "VOTOS_P2"]].head(10).rename(
            columns={"VOTOS_P1": "Candidato 1", "VOTOS_P2": "Candidato 2"}
        )
        selected_votes = f"C1: {total_p1:,} | C2: {total_p2:,}".replace(",", ".")
    else:
        candidate_column = CANDIDATE_COLUMN_MAP[selected_candidate]
        chart_df = grouped[[candidate_column]].head(10).rename(
            columns={candidate_column: selected_candidate}
        )
        selected_votes = f"{int(grouped[candidate_column].sum()):,}".replace(",", ".")

    comparison_df = (
        grouped[["VOTOS_P1", "VOTOS_P2", "TOTAL_VALIDOS", "PARTICIPACION"]]
        .head(8)
        .reset_index()
        .rename(
            columns={
                group_field: "Ubicacion",
                "VOTOS_P1": "Candidato 1",
                "VOTOS_P2": "Candidato 2",
                "TOTAL_VALIDOS": "Votos validos",
                "PARTICIPACION": "Participacion",
            }
        )
    )
    comparison_df["Participacion"] = (comparison_df["Participacion"] * 100).round(2)
    participation_df = comparison_df[["Ubicacion", "Participacion"]].set_index("Ubicacion")
    vote_share_df = pd.DataFrame(
        {
            "Categoria": ["Candidato 1", "Candidato 2", "Blancos", "Nulos"],
            "Votos": [total_p1, total_p2, total_blancos, total_nulos],
        }
    )

    if total_p1 > total_p2:
        leading_text = "Candidato 1"
        leading_diff = total_p1 - total_p2
    elif total_p2 > total_p1:
        leading_text = "Candidato 2"
        leading_diff = total_p2 - total_p1
    else:
        leading_text = "Empate"
        leading_diff = 0

    interpretation = (
        f"En {selected_region.lower() if selected_region != 'Todo el pais' else 'el ambito nacional'}, "
        f"{leading_text.lower()} presenta la ventaja principal con una diferencia de "
        f"{leading_diff:,} votos validos. La zona con mayor volumen de participacion en la vista "
        f"actual es {top_area}.".replace(",", ".")
    )

    flow_df = pd.DataFrame(
        [
            {
                "Paso": "1. Seleccion de region",
                "Descripcion": "La persona elige un departamento o revisa todo el pais para acotar la consulta.",
            },
            {
                "Paso": "2. Visualizacion de resultados",
                "Descripcion": "La interfaz muestra metricas, graficos y comparaciones del ambito seleccionado.",
            },
            {
                "Paso": "3. Interpretacion",
                "Descripcion": "Se resume quien lidera y que zonas concentran mayor volumen o participacion.",
            },
        ]
    )

    summary = {
        "mesas": f"{total_mesas:,}".replace(",", "."),
        "validos": f"{total_validos:,}".replace(",", "."),
        "seleccionado": selected_votes,
        "participacion": top_participation,
        "blancos": f"{total_blancos:,}".replace(",", "."),
        "nulos": f"{total_nulos:,}".replace(",", "."),
    }

    return {
        "chart_df": chart_df,
        "comparison_df": comparison_df,
        "participation_df": participation_df,
        "vote_share_df": vote_share_df,
        "flow_df": flow_df,
        "interpretation": interpretation,
        "summary": summary,
        "group_label": group_field,
    }
