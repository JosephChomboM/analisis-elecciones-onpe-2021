from pathlib import Path

import pandas as pd
import streamlit as st

from interface_backend import build_interface_results, get_region_options
from ml_backend import run_part4_analysis, run_part5_analysis
from quantum_backend import run_part7_analysis

DATA_FILE = Path(__file__).parent / "data" / "Resultados_2da_vuelta.csv"
CSV_COLUMNS = [
    "UBIGEO",
    "DEPARTAMENTO",
    "PROVINCIA",
    "DISTRITO",
    "TIPO_ELECCION",
    "MESA_DE_VOTACION",
    "DESCRIP_ESTADO_ACTA",
    "TIPO_OBSERVACION",
    "N_CVAS",
    "N_ELEC_HABIL",
    "VOTOS_P1",
    "VOTOS_P2",
    "VOTOS_VB",
    "VOTOS_VN",
    "VOTOS_VI",
    "EXTRA",
]
NUMERIC_COLUMNS = [
    "N_CVAS",
    "N_ELEC_HABIL",
    "VOTOS_P1",
    "VOTOS_P2",
    "VOTOS_VB",
    "VOTOS_VN",
    "VOTOS_VI",
]


st.set_page_config(
    page_title="Resultados Electorales ONPE 2021",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data
def load_electoral_data() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_FILE,
        encoding="latin1",
        sep=";",
        engine="python",
        names=CSV_COLUMNS,
        skiprows=1,
        dtype=str,
    )

    # El archivo trae una ultima columna vacia en los registros de datos.
    df = df.drop(columns=["EXTRA"])
    df["MESA_DE_VOTACION"] = df["MESA_DE_VOTACION"].astype(str).str.zfill(6)

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    return df


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def style_table(df: pd.DataFrame):
    return (
        df.style.hide(axis="index")
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#dbeafe"),
                        ("color", "#0f172a"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
        .set_properties(**{"background-color": "#ffffff", "color": "#0f172a"})
    )


@st.cache_data
def build_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    department_df = (
        df.groupby("DEPARTAMENTO")[["VOTOS_P1", "VOTOS_P2", "VOTOS_VB", "VOTOS_VN"]]
        .sum()
        .sort_index()
    )
    department_df["VOTOS_VALIDOS"] = department_df["VOTOS_P1"] + department_df["VOTOS_P2"]
    department_df["DIFERENCIA"] = (department_df["VOTOS_P1"] - department_df["VOTOS_P2"]).abs()
    department_df["GANADOR"] = department_df.apply(
        lambda row: (
            "Candidato 1"
            if row["VOTOS_P1"] > row["VOTOS_P2"]
            else "Candidato 2" if row["VOTOS_P2"] > row["VOTOS_P1"] else "Empate"
        ),
        axis=1,
    )
    return department_df.sort_values("VOTOS_VALIDOS", ascending=False)


@st.cache_data
def get_part4_results(df: pd.DataFrame) -> dict[str, object]:
    return run_part4_analysis(df)


@st.cache_data
def get_part5_results(df: pd.DataFrame) -> dict[str, object]:
    return run_part5_analysis(df)


@st.cache_data
def get_part7_results(df: pd.DataFrame) -> dict[str, object]:
    return run_part7_analysis(df)


@st.cache_data
def get_region_list(df: pd.DataFrame) -> list[str]:
    return get_region_options(df)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14, 165, 233, 0.14), transparent 30%),
                radial-gradient(circle at top right, rgba(251, 191, 36, 0.12), transparent 22%),
                linear-gradient(180deg, #fefce8 0%, #f8fafc 38%, #eff6ff 100%);
            color: #0f172a;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #12314a 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .stMarkdown, .stText, p, label, li, span {
            color: #0f172a;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f766e 0%, #155e75 55%, #1d4ed8 100%);
            color: #ffffff;
            padding: 2rem;
            border-radius: 22px;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.25rem;
        }

        .hero-card h1 {
            margin: 0 0 0.5rem 0;
            font-size: 2.3rem;
        }

        .hero-card p {
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.22);
            backdrop-filter: blur(6px);
            border-radius: 18px;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }

        .section-card h3 {
            margin: 0 0 0.5rem 0;
            color: #0f172a;
        }

        .section-card p, .section-card li {
            color: #334155;
            line-height: 1.6;
        }

        .stDataFrame, .stTable {
            background: rgba(255, 255, 255, 0.92);
            border-radius: 16px;
        }

        .stat-card {
            background: linear-gradient(180deg, #ffffff 0%, #eff6ff 100%);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
            margin-bottom: 0.85rem;
        }

        .stat-label {
            color: #475569;
            font-size: 0.92rem;
            margin-bottom: 0.45rem;
        }

        .stat-value {
            color: #0f172a;
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.1;
        }

        .mini-tag {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.12);
            color: #0f766e;
            font-size: 0.85rem;
            font-weight: 600;
        }

        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
        }

        div[data-testid="metric-container"] label {
            color: #475569;
        }

        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #0f172a;
        }

        [data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 16px;
        }

        [data-testid="stDataFrame"] * {
            color: #0f172a !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Resultados Electorales ONPE 2021</h1>
            <p>
                Sistema de analisis, visualizacion y despliegue de informacion por mesa
                de la Segunda Eleccion Presidencial 2021. La aplicacion esta organizada
                para seguir creciendo por etapas dentro del mismo proyecto.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_cards(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_home() -> None:
    render_hero()

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown(
            """
            <div class="section-card">
                <h3>Resumen del proyecto</h3>
                <p>
                    Este espacio centraliza el trabajo del curso sobre datos electorales de la ONPE.
                    Aqui iremos incorporando carga de datos, visualizaciones, analisis y despliegue
                    sin cambiar la estructura general de la aplicacion.
                </p>
                <span class="mini-tag">Git y GitHub</span>
                <span class="mini-tag">Streamlit</span>
                <span class="mini-tag">CSV oficial</span>
                <span class="mini-tag">Analisis electoral</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            """
            <div class="section-card">
                <h3>Estado actual</h3>
                <p>
                    La aplicacion ya cuenta con una base visual mas ordenada y una seccion
                    dedicada al procesamiento de datos electorales oficiales.
                </p>
                <p>
                    Usa la barra lateral para navegar entre apartados y seguir ampliando el proyecto.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_placeholder(title: str, description: str) -> None:
    render_hero()
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_section(df: pd.DataFrame) -> None:
    render_hero()

    st.markdown(
        """
        <div class="section-card">
            <h3>Datos electorales publicos</h3>
            <p>
                Se cargo el dataset oficial en formato CSV desde la carpeta <code>data/</code>.
                Durante el proceso se corrigio la estructura del archivo eliminando una columna vacia,
                se completaron valores numericos faltantes con 0 y se normalizo la columna de mesa de votacion.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "Fuente oficial del dataset: "
        "[Datos Abiertos - Resultados por mesa de las Elecciones Presidenciales 2021, "
        "segunda vuelta](https://www.datosabiertos.gob.pe/dataset/"
        "resultados-por-mesa-de-las-elecciones-presidenciales-2021-segunda-vuelta-"
        "oficina-nacional-de)"
    )

    total_valid_votes = int((df["VOTOS_P1"] + df["VOTOS_P2"]).sum())

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Numero de mesas", format_int(df["MESA_DE_VOTACION"].nunique()))
    metric_2.metric("Ubigeos unicos", format_int(df["UBIGEO"].nunique()))
    metric_3.metric("Departamentos", format_int(df["DEPARTAMENTO"].nunique()))

    vote_1, vote_2, vote_3 = st.columns(3)
    vote_1.metric("Votos candidato 1", format_int(int(df["VOTOS_P1"].sum())))
    vote_2.metric("Votos candidato 2", format_int(int(df["VOTOS_P2"].sum())))
    vote_3.metric("Votos validos", format_int(total_valid_votes))

    vote_4, vote_5, vote_6 = st.columns(3)
    vote_4.metric("Votos en blanco", format_int(int(df["VOTOS_VB"].sum())))
    vote_5.metric("Votos nulos", format_int(int(df["VOTOS_VN"].sum())))
    vote_6.metric("Votos observados/otros", format_int(int(df["VOTOS_VI"].sum())))

    st.subheader("Descripcion del dataset")
    summary_df = pd.DataFrame(
        [
            {"Campo": "Numero de mesas", "Descripcion": format_int(df["MESA_DE_VOTACION"].nunique())},
            {"Campo": "Ubigeo", "Descripcion": f"{format_int(df['UBIGEO'].nunique())} codigos unicos"},
            {
                "Campo": "Votos por candidato",
                "Descripcion": (
                    f"Candidato 1: {format_int(int(df['VOTOS_P1'].sum()))} | "
                    f"Candidato 2: {format_int(int(df['VOTOS_P2'].sum()))}"
                ),
            },
            {
                "Campo": "Votos validos, nulos y en blanco",
                "Descripcion": (
                    f"Validos: {format_int(total_valid_votes)} | "
                    f"Nulos: {format_int(int(df['VOTOS_VN'].sum()))} | "
                    f"Blanco: {format_int(int(df['VOTOS_VB'].sum()))}"
                ),
            },
        ]
    )
    st.dataframe(style_table(summary_df), use_container_width=True)

    st.subheader("Vista previa de los datos")
    preview_columns = [
        "UBIGEO",
        "DEPARTAMENTO",
        "PROVINCIA",
        "DISTRITO",
        "MESA_DE_VOTACION",
        "VOTOS_P1",
        "VOTOS_P2",
        "VOTOS_VB",
        "VOTOS_VN",
    ]
    st.dataframe(style_table(df[preview_columns].head(10)), use_container_width=True)


def render_visualizations(df: pd.DataFrame) -> None:
    render_hero()

    department_df = build_department_summary(df)
    total_p1 = int(df["VOTOS_P1"].sum())
    total_p2 = int(df["VOTOS_P2"].sum())
    total_valid = total_p1 + total_p2
    overall_winner = "Candidato 1" if total_p1 > total_p2 else "Candidato 2"
    overall_diff = abs(total_p1 - total_p2)
    winner_counts = department_df["GANADOR"].value_counts()
    top_turnout = department_df.index[0]
    top_margin = department_df["DIFERENCIA"].idxmax()

    st.markdown(
        """
        <div class="section-card">
            <h3>Visualizacion de resultados electorales</h3>
            <p>
                En este apartado se representan los resultados de forma clara mediante
                graficos de barras, distribucion por departamento y comparaciones directas
                entre candidatos para facilitar la lectura del comportamiento electoral.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_1, tab_2, tab_3, tab_4 = st.tabs(
        ["Barras", "Distribucion por region", "Comparacion", "Interpretacion"]
    )

    with tab_1:
        st.subheader("Votos nacionales por candidato")
        candidate_chart = pd.DataFrame(
            {
                "Candidato": ["Candidato 1", "Candidato 2"],
                "Votos": [total_p1, total_p2],
            }
        ).set_index("Candidato")
        st.bar_chart(candidate_chart)
        st.caption(
            f"El total nacional de votos validos registrados es {format_int(total_valid)}."
        )

    with tab_2:
        st.subheader("Distribucion de votos validos por departamento")
        region_chart = department_df[["VOTOS_VALIDOS"]].head(15)
        st.bar_chart(region_chart)
        st.caption(
            "Se muestran los 15 departamentos con mayor volumen de votos validos "
            "para facilitar la comparacion visual."
        )

    with tab_3:
        st.subheader("Comparacion de resultados por departamento")
        selected_department = st.selectbox(
            "Selecciona un departamento",
            options=department_df.index.tolist(),
        )
        selected_row = department_df.loc[selected_department]

        comparison_col_1, comparison_col_2 = st.columns([1.25, 1])

        with comparison_col_1:
            comparison_chart = pd.DataFrame(
                {
                    "Candidato 1": department_df["VOTOS_P1"].head(10),
                    "Candidato 2": department_df["VOTOS_P2"].head(10),
                }
            )
            st.bar_chart(comparison_chart)
            st.caption(
                "Comparacion entre ambos candidatos en los 10 departamentos "
                "con mayor cantidad de votos validos."
            )

        with comparison_col_2:
            st.metric("Departamento seleccionado", selected_department)
            st.metric("Ganador", selected_row["GANADOR"])
            st.metric("Diferencia de votos", format_int(int(selected_row["DIFERENCIA"])))
            st.metric("Votos validos", format_int(int(selected_row["VOTOS_VALIDOS"])))

            selected_summary = pd.DataFrame(
                [
                    {"Concepto": "Candidato 1", "Votos": format_int(int(selected_row["VOTOS_P1"]))},
                    {"Concepto": "Candidato 2", "Votos": format_int(int(selected_row["VOTOS_P2"]))},
                    {"Concepto": "Votos en blanco", "Votos": format_int(int(selected_row["VOTOS_VB"]))},
                    {"Concepto": "Votos nulos", "Votos": format_int(int(selected_row["VOTOS_VN"]))},
                ]
            )
            st.dataframe(style_table(selected_summary), use_container_width=True)

    with tab_4:
        st.subheader("Interpretacion de resultados obtenidos")
        interpretation_df = pd.DataFrame(
            [
                {
                    "Hallazgo": "Resultado nacional",
                    "Interpretacion": (
                        f"{overall_winner} obtiene la mayor votacion nacional con una diferencia "
                        f"de {format_int(overall_diff)} votos."
                    ),
                },
                {
                    "Hallazgo": "Cobertura territorial",
                    "Interpretacion": (
                        f"Candidato 1 lidera en {winner_counts.get('Candidato 1', 0)} departamentos "
                        f"y Candidato 2 en {winner_counts.get('Candidato 2', 0)}."
                    ),
                },
                {
                    "Hallazgo": "Mayor concentracion de votos",
                    "Interpretacion": (
                        f"{top_turnout} es el departamento con mayor cantidad de votos validos, "
                        "por lo que tiene un peso decisivo en el resultado general."
                    ),
                },
                {
                    "Hallazgo": "Mayor diferencia",
                    "Interpretacion": (
                        f"La distancia mas amplia entre candidatos se observa en {top_margin}, "
                        "lo que evidencia una preferencia regional marcada."
                    ),
                },
            ]
        )
        st.dataframe(style_table(interpretation_df), use_container_width=True)
        st.markdown(
            """
            <div class="section-card">
                <h3>Lectura general</h3>
                <p>
                    Los graficos muestran una contienda nacional ajustada, pero con diferencias
                    regionales bastante marcadas. Algunos departamentos concentran gran parte del
                    volumen de votos, mientras que otros destacan por la amplitud de la ventaja de
                    uno de los candidatos.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_machine_learning(df: pd.DataFrame) -> None:
    render_hero()
    ml_results = get_part4_results(df)

    st.markdown(
        """
        <div class="section-card">
            <h3>Fundamentos de Machine Learning aplicado a datos electorales</h3>
            <p>
                En esta seccion se exploran patrones en los datos usando dos enfoques:
                clasificacion para predecir la tendencia de voto por mesa y agrupamiento
                no supervisado para detectar regiones con comportamientos electorales similares.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_stat_cards(
        [
            ("Tipo de problema", "Clasificacion"),
            ("Clase positiva", f"{ml_results['baseline'] * 100:.2f}%"),
            ("Accuracy", f"{ml_results['accuracy'] * 100:.2f}%"),
        ]
    )

    tab_1, tab_2, tab_3, tab_4 = st.tabs(
        ["Clasificacion", "Agrupamiento", "Evaluacion", "Interpretacion"]
    )

    with tab_1:
        st.subheader("Prediccion simplificada de tendencia de voto")
        st.markdown(
            """
            Se utilizo una regresion logistica para estimar si una mesa favorece al
            Candidato 1 o al Candidato 2. Las variables empleadas fueron departamento,
            ciudadanos habilitados, ciudadanos que votaron, votos en blanco, votos nulos
            y nivel de participacion.
            """
        )
        confusion_chart = pd.DataFrame(
            {
                "Valor": [5987, 1472, 845, 8994],
            },
            index=[
                "Real C1 / Pred C1",
                "Real C1 / Pred C2",
                "Real C2 / Pred C1",
                "Real C2 / Pred C2",
            ],
        )
        chart_col, table_col = st.columns([1.15, 0.85])
        with chart_col:
            confusion_chart = pd.DataFrame(
                {
                    "Valor": ml_results["confusion_df"][
                        ["Predicho: Candidato 1", "Predicho: Candidato 2"]
                    ]
                    .to_numpy()
                    .flatten()
                    .tolist()
                },
                index=[
                    "Real C1 / Pred C1",
                    "Real C1 / Pred C2",
                    "Real C2 / Pred C1",
                    "Real C2 / Pred C2",
                ],
            )
            st.bar_chart(confusion_chart)
        with table_col:
            st.dataframe(style_table(ml_results["confusion_df"]), use_container_width=True)

    with tab_2:
        st.subheader("Agrupamiento basico por departamento")
        st.markdown(
            """
            Se aplico K-Means con 3 clusters para agrupar departamentos segun porcentaje
            de voto por candidato, votos en blanco, votos nulos y participacion.
            """
        )
        cluster_col_1, cluster_col_2 = st.columns([1, 1.25])
        with cluster_col_1:
            render_stat_cards([("Silhouette score", f"{ml_results['silhouette']:.3f}")])
            cluster_profile_chart = ml_results["cluster_profile"].copy()
            cluster_profile_chart["% Promedio Candidato 1"] = (
                cluster_profile_chart["% Promedio Candidato 1"] * 100
            )
            cluster_profile_chart["% Promedio Candidato 2"] = (
                cluster_profile_chart["% Promedio Candidato 2"] * 100
            )
            cluster_profile_chart = cluster_profile_chart.set_index("Cluster")[
                ["% Promedio Candidato 1", "% Promedio Candidato 2"]
            ]
            st.bar_chart(cluster_profile_chart)
            st.dataframe(style_table(ml_results["cluster_profile"]), use_container_width=True)
        with cluster_col_2:
            cluster_count_chart = (
                ml_results["cluster_profile"][["Cluster", "Departamentos"]]
                .set_index("Cluster")
                .rename(columns={"Departamentos": "Cantidad"})
            )
            st.bar_chart(cluster_count_chart)
            st.dataframe(
                style_table(ml_results["cluster_assignment"].head(12)),
                use_container_width=True,
            )

    with tab_3:
        st.subheader("Evaluacion del modelo")
        evaluation_df = pd.DataFrame(
            [
                {
                    "Metrica": "Accuracy de clasificacion",
                    "Resultado": f"{ml_results['accuracy'] * 100:.2f}%",
                },
                {
                    "Metrica": "Silhouette del agrupamiento",
                    "Resultado": f"{ml_results['silhouette']:.3f}",
                },
                {
                    "Metrica": "Problema identificado",
                    "Resultado": "Clasificacion binaria y agrupamiento no supervisado",
                },
            ]
        )
        evaluation_chart = pd.DataFrame(
            {
                "Valor": [
                    ml_results["accuracy"] * 100,
                    ml_results["silhouette"],
                ]
            },
            index=["Accuracy (%)", "Silhouette"],
        )
        chart_col, table_col = st.columns([1.1, 0.9])
        with chart_col:
            st.bar_chart(evaluation_chart)
        with table_col:
            st.dataframe(style_table(evaluation_df), use_container_width=True)
        st.markdown(
            """
            <div class="section-card">
                <h3>Lectura tecnica</h3>
                <p>
                    El accuracy indica que el modelo puede reconocer correctamente una gran
                    parte de las tendencias por mesa. El silhouette score muestra que los
                    clusters encontrados tienen una separacion aceptable para un analisis exploratorio.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_4:
        st.subheader("Interpretacion de resultados")
        interpretation_df = pd.DataFrame(
            [
                {
                    "Hallazgo": "Clasificacion",
                    "Interpretacion": (
                        f"El modelo obtuvo un accuracy de {ml_results['accuracy'] * 100:.2f}%, "
                        "por lo que funciona como una aproximacion inicial de tendencia de voto."
                    ),
                },
                {
                    "Hallazgo": "Agrupamiento",
                    "Interpretacion": (
                        f"El silhouette score de {ml_results['silhouette']:.3f} sugiere que existen "
                        "patrones regionales identificables, aunque no totalmente separados."
                    ),
                },
                {
                    "Hallazgo": "Aplicacion",
                    "Interpretacion": (
                        "La combinacion de clasificacion y clustering ayuda a detectar comportamientos "
                        "recurrentes en las mesas y diferencias territoriales entre departamentos."
                    ),
                },
            ]
        )
        st.dataframe(style_table(interpretation_df), use_container_width=True)


def render_training_evaluation(df: pd.DataFrame) -> None:
    render_hero()
    part5_results = get_part5_results(df)

    st.markdown(
        """
        <div class="section-card">
            <h3>Entrenamiento y evaluacion</h3>
            <p>
                En esta parte se valida el analisis separando el dataset en entrenamiento
                y prueba, entrenando un modelo basico y evaluando su rendimiento para detectar
                si existe sobreajuste, subajuste o un comportamiento estable.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_stat_cards(
        [
            ("Accuracy entrenamiento", f"{part5_results['train_accuracy'] * 100:.2f}%"),
            ("Accuracy prueba", f"{part5_results['test_accuracy'] * 100:.2f}%"),
            ("Diagnostico", part5_results["fit_status"]),
        ]
    )

    tab_1, tab_2, tab_3, tab_4 = st.tabs(
        ["Division de datos", "Evaluacion", "Ajuste", "Limitaciones"]
    )

    with tab_1:
        st.subheader("Division entre entrenamiento y prueba")
        st.markdown(
            """
            El dataset fue dividido en dos subconjuntos: 80% para entrenamiento y 20%
            para prueba. Esta separacion permite entrenar el modelo con una parte de los
            datos y validar su comportamiento con registros no vistos.
            """
        )
        split_chart = (
            part5_results["split_df"][["Conjunto", "Registros"]]
            .set_index("Conjunto")
            .rename(columns={"Registros": "Cantidad"})
        )
        chart_col, table_col = st.columns([1.1, 0.9])
        with chart_col:
            st.bar_chart(split_chart)
        with table_col:
            st.dataframe(style_table(part5_results["split_df"]), use_container_width=True)

    with tab_2:
        st.subheader("Resultados del modelo")
        metrics_chart_source = part5_results["metrics_df"].copy()
        metrics_chart = (
            metrics_chart_source[
                metrics_chart_source["Metrica"].isin(
                    [
                        "Accuracy entrenamiento",
                        "Accuracy prueba",
                        "Precision",
                        "Recall",
                        "F1-score",
                    ]
                )
            ][["Metrica", "Resultado"]]
            .set_index("Metrica")
        )
        result_col, confusion_col = st.columns([1.05, 0.95])
        with result_col:
            st.bar_chart(metrics_chart)
            st.dataframe(style_table(part5_results["metrics_df"]), use_container_width=True)
        with confusion_col:
            st.markdown("**Matriz de confusion del conjunto de prueba**")
            confusion_chart = pd.DataFrame(
                {
                    "Valor": part5_results["confusion_df"][
                        ["Predicho: Candidato 1", "Predicho: Candidato 2"]
                    ]
                    .to_numpy()
                    .flatten()
                    .tolist()
                },
                index=[
                    "Real C1 / Pred C1",
                    "Real C1 / Pred C2",
                    "Real C2 / Pred C1",
                    "Real C2 / Pred C2",
                ],
            )
            st.bar_chart(confusion_chart)

    with tab_3:
        st.subheader("Identificacion de sobreajuste o subajuste")
        fit_chart = pd.DataFrame(
            {
                "Accuracy": [
                    part5_results["train_accuracy"] * 100,
                    part5_results["test_accuracy"] * 100,
                ]
            },
            index=["Entrenamiento", "Prueba"],
        )
        chart_col, table_col = st.columns([1.05, 0.95])
        with chart_col:
            st.bar_chart(fit_chart)
        with table_col:
            st.dataframe(style_table(part5_results["diagnosis_df"]), use_container_width=True)
        st.markdown(
            """
            <div class="section-card">
                <h3>Lectura del ajuste</h3>
                <p>
                    Si la diferencia entre entrenamiento y prueba fuese muy alta, existiria
                    riesgo de sobreajuste. Si ambas precisiones fueran bajas, hablaríamos de
                    subajuste. En este caso el comportamiento del modelo es estable y adecuado
                    para una validacion inicial.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_4:
        st.subheader("Limitaciones del modelo en contexto electoral")
        st.dataframe(style_table(part5_results["limitations_df"]), use_container_width=True)


def render_quantum_section(df: pd.DataFrame) -> None:
    render_hero()
    quantum_results = get_part7_results(df)

    st.markdown(
        """
        <div class="section-card">
            <h3>Computacion cuantica aplicada</h3>
            <p>
                Esta seccion presenta conceptos basicos de computacion cuantica y analiza
                su posible aplicacion futura en el procesamiento masivo de datos electorales.
                Se trata de una exploracion conceptual, no de una implementacion cuantica real.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_stat_cards(
        [
            ("Mesas analizadas", quantum_results["summary"]["mesas"]),
            ("Ubigeos unicos", quantum_results["summary"]["ubigeos"]),
            ("Departamentos", quantum_results["summary"]["departamentos"]),
        ]
    )

    tab_1, tab_2, tab_3 = st.tabs(
        ["Conceptos", "Aplicaciones potenciales", "Analisis"]
    )

    with tab_1:
        st.subheader("Conceptos clave")
        st.dataframe(style_table(quantum_results["concepts_df"]), use_container_width=True)

    with tab_2:
        st.subheader("Posible uso en datos electorales")
        st.dataframe(style_table(quantum_results["applications_df"]), use_container_width=True)
        st.markdown("**Regiones con mayor volumen de votos para dimensionar el problema**")
        st.dataframe(style_table(quantum_results["top_regions_df"]), use_container_width=True)

    with tab_3:
        st.subheader("Interpretacion")
        st.dataframe(style_table(quantum_results["interpretation_df"]), use_container_width=True)
        st.markdown(
            """
            <div class="section-card">
                <h3>Lectura general</h3>
                <p>
                    En el contexto de la ONPE, la computacion cuantica puede entenderse como
                    una tecnologia emergente con potencial para acelerar tareas de optimizacion,
                    simulacion y analisis de patrones. Sin embargo, para este proyecto su valor
                    es principalmente formativo y prospectivo.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_interface_design(df: pd.DataFrame) -> None:
    render_hero()

    st.markdown(
        """
        <div class="section-card">
            <h3>Oficina Nacional de Procesos Electorales (ONPE)</h3>
            <p>
                Consulta ciudadana de resultados electorales. Esta interfaz esta pensada para
                que cualquier persona pueda seleccionar una region, revisar resultados y leer
                una interpretacion breve sin necesidad de conocimientos tecnicos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filter_col_1, filter_col_2 = st.columns([1.1, 0.9])
    with filter_col_1:
        selected_region = st.selectbox(
            "Selecciona una region",
            options=get_region_list(df),
        )
    with filter_col_2:
        selected_candidate = st.selectbox(
            "Selecciona un candidato",
            options=["Todos", "Candidato 1", "Candidato 2"],
        )

    interface_results = build_interface_results(df, selected_region, selected_candidate)

    render_stat_cards(
        [
            ("Mesas consultadas", interface_results["summary"]["mesas"]),
            ("Votos validos", interface_results["summary"]["validos"]),
            ("Votos del filtro", interface_results["summary"]["seleccionado"]),
        ]
    )
    render_stat_cards(
        [
            ("Mayor participacion", interface_results["summary"]["participacion"]),
            ("Votos en blanco", interface_results["summary"]["blancos"]),
            ("Votos nulos", interface_results["summary"]["nulos"]),
        ]
    )

    tab_1, tab_2, tab_3 = st.tabs(
        ["Resultados", "Interpretacion", "User Flow"]
    )

    with tab_1:
        st.subheader("Visualizacion clara de resultados")
        results_col, detail_col = st.columns([1.2, 0.8])
        with results_col:
            st.bar_chart(interface_results["chart_df"])
        with detail_col:
            st.dataframe(
                style_table(interface_results["comparison_df"]),
                use_container_width=True,
            )

    with tab_2:
        st.subheader("Interpretacion de la consulta")
        st.markdown(
            f"""
            <div class="section-card">
                <h3>Lectura del resultado</h3>
                <p>{interface_results["interpretation"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_3:
        st.subheader("User Flow propuesto")
        st.dataframe(style_table(interface_results["flow_df"]), use_container_width=True)
        flow_cols = st.columns(3)
        flow_cards = [
            ("Seleccion de region", "La persona elige el ambito geografico que desea consultar."),
            ("Visualizacion de resultados", "La interfaz responde con graficos, metricas y comparaciones claras."),
            ("Interpretacion", "Se resume el resultado principal para facilitar la comprension ciudadana."),
        ]
        for column, (title, text) in zip(flow_cols, flow_cards):
            with column:
                st.markdown(
                    f"""
                    <div class="section-card">
                        <h3>{title}</h3>
                        <p>{text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


inject_styles()
df = load_electoral_data()

with st.sidebar:
    st.markdown("## Navegacion")
    selected_section = st.radio(
        "Ir a",
        [
            "Inicio",
            "Datos electorales",
            "Visualizaciones",
            "Machine Learning",
            "Entrenamiento y evaluacion",
            "Computacion cuantica",
            "Interfaz ciudadana",
            "Docker",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Proyecto ONPE 2021")

if selected_section == "Inicio":
    render_home()
elif selected_section == "Datos electorales":
    render_data_section(df)
elif selected_section == "Visualizaciones":
    render_visualizations(df)
elif selected_section == "Machine Learning":
    render_machine_learning(df)
elif selected_section == "Entrenamiento y evaluacion":
    render_training_evaluation(df)
elif selected_section == "Computacion cuantica":
    render_quantum_section(df)
elif selected_section == "Interfaz ciudadana":
    render_interface_design(df)
else:
    render_placeholder(
        "Docker y despliegue",
        "En esta seccion se agregara la parte de contenerizacion y despliegue del sistema "
        "para asegurar reproducibilidad.",
    )
