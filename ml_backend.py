import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "DEPARTAMENTO",
    "N_CVAS",
    "N_ELEC_HABIL",
    "VOTOS_VB",
    "VOTOS_VN",
    "PARTICIPACION",
]


def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    ml_df = df.copy()
    ml_df["PARTICIPACION"] = (
        ml_df["N_CVAS"] / ml_df["N_ELEC_HABIL"].replace(0, 1)
    ).round(4)
    ml_df["GANADOR_MESA"] = (ml_df["VOTOS_P2"] > ml_df["VOTOS_P1"]).astype(int)
    return ml_df


def build_classifier() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["DEPARTAMENTO"]),
            (
                "num",
                StandardScaler(),
                ["N_CVAS", "N_ELEC_HABIL", "VOTOS_VB", "VOTOS_VN", "PARTICIPACION"],
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
def run_part4_analysis(df: pd.DataFrame) -> dict[str, object]:
    ml_df = prepare_ml_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        ml_df[FEATURE_COLUMNS],
        ml_df["GANADOR_MESA"],
        test_size=0.2,
        random_state=42,
        stratify=ml_df["GANADOR_MESA"],
    )

    classifier = build_classifier()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)[:, 1]

    confusion = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_auc = roc_auc_score(y_test, probabilities)

    feature_names = classifier.named_steps["preprocessor"].get_feature_names_out()
    coefficients = classifier.named_steps["model"].coef_[0]
    coefficient_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    )
    coefficient_df["abs_coefficient"] = coefficient_df["coefficient"].abs()
    top_coefficients = (
        coefficient_df.sort_values("abs_coefficient", ascending=False)
        .head(12)[["feature", "coefficient"]]
        .assign(direction=lambda x: np.where(x["coefficient"] > 0, "Favorece Candidato 2", "Favorece Candidato 1"))
    )

    department_df = (
        ml_df.groupby("DEPARTAMENTO")[
            ["VOTOS_P1", "VOTOS_P2", "VOTOS_VB", "VOTOS_VN", "N_CVAS", "N_ELEC_HABIL"]
        ]
        .sum()
        .sort_index()
    )
    department_df["PARTICIPACION"] = (
        department_df["N_CVAS"] / department_df["N_ELEC_HABIL"].replace(0, 1)
    )
    total_valid = (department_df["VOTOS_P1"] + department_df["VOTOS_P2"]).replace(0, 1)
    department_df["PORC_P1"] = department_df["VOTOS_P1"] / total_valid
    department_df["PORC_P2"] = department_df["VOTOS_P2"] / total_valid

    cluster_features = department_df[
        ["PORC_P1", "PORC_P2", "VOTOS_VB", "VOTOS_VN", "PARTICIPACION"]
    ].copy()
    cluster_features[["VOTOS_VB", "VOTOS_VN"]] = StandardScaler().fit_transform(
        cluster_features[["VOTOS_VB", "VOTOS_VN"]]
    )

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    department_df["CLUSTER"] = kmeans.fit_predict(cluster_features)
    silhouette = silhouette_score(cluster_features, department_df["CLUSTER"])
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(cluster_features)
    cluster_projection = pd.DataFrame(
        {
            "Departamento": department_df.index,
            "PC1": pca_coords[:, 0],
            "PC2": pca_coords[:, 1],
            "Cluster": department_df["CLUSTER"].values,
        }
    )

    cluster_profile = (
        department_df.groupby("CLUSTER")[["PORC_P1", "PORC_P2", "PARTICIPACION"]]
        .mean()
        .round(3)
        .reset_index()
    )
    cluster_profile["Departamentos"] = (
        department_df.groupby("CLUSTER").size().reindex(cluster_profile["CLUSTER"]).values
    )
    cluster_profile["Perfil"] = cluster_profile.apply(
        lambda row: (
            "Predominio Candidato 1"
            if row["PORC_P1"] > row["PORC_P2"] + 0.08
            else "Predominio Candidato 2"
            if row["PORC_P2"] > row["PORC_P1"] + 0.08
            else "Competencia equilibrada"
        ),
        axis=1,
    )

    cluster_assignment = department_df[
        ["PORC_P1", "PORC_P2", "PARTICIPACION", "CLUSTER"]
    ].reset_index()
    cluster_assignment["PORC_P1"] = (cluster_assignment["PORC_P1"] * 100).round(2)
    cluster_assignment["PORC_P2"] = (cluster_assignment["PORC_P2"] * 100).round(2)
    cluster_assignment["PARTICIPACION"] = (
        cluster_assignment["PARTICIPACION"] * 100
    ).round(2)
    cluster_assignment = cluster_assignment.rename(
        columns={
            "DEPARTAMENTO": "Departamento",
            "PORC_P1": "% Candidato 1",
            "PORC_P2": "% Candidato 2",
            "PARTICIPACION": "% Participacion",
            "CLUSTER": "Cluster",
        }
    ).sort_values(["Cluster", "Departamento"])

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "baseline": float(ml_df["GANADOR_MESA"].mean()),
        "confusion_matrix": confusion.tolist(),
        "roc_df": roc_df,
        "roc_auc": float(roc_auc),
        "top_coefficients": top_coefficients,
        "cluster_profile": cluster_profile.rename(
            columns={
                "CLUSTER": "Cluster",
                "PORC_P1": "% Promedio Candidato 1",
                "PORC_P2": "% Promedio Candidato 2",
                "PARTICIPACION": "% Participacion promedio",
            }
        ),
        "cluster_assignment": cluster_assignment,
        "cluster_projection": cluster_projection,
        "silhouette": float(silhouette),
    }


def run_part5_analysis(df: pd.DataFrame) -> dict[str, object]:
    ml_df = prepare_ml_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        ml_df[FEATURE_COLUMNS],
        ml_df["GANADOR_MESA"],
        test_size=0.2,
        random_state=42,
        stratify=ml_df["GANADOR_MESA"],
    )

    classifier = build_classifier()
    classifier.fit(X_train, y_train)

    train_predictions = classifier.predict(X_train)
    test_predictions = classifier.predict(X_test)
    test_probabilities = classifier.predict_proba(X_test)[:, 1]

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    generalization_gap = abs(train_accuracy - test_accuracy)

    if generalization_gap > 0.08:
        fit_status = "Sobreajuste"
        fit_explanation = (
            "La diferencia entre entrenamiento y prueba es amplia, por lo que el modelo "
            "aprende demasiado bien el conjunto de entrenamiento y generaliza peor."
        )
    elif train_accuracy < 0.70 and test_accuracy < 0.70:
        fit_status = "Subajuste"
        fit_explanation = (
            "El rendimiento es bajo tanto en entrenamiento como en prueba, lo que sugiere "
            "que el modelo es demasiado simple para capturar mejor los patrones."
        )
    else:
        fit_status = "Ajuste estable"
        fit_explanation = (
            "Las precisiones de entrenamiento y prueba son parecidas, por lo que no se "
            "observa evidencia fuerte de sobreajuste ni de subajuste."
        )

    confusion = confusion_matrix(y_test, test_predictions)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, test_probabilities)
    pr_df = pd.DataFrame({"precision": precision_vals[:-1], "recall": recall_vals[:-1]})

    metrics_df = pd.DataFrame(
        [
            {"Metrica": "Accuracy entrenamiento", "Resultado": round(train_accuracy * 100, 2)},
            {"Metrica": "Accuracy prueba", "Resultado": round(test_accuracy * 100, 2)},
            {"Metrica": "Precision", "Resultado": round(precision_score(y_test, test_predictions) * 100, 2)},
            {"Metrica": "Recall", "Resultado": round(recall_score(y_test, test_predictions) * 100, 2)},
            {"Metrica": "F1-score", "Resultado": round(f1_score(y_test, test_predictions) * 100, 2)},
            {"Metrica": "Brecha entrenamiento-prueba", "Resultado": round(generalization_gap * 100, 2)},
        ]
    )

    split_df = pd.DataFrame(
        [
            {"Conjunto": "Entrenamiento", "Registros": len(X_train), "Porcentaje": round(len(X_train) / len(ml_df) * 100, 2)},
            {"Conjunto": "Prueba", "Registros": len(X_test), "Porcentaje": round(len(X_test) / len(ml_df) * 100, 2)},
        ]
    )

    train_sizes, train_scores, test_scores = learning_curve(
        build_classifier(),
        ml_df[FEATURE_COLUMNS],
        ml_df["GANADOR_MESA"],
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=None,
    )
    learning_curve_df = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_score": train_scores.mean(axis=1),
            "test_score": test_scores.mean(axis=1),
        }
    )

    diagnosis_df = pd.DataFrame(
        [
            {"Aspecto": "Diagnostico", "Detalle": fit_status},
            {"Aspecto": "Interpretacion", "Detalle": fit_explanation},
        ]
    )

    limitations_df = pd.DataFrame(
        [
            {
                "Limitacion": "Variables limitadas",
                "Descripcion": "El modelo usa solo variables agregadas del acta y no incorpora factores sociales, economicos o historicos.",
            },
            {
                "Limitacion": "Prediccion simplificada",
                "Descripcion": "La salida binaria reduce la complejidad del voto real a una sola tendencia ganadora por mesa.",
            },
            {
                "Limitacion": "Contexto electoral",
                "Descripcion": "Los resultados no deben interpretarse como una prediccion causal del comportamiento ciudadano, sino como una aproximacion estadistica.",
            },
        ]
    )

    return {
        "split_df": split_df,
        "metrics_df": metrics_df,
        "confusion_matrix": confusion.tolist(),
        "diagnosis_df": diagnosis_df,
        "limitations_df": limitations_df,
        "pr_df": pr_df,
        "learning_curve_df": learning_curve_df,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "fit_status": fit_status,
    }
