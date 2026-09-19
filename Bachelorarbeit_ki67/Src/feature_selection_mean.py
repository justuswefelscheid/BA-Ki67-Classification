"""Vergleicht die Feature-Selektion für mean-gepoolte Slide-Features.

Die Patch-Features jedes Slides werden durch Mean Pooling zu einem
Slide-Level-Featurevektor zusammengefasst. Für GigaPath, UNI2-h und Virchow2
wird untersucht, wie sich unterschiedliche Anzahlen ausgewählter Features
gegenüber der Verwendung aller Features verhalten.

Die Auswertung erfolgt mit 50 Wiederholungen einer stratifizierten
5-Fold-Cross-Validation. Innerhalb jeder Wiederholung wird jedes Slide genau
einmal als Testfall verwendet. Feature-Ranking und Standardisierung werden
ausschließlich anhand der jeweiligen Trainingsdaten bestimmt.

Die Out-of-Fold-Vorhersagen werden zusätzlich gespeichert. Balanced Accuracy
mittelt die klassenspezifischen Trefferquoten, Macro F1 den F1-Score über alle
Klassen mit gleicher Gewichtung.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/ubuntu/applications/ki67-thesis")

LABEL_FILE = ROOT / "data/labels/ki67_clean_20_HE.csv"
OUT_DIR = ROOT / "results/feature_selection_multiple_k"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIRS = {
    "gigapath": ROOT / "trident_runs/HE_3models/20x_256px_0px_overlap/features_gigapath",
    "uni2": ROOT / "trident_runs/HE_3models/20x_256px_0px_overlap/features_uni_v2",
    "virchow2": ROOT / "trident_runs/HE_3models/20x_224px_0px_overlap/features_virchow2",
}

K_VALUES = [20, 50, 100, 200, 500, 1000, 1500]
N_SPLITS = 5
N_REPEATS = 50


labels = pd.read_csv(LABEL_FILE)
labels["he_id"] = labels["he_id"].str.replace(".svs", "", regex=False)

class_map = {
    "low": 0,
    "intermediate": 1,
    "high": 2,
}

y = labels["ki67_class"].map(class_map).to_numpy()


def load_features(feature_dir):
    """Berechnet durch Mean Pooling einen Featurevektor pro Slide."""

    slide_features = []

    for he_id in labels["he_id"]:

        with h5py.File(feature_dir / f"{he_id}.h5", "r") as f:
            patch_features = f["features"][:]

        slide_features.append(
            patch_features.mean(axis=0)
        )

    return np.array(slide_features)


def rank_features(X_train, y_train):
    """Sortiert Features nach der Stärke ihrer One-vs-Rest-Koeffizienten."""

    classifier = OneVsRestClassifier(LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=2000,
        random_state=42,
    ))

    classifier.fit(
        X_train,
        y_train,
    )

    # Jede Klasse besitzt ein eigenes logistisches Regressionsmodell.
    coefficients = np.vstack([model.coef_ for model in classifier.estimators_])

    # Die Koeffizienten aller Klassen werden über ihre L2-Norm zusammengefasst.
    importance = np.sqrt(
        np.sum(coefficients ** 2, axis=0)
    )

    return np.argsort(importance)[::-1]


all_features = {
    model_name: load_features(feature_dir)
    for model_name, feature_dir in FEATURE_DIRS.items()
}


# Zählt, wie häufig ein Feature über die CV-Splits hinweg ausgewählt wird.
selection_counts = {
    model_name: {
        k: np.zeros(X.shape[1], dtype=int)
        for k in K_VALUES
    }
    for model_name, X in all_features.items()
}


splitter = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=42,
)

# Alle Modelle und k-Werte verwenden dieselben stratifizierten CV-Splits.
splits = list(
    splitter.split(
        np.zeros(len(y)),
        y,
    )
)


results = []
oof_frames = []


for split_number, (train_idx, test_idx) in enumerate(splits, start=1):

    repeat = (split_number - 1) // N_SPLITS + 1
    fold = (split_number - 1) % N_SPLITS + 1

    y_train = y[train_idx]
    y_test = y[test_idx]

    prepared = {}

    for model_name, X in all_features.items():

        scaler = StandardScaler()

        # Der Scaler wird nur auf den Trainingsdaten des jeweiligen Folds angepasst.
        train_scaled = scaler.fit_transform(
            X[train_idx]
        )

        test_scaled = scaler.transform(
            X[test_idx]
        )

        classifier = OneVsRestClassifier(LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=2000,
            random_state=42,
        ))

        classifier.fit(
            train_scaled,
            y_train,
        )

        predictions = classifier.predict(
            test_scaled
        )

        oof_frames.append(pd.DataFrame({
            "model": model_name, "variant": "full", "k": 0,
            "repeat": repeat,
            "fold": fold,
            "sample_index": test_idx,
            "he_id": labels.iloc[test_idx]["he_id"].to_numpy(),
            "y_true": y[test_idx],
            "y_pred": predictions,
        }))

        full_balacc = balanced_accuracy_score(
            y_test,
            predictions,
        )

        full_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        # Das Feature-Ranking wird ausschließlich aus den Trainingsdaten bestimmt.
        ranking = rank_features(
            train_scaled,
            y_train,
        )

        prepared[model_name] = {
            "train": train_scaled,
            "test": test_scaled,
            "ranking": ranking,
            "full_balacc": full_balacc,
            "full_f1": full_f1,
        }


    for k in K_VALUES:

        row = {
            "split": split_number,
            "k": k,
        }

        for model_name in FEATURE_DIRS:

            train_scaled = prepared[model_name]["train"]
            test_scaled = prepared[model_name]["test"]
            top_k = prepared[model_name]["ranking"][:k]

            selection_counts[model_name][k][top_k] += 1

            classifier = OneVsRestClassifier(LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=2000,
                random_state=42,
            ))

            classifier.fit(
                train_scaled[:, top_k],
                y_train,
            )

            predictions = classifier.predict(
                test_scaled[:, top_k]
            )

            oof_frames.append(pd.DataFrame({
                "model": model_name, "variant": "top_k", "k": k,
                "repeat": repeat,
                "fold": fold,
                "sample_index": test_idx,
                "he_id": labels.iloc[test_idx]["he_id"].to_numpy(),
                "y_true": y[test_idx],
                "y_pred": predictions,
            }))

            row[f"{model_name}_full_balacc"] = (
                prepared[model_name]["full_balacc"]
            )

            row[f"{model_name}_full_f1"] = (
                prepared[model_name]["full_f1"]
            )

            row[f"{model_name}_topk_balacc"] = (
                balanced_accuracy_score(
                    y_test,
                    predictions,
                )
            )

            row[f"{model_name}_topk_f1"] = (
                f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )

        results.append(row)


# Speichert für jede OOF-Vorhersage Slide, Fold und Wiederholung. k=0 kennzeichnet die Variante mit allen Features.
oof_df = pd.concat(oof_frames, ignore_index=True)
oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)


# Die Kennzahlen werden aus den zusammengeführten OOF-Vorhersagen einer vollständigen 5-Fold-Wiederholung berechnet.
repeat_rows = []

for (model_name, repeat), model_oof in oof_df.groupby(["model", "repeat"]):

    full = model_oof[model_oof["variant"] == "full"]

    for k in K_VALUES:

        selected = model_oof[
            (model_oof["variant"] == "top_k")
            & (model_oof["k"] == k)
        ]

        row = {
            "model": model_name,
            "repeat": repeat,
            "k": k,
            "full": balanced_accuracy_score(
                full["y_true"],
                full["y_pred"],
            ),
            "top_k": balanced_accuracy_score(
                selected["y_true"],
                selected["y_pred"],
            ),
        }

        row["full_f1"] = f1_score(
            full["y_true"],
            full["y_pred"],
            average="macro",
            zero_division=0,
        )

        row["top_k_f1"] = f1_score(
            selected["y_true"],
            selected["y_pred"],
            average="macro",
            zero_division=0,
        )

        repeat_rows.append(row)


repeat_df = pd.DataFrame(repeat_rows)

repeat_df.to_csv(
    OUT_DIR / "repeat_metrics.csv",
    index=False,
)


results = pd.DataFrame(results)

results.to_csv(
    OUT_DIR / "multiple_k_results_3models.csv",
    index=False,
)


# Zusammenfassung der OOF-Kennzahlen über alle 50 Wiederholungen.
oof_results = repeat_df.pivot(
    index=["repeat", "k"],
    columns="model",
    values=["full", "top_k", "full_f1", "top_k_f1"],
)

metric_names = {
    "full": "full_balacc",
    "top_k": "topk_balacc",
    "full_f1": "full_f1",
    "top_k_f1": "topk_f1",
}

oof_results.columns = [
    f"{model}_{metric_names[metric]}"
    for metric, model in oof_results.columns
]

oof_results = oof_results.reset_index()


summary = (
    oof_results
    .groupby("k")
    .agg({
        "gigapath_topk_balacc": ["mean", "std"],
        "uni2_topk_balacc": ["mean", "std"],
        "virchow2_topk_balacc": ["mean", "std"],
    })
)

summary.to_csv(
    OUT_DIR / "multiple_k_summary_3models.csv"
)


# Die Auswahlhäufigkeit beschreibt die Stabilität der Feature-Selektion über alle CV-Splits, nicht die Vorhersageleistung eines Features.
frequency_rows = []

for model_name in FEATURE_DIRS:

    for k in K_VALUES:

        for feature, count in enumerate(
            selection_counts[model_name][k]
        ):

            frequency_rows.append({
                "model": model_name,
                "k": k,
                "feature": feature,
                "selected_count": count,
                "selected_percent": count / len(splits) * 100,
            })


frequency = pd.DataFrame(frequency_rows)

frequency.to_csv(
    OUT_DIR / "feature_frequencies_3models.csv",
    index=False,
)


print("\nBalanced Accuracy:")

for model_name in FEATURE_DIRS:

    print(
        f"\n{model_name} full: "
        f"{oof_results[f'{model_name}_full_balacc'].mean():.3f}"
    )

    for k in K_VALUES:

        scores = oof_results[
            oof_results["k"] == k
        ][f"{model_name}_topk_balacc"]

        print(
            f"k={k:4d}: "
            f"{scores.mean():.3f} "
            f"± {scores.std():.3f}"
        )