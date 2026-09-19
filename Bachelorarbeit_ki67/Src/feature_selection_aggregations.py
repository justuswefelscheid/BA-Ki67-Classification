"""Prüft, wie gut eine Reduktion der Features die Ki-67-Klasse vorhersagt.

Für jedes Slide werden die Features der Patches durch fünf
Werte zusammengefasst: Durchschnitt, Streuung, mittlerer Wert sowie die Werte
bei 25 und 75 Prozent der sortierten Daten. Verglichen werden alle Merkmale
und unterschiedlich große Auswahlen. Zu jedem ausgewählten Merkmal bleiben
alle fünf Werte erhalten. In 50 Durchläufen wird jedes Bild genau einmal
geprüft. Jeweils vier von fünf Gruppen dienen zum Lernen, eine zum Prüfen.
Die einzelnen Vorhersagen stehen in ``oof_predictions.csv``, die Kennzahlen
aus allen Bildern je Durchlauf in ``repeat_metrics.csv``.

Die Konsole zeigt die Balanced Accuracy: den Durchschnitt der Trefferquoten
der einzelnen Klassen. Dadurch zählt jede Klasse gleich viel.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/ubuntu/applications/ki67-thesis")

LABEL_FILE = ROOT / "data/labels/ki67_clean_20_HE.csv"
OUT_DIR = ROOT / "results/feature_selection_aggregations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIRS = {
    "gigapath": ROOT / "trident_runs/HE_3models/20x_256px_0px_overlap/features_gigapath",
    "uni2": ROOT / "trident_runs/HE_3models/20x_256px_0px_overlap/features_uni_v2",
    "virchow2": ROOT / "trident_runs/HE_3models/20x_224px_0px_overlap/features_virchow2",
}

K_VALUES = [20, 50, 100, 200, 500, 1000]
N_SPLITS = 5
N_REPEATS = 50
N_STATS = 5


labels = pd.read_csv(LABEL_FILE)
labels["he_id"] = labels["he_id"].str.replace(".svs", "", regex=False)

class_map = {
    "low": 0,
    "intermediate": 1,
    "high": 2,
}

y = labels["ki67_class"].map(class_map).to_numpy()


def load_features(feature_dir):
    """
    Fasst die Bildausschnitte jedes Bildes durch fünf Werte je Merkmal zusammen.
    """

    slide_features = []

    for he_id in labels["he_id"]:

        with h5py.File(feature_dir / f"{he_id}.h5", "r") as f:
            features = f["features"][:]

        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        median = np.median(features, axis=0)
        q25 = np.quantile(features, 0.25, axis=0)
        q75 = np.quantile(features, 0.75, axis=0)

        slide_features.append(
            np.stack([
                mean,
                std,
                median,
                q25,
                q75,
            ])
        )

    return np.array(slide_features)


def rank_dimensions(X, y_train):
    """Sortiert Merkmale nach der Stärke ihrer gelernten Modellgewichte.
    """

    n_dims = X.shape[2]

    classifier = OneVsRestClassifier(LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=2000,
    ))

    classifier.fit(
        X.reshape(len(X), -1),
        y_train,
    )

    # Die Gewichte der einzelnen Klassenmodelle werden gemeinsam ausgewertet.
    coefficients = np.vstack([model.coef_ for model in classifier.estimators_])
    coefficients = coefficients.reshape(
        len(classifier.estimators_),
        N_STATS,
        n_dims,
    )

    # Quadrieren verhindert, dass sich positive und negative Gewichte aufheben.
    importance = np.sqrt(
        np.sum(coefficients ** 2, axis=(0, 1))
    )

    return np.argsort(importance)[::-1]


all_features = {
    model_name: load_features(feature_dir)
    for model_name, feature_dir in FEATURE_DIRS.items()
}


splitter = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=42,
)

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

    for model_name, X in all_features.items():

        train_full = X[train_idx].reshape(len(train_idx), -1)
        test_full = X[test_idx].reshape(len(test_idx), -1)

        scaler = StandardScaler()

        # Die Anpassung der Größen wird nur an Trainingsbildern gelernt.
        train_scaled = scaler.fit_transform(train_full)
        test_scaled = scaler.transform(test_full)

        train_3d = train_scaled.reshape(
            len(train_idx),
            N_STATS,
            X.shape[2],
        )

        test_3d = test_scaled.reshape(
            len(test_idx),
            N_STATS,
            X.shape[2],
        )

        classifier = OneVsRestClassifier(LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=2000,
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

        full_score = balanced_accuracy_score(
            y_test,
            predictions,
        )

        # Die Prüfbilder beeinflussen die Auswahl der Features nicht.
        ranking = rank_dimensions(
            train_3d,
            y_train,
        )

        for k in K_VALUES:

            # Ist k größer als die Featurezahl, werden alle Features genutzt.
            # Pro ausgewähltem Feature werden alle fünf Werte verwendet.
            top_k = ranking[:k]

            train_top_k = train_3d[:, :, top_k].reshape(
                len(train_idx),
                -1,
            )

            test_top_k = test_3d[:, :, top_k].reshape(
                len(test_idx),
                -1,
            )

            classifier = OneVsRestClassifier(LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=2000,
            ))

            classifier.fit(
                train_top_k,
                y_train,
            )

            predictions = classifier.predict(
                test_top_k
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

            top_k_score = balanced_accuracy_score(
                y_test,
                predictions,
            )

            results.append({
                "split": split_number,
                "model": model_name,
                "k": k,
                "full": full_score,
                "top_k": top_k_score,
            })


# Jede Prüfvorhersage bleibt mit Bild, Prüfgruppe und Durchlauf nachvollziehbar.
# k=0 kennzeichnet die Variante mit allen Merkmalen.
oof_df = pd.concat(oof_frames, ignore_index=True)
oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

# Die Kennzahlen werden aus allen Bildern eines Durchlaufs berechnet.
repeat_rows = []
for (model_name, repeat), model_oof in oof_df.groupby(["model", "repeat"]):
    full = model_oof[model_oof["variant"] == "full"]
    for k in K_VALUES:
        selected = model_oof[
            (model_oof["variant"] == "top_k") & (model_oof["k"] == k)
        ]
        row = {
            "model": model_name,
            "repeat": repeat,
            "k": k,
            "full": balanced_accuracy_score(full["y_true"], full["y_pred"]),
            "top_k": balanced_accuracy_score(selected["y_true"], selected["y_pred"]),
        }
        repeat_rows.append(row)

repeat_df = pd.DataFrame(repeat_rows)
repeat_df.to_csv(OUT_DIR / "repeat_metrics.csv", index=False)

results = pd.DataFrame(results)

results.to_csv(
    OUT_DIR / "aggregation_feature_selection_results_3models.csv",
    index=False,
)


print("\nBalanced Accuracy:")

for model_name in FEATURE_DIRS:

    model_results = repeat_df[
        repeat_df["model"] == model_name
    ]

    print(
        f"\n{model_name} full: "
        f"{model_results['full'].mean():.3f}"
    )

    for k in K_VALUES:

        scores = model_results[
            model_results["k"] == k
        ]["top_k"]

        print(
            f"k={k:4d}: "
            f"{scores.mean():.3f} "
            f"± {scores.std():.3f}"
        )
