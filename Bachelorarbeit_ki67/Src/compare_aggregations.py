"""Vergleicht vier verschiedene Aggregationsmethoden, um die Patch-Repräsentationen zusammenzufassen.

Pro Gesamtbild werden entweder nur Durchschnitte oder zusätzliche Verteilungsmerkmale
zur Verteilung der Merkmale genutzt. Drei Modelle liefern die Bildmerkmale.
Für jede Kombination wird die Vorhersage der Ki-67-Klasse in 50 Durchläufen
geprüft. Jeder Durchlauf teilt die Bilder in fünf Gruppen: Vier dienen zum
Lernen, die fünfte zum Prüfen. Jede Gruppe wird einmal zum Prüfen verwendet.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/ubuntu/applications/ki67-thesis")

LABEL_FILE = ROOT / "data/labels/ki67_clean_20_HE.csv"
RESULT_DIR = ROOT / "results/aggregation_comparison"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIRS = {
    "gigapath": ROOT / "trident_runs/HE_3models/20x_256px_0px_overlap/features_gigapath",
    "uni_v2": ROOT / "trident_runs/HE_3models/20x_256px_0px_overlap/features_uni_v2",
    "virchow2": ROOT / "trident_runs/HE_3models/20x_224px_0px_overlap/features_virchow2",
}

AGGREGATIONS = [
    "mean",
    "mean_std",
    "mean_std_median",
    "mean_std_quantiles",
]

N_SPLITS = 5
N_REPEATS = 50
RANDOM_STATE = 42


labels = pd.read_csv(LABEL_FILE)

class_map = {
    "low": 0,
    "intermediate": 1,
    "high": 2,
}

y = labels["ki67_class"].map(class_map).to_numpy().astype(int)


cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)

# Alle Durchläufe nutzen dieselben Gruppen mit annähernd gleichen Klassenanteilen.
splits = list(cv.split(np.zeros(len(y)), y))


slide_features = {}

for model_name, feature_dir in FEATURE_DIRS.items():

    model_features = {
        agg_name: []
        for agg_name in AGGREGATIONS
    }

    for he_id in labels["he_id"]:

        h5_path = feature_dir / f"{Path(he_id).stem}.h5"

        with h5py.File(h5_path, "r") as f:
            patch_features = f["features"][:]

        # Jede Spalte wird über die Bildausschnitte hinweg zusammengefasst.
        mean = np.mean(
            patch_features,
            axis=0,
            dtype=np.float64,
        )

        std = np.std(
            patch_features,
            axis=0,
            dtype=np.float64,
        )

        q25, median, q75 = np.quantile(
            patch_features,
            [0.25, 0.50, 0.75],
            axis=0,
        )

        model_features["mean"].append(
            mean
        )

        model_features["mean_std"].append(
            np.concatenate([
                mean,
                std,
            ])
        )

        model_features["mean_std_median"].append(
            np.concatenate([
                mean,
                std,
                median,
            ])
        )

        model_features["mean_std_quantiles"].append(
            np.concatenate([
                mean,
                std,
                q25,
                median,
                q75,
            ])
        )

    for agg_name in AGGREGATIONS:

        slide_features[(model_name, agg_name)] = np.stack(
            model_features[agg_name]
        )


repeat_rows = []
oof_frames = []


for model_name in FEATURE_DIRS:

    for agg_name in AGGREGATIONS:

        X = slide_features[(model_name, agg_name)]

        # Pro Durchlauf wird für jedes Bild genau eine Prüfvorhersage gesammelt.
        repeat_predictions = [
            np.zeros(len(y), dtype=int)
            for _ in range(N_REPEATS)
        ]

        for split_number, (train_idx, test_idx) in enumerate(splits):

            # Je fünf aufeinanderfolgende Aufteilungen gehören zu einem Durchlauf.
            repeat = split_number // N_SPLITS

            # Die standarfisierung wird ausschließlich auf dem Trainingsfold gelernt und anschließend auf die Testdaten angewendet.
            classifier = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    OneVsRestClassifier(LogisticRegression(
                        C=1.0,
                        solver="liblinear",
                        dual=True,
                        max_iter=2000,
                    )),
                ),
            ])

            classifier.fit(
                X[train_idx],
                y[train_idx],
            )

            predictions = classifier.predict(
                X[test_idx]
            )

            repeat_predictions[repeat][test_idx] = predictions

            oof_frames.append(pd.DataFrame({
                "model": model_name, "aggregation": agg_name,
                "repeat": repeat + 1,
                "fold": split_number % N_SPLITS + 1,
                "sample_index": test_idx,
                "he_id": labels.iloc[test_idx]["he_id"].to_numpy(),
                "y_true": y[test_idx],
                "y_pred": predictions,
            }))

        # Ein Ergebnis wird aus sämtlichen Prüfvorhersagen des Durchlaufs
        # berechnet, nicht aus dem Durchschnitt der fünf Gruppenergebnisse.
        for repeat in range(N_REPEATS):

            predictions = repeat_predictions[repeat]

            repeat_rows.append({
                "model": model_name,
                "aggregation": agg_name,
                "repeat": repeat + 1,
                "feature_dim": X.shape[1],
                "accuracy": accuracy_score(
                    y,
                    predictions,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y,
                    predictions,
                ),
                "macro_f1": f1_score(
                    y,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
            })


# Eine Zeile je Bild, Verfahren und Durchlauf; Klassen: 0=niedrig, 1=mittel, 2=hoch.
pd.concat(oof_frames, ignore_index=True).to_csv(
    RESULT_DIR / "oof_predictions.csv", index=False,
)
repeat_df = pd.DataFrame(repeat_rows)

repeat_df.to_csv(
    RESULT_DIR / "repeat_metrics.csv",
    index=False,
)


summary_rows = []

for model_name in FEATURE_DIRS:

    for agg_name in AGGREGATIONS:

        model_results = repeat_df[
            (repeat_df["model"] == model_name)
            & (repeat_df["aggregation"] == agg_name)
        ]

        row = {
            "model": model_name,
            "aggregation": agg_name,
            "feature_dim": int(
                model_results["feature_dim"].iloc[0]
            ),
        }

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ]:

            values = model_results[metric].to_numpy()

            # Die Streuung und Prozentwerte beschreiben die 50 Ergebnisse.
            # Sie geben keine gesicherten Grenzen für zukünftige Ergebnisse an.
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
            row[f"{metric}_median"] = np.median(values)
            row[f"{metric}_q025"] = np.quantile(values, 0.025)
            row[f"{metric}_q975"] = np.quantile(values, 0.975)

        summary_rows.append(row)


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    RESULT_DIR / "summary.csv",
    index=False,
)


print("\nBalanced Accuracy:")

for _, row in summary.iterrows():

    print(
        f"{row['model']:10s} "
        f"{row['aggregation']:20s} "
        f"{row['balanced_accuracy_mean']:.3f} "
        f"± {row['balanced_accuracy_std']:.3f}"
    )