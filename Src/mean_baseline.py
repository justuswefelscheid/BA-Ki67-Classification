"""Vergleicht mean-gepoolte Slide-Features mittels wiederholter Kreuzvalidierung.

Für GigaPath, UNI v2 und Virchow2 werden die Patch-Features jedes Slides durch
Mean Pooling zu einem Slide-Level-Featurevektor zusammengefasst. Anschließend
werden die Ki-67-Klassen mit standardisierten Features und logistischer
Regression vorhergesagt.

Die Auswertung erfolgt mit 50 Wiederholungen einer stratifizierten
5-Fold-Cross-Validation. Alle Foundation-Modelle und die Dummy-Baseline
verwenden dieselben Trainings- und Testaufteilungen.

Neben den einzelnen Fold- und OOF-Ergebnissen werden die Kennzahlen jeder
vollständigen Wiederholung sowie deren Zusammenfassung als CSV gespeichert.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/ubuntu/applications/ki67-thesis")

LABEL_FILE = ROOT / "data/labels/ki67_clean_20_HE.csv"
RESULT_DIR = ROOT / "results/meanpool_repeated_cv"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIRS = {
    "gigapath": ROOT / "trident_runs/HE_3models/20x_256px_0px_overlap/features_gigapath",
    "uni_v2": ROOT / "trident_runs/HE_3models/20x_256px_0px_overlap/features_uni_v2",
    "virchow2": ROOT / "trident_runs/HE_3models/20x_224px_0px_overlap/features_virchow2",
}

N_SPLITS = 5
N_REPEATS = 50
RANDOM_STATE = 42


labels = pd.read_csv(LABEL_FILE)

class_map = {
    "low": 0,
    "intermediate": 1,
    "high": 2,
}

# Numerische Kodierung der drei Ki-67-Klassen.
y = labels["ki67_class"].map(class_map).to_numpy().astype(int)


cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)

# Alle Modelle und die Dummy-Baseline verwenden dieselben CV-Splits.
# Für die Split-Erzeugung werden nur die Klassenlabels benötigt.
splits = list(cv.split(np.zeros(len(y)), y))


slide_features = {}

for model_name, feature_dir in FEATURE_DIRS.items():

    model_features = []

    for he_id in labels["he_id"]:

        h5_path = feature_dir / f"{Path(he_id).stem}.h5"

        with h5py.File(h5_path, "r") as f:
            patch_features = f["features"][:]

        # Mean Pooling erzeugt aus allen Patch-Features einen Vektor pro Slide.
        model_features.append(
            patch_features.mean(axis=0)
        )

    slide_features[model_name] = np.stack(model_features)


fold_rows = []
repeat_rows = []
oof_frames = []


for model_name in FEATURE_DIRS:

    X = slide_features[model_name]

    # Sammelt die Out-of-Fold-Vorhersagen einer vollständigen 5-Fold-Wiederholung.
    repeat_predictions = [
        np.zeros(len(y), dtype=int)
        for _ in range(N_REPEATS)
    ]

    # Ebenfalls Train Balanced Accuracy sammeln.
    train_balanced_accuracies = [
    []
    for _ in range(N_REPEATS)
    ]  

    for split_number, (train_idx, test_idx) in enumerate(splits):

        # Ordnet jeden Split seiner Wiederholung und seinem Fold zu.
        repeat = split_number // N_SPLITS
        fold = split_number % N_SPLITS

        # Die Standardisierung wird innerhalb jedes Folds nur auf den
        # Trainingsdaten gelernt und anschließend auf die Testdaten angewendet.
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

        train_predictions = classifier.predict(
            X[train_idx]
        )

        train_balanced_accuracies[repeat].append(
            balanced_accuracy_score(
                y[train_idx],
                train_predictions,
            )       
        )

        predictions = classifier.predict(
            X[test_idx]
        )       

        repeat_predictions[repeat][test_idx] = predictions

        oof_frames.append(pd.DataFrame({
            "model": model_name,
            "repeat": repeat + 1,
            "fold": split_number % N_SPLITS + 1,
            "sample_index": test_idx,
            "he_id": labels.iloc[test_idx]["he_id"].to_numpy(),
            "y_true": y[test_idx],
            "y_pred": predictions,
        }))

        fold_rows.append({
            "model": model_name,
            "repeat": repeat + 1,
            "fold": fold + 1,
            "accuracy": accuracy_score(
                y[test_idx],
                predictions,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y[test_idx],
                predictions,
            ),
            "macro_f1": f1_score(
                y[test_idx],
                predictions,
                average="macro",
                zero_division=0,
            ),
        })

    # Die Kennzahlen einer Wiederholung werden aus allen zusammengeführten
    # OOF-Vorhersagen berechnet, nicht aus dem Mittelwert der Fold-Kennzahlen.
    for repeat in range(N_REPEATS):

        predictions = repeat_predictions[repeat]

        repeat_rows.append({
            "model": model_name,
            "repeat": repeat + 1,
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
            "train_balanced_accuracy": np.mean(
                train_balanced_accuracies[repeat]
            ),
        })


# Majority-Class-Baseline auf denselben CV-Splits wie die Foundation-Modelle.
dummy_predictions = [
    np.zeros(len(y), dtype=int)
    for _ in range(N_REPEATS)
]

for split_number, (train_idx, test_idx) in enumerate(splits):

    repeat = split_number // N_SPLITS

    dummy = DummyClassifier(strategy="most_frequent")

    dummy.fit(
        np.zeros((len(train_idx), 1)),
        y[train_idx],
    )

    predictions = dummy.predict(
        np.zeros((len(test_idx), 1))
    )

    dummy_predictions[repeat][test_idx] = predictions

    oof_frames.append(pd.DataFrame({
        "model": "dummy",
        "repeat": repeat + 1,
        "fold": split_number % N_SPLITS + 1,
        "sample_index": test_idx,
        "he_id": labels.iloc[test_idx]["he_id"].to_numpy(),
        "y_true": y[test_idx],
        "y_pred": predictions,
    }))


for repeat in range(N_REPEATS):

    predictions = dummy_predictions[repeat]

    repeat_rows.append({
        "model": "dummy",
        "repeat": repeat + 1,
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


fold_df = pd.DataFrame(fold_rows)

# Speichert jede OOF-Vorhersage zusammen mit Slide, Fold und Wiederholung.
pd.concat(oof_frames, ignore_index=True).to_csv(
    RESULT_DIR / "oof_predictions.csv", index=False,
)

repeat_df = pd.DataFrame(repeat_rows)

fold_df.to_csv(
    RESULT_DIR / "fold_metrics.csv",
    index=False,
)

repeat_df.to_csv(
    RESULT_DIR / "repeat_metrics.csv",
    index=False,
)


summary_rows = []

for model_name in ["dummy", "gigapath", "uni_v2", "virchow2"]:

    model_results = repeat_df[
        repeat_df["model"] == model_name
    ]

    row = {
        "model": model_name,
    }

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]:

        values = model_results[metric].to_numpy()

        # Beschreibt die Verteilung der Kennzahl über die 50 Wiederholungen.
        # Die Quantile sind keine Konfidenzintervalle des Mittelwerts.
        row[f"{metric}_mean"] = values.mean()

        # Stichprobenstandardabweichung über die Wiederholungen.
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
        f"{row['balanced_accuracy_mean']:.3f} "
        f"± {row['balanced_accuracy_std']:.3f}"
    )