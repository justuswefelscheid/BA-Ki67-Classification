"""Vergleicht Mean-MIL und Attention-MIL.

Beide Verfahren sagen aus gespeicherten Features eine von drei
Ki67-Klassen voraus. Mean-MIL bildet den einfachen Durchschnitt aller
Ausschnitte. Attention-MIL lernt, welche Ausschnitte stärker zählen sollen.
Anschließend berechnet jeweils ein kleines lernendes Modell drei Klassenwerte.

Die Prüfung erfolgt in 50 Durchläufen mit jeweils fünf Bildgruppen. Vier
Gruppen dienen zum Lernen, die übrige zum Prüfen; jede Gruppe wird einmal
geprüft.

Die Konsole zeigt Prüfkennzahlen aus vollständigen OOF-Durchläufen. Die
einzelnen Vorhersagen und Durchlaufkennzahlen werden zusätzlich als CSV
gespeichert. Accuracy ist der Anteil richtig vorhergesagter Slides. Balanced Accuracy
mittelt die Trefferquoten der Klassen mit gleichem Gewicht. Macro F1 bezieht
auch falsche Zuordnungen und übersehene Slides je Klasse ein.
"""

from pathlib import Path
import random

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold


ROOT = Path("/home/ubuntu/applications/ki67-thesis")

LABEL_FILE = ROOT / "data/labels/ki67_clean_20_HE.csv"
OUT_DIR = ROOT / "results/mean_vs_attention_mlp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIRS = {
    "gigapath": ROOT / "trident_runs/HE_3models/20x_256px_0px_overlap/features_gigapath",
    "uni_v2": ROOT / "trident_runs/HE_3models/20x_256px_0px_overlap/features_uni_v2",
    "virchow2": ROOT / "trident_runs/HE_3models/20x_224px_0px_overlap/features_virchow2",
}

MODEL_NAME = "virchow2"
N_REPEATS = 50

EPOCHS = 15
HIDDEN_DIM = 32
ATTENTION_DIM = 32
DROPOUT = 0.3
LR = 1e-4
WEIGHT_DECAY = 1e-3
MAX_TRAIN_PATCHES = 1024
SEED = 42

FEATURE_DIR = FEATURE_DIRS[MODEL_NAME]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


labels = pd.read_csv(LABEL_FILE)
labels["he_id"] = labels["he_id"].str.replace(".svs", "", regex=False)

class_map = {
    "low": 0,
    "intermediate": 1,
    "high": 2,
}

y = labels["ki67_class"].map(class_map).to_numpy()


# Jedes Bild behält seine eigene Liste von Patches. Die Anzahl variiert zwischen Slides, die Featurezahl pro Ausschnitt bleibt gleich.
bags = []

for he_id in labels["he_id"]:

    with h5py.File(FEATURE_DIR / f"{he_id}.h5", "r") as f:
        features = f["features"][:]

    bags.append(
        torch.tensor(
            features,
            dtype=torch.float32,
        )
    )


INPUT_DIM = bags[0].shape[1]


def set_seed(seed):
    """
    Setzt die Seeds der verwendeten Zufallsgeneratoren.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPClassifier(nn.Module):
    """
    Berechnet aus den Merkmalen eines Slides drei Klassenwerte.
    """

    def __init__(self, input_dim):
        """
        Legt die Schichten den Klassifikationsmodelles an.
        """

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 3),
        )

    def forward(self, x):
        """
        Berechnet die Klassenwerte aus bereits zusammengefassten Features.
        """

        return self.layers(x)


class MeanMLP(nn.Module):
    """
    Gewichtet alle Patches gleich und sagt daraus die Bildklasse voraus.
    """

    def __init__(self, input_dim):
        """Erstellt das Modell für gemittelte Patch Features.
        """

        super().__init__()
        self.classifier = MLPClassifier(input_dim)

    def forward(self, x):
        """Mittelt die Patch Features und berechnet drei Klassenwerte.
        """

        slide_embedding = x.mean(dim=0)

        return self.classifier(
            slide_embedding
        )


class AttentionMLP(nn.Module):
    """Lernt die Gewichtung der Patches und sagt daraus die Bildklasse voraus.

    Attributes:
        attention_v (nn.Linear): Lineare Projektion für Tanh-Zweig.
        attention_u (nn.Linear): Lineare Projektion für Sigmoid-Zweig.
        attention_w (nn.Linear): Berechnet einen Attentionsscore je Patch.
        classifier (MLPClassifier): Klassifiziert das aggregierte Slide-Embedding.
    """

    def __init__(self, input_dim):
        """
        Legt die lernbare Gewichtung und die Klassenvorhersage an.
        """

        super().__init__()

        self.attention_v = nn.Linear(
            input_dim,
            ATTENTION_DIM,
        )

        self.attention_u = nn.Linear(
            input_dim,
            ATTENTION_DIM,
        )

        self.attention_w = nn.Linear(
            ATTENTION_DIM,
            1,
        )

        self.classifier = MLPClassifier(
            input_dim
        )

    def forward(self, x):
        """
        Fasst Patches mit gelernten Gewichten zusammen und bewertet Klassen.
        """

        a_v = torch.tanh(
            self.attention_v(x)
        )

        a_u = torch.sigmoid(
            self.attention_u(x)
        )

        scores = self.attention_w(
            a_v * a_u
        ).squeeze(1)

        # Softmax erzeugt positive und negative Gewichte mit Summe 1. So entsteht ein gewichteter Durchschnitt der Patch Features.
        weights = torch.softmax(
            scores,
            dim=0,
        )

        slide_embedding = torch.sum(
            weights.unsqueeze(1) * x,
            dim=0,
        )

        return self.classifier(
            slide_embedding
        )


def train_model(model, train_idx):
    """
    Trainiert das übergebene Modell an den angegebenen Slides.

    Verwendet die global geladenen Merkmale bags, die Klassen y und
    die Einstellungen für Trainingsdauer, Schrittgröße und Rechengerät.
    Die Gewichte des übergebenen Modells werden direkt verändert.
    """

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    for _ in range(EPOCHS):

        model.train()

        for idx in np.random.permutation(train_idx):

            x = bags[idx]

            # Bei vielen Ausschnitten wird bei jedem Besuch eine neue zufällige Auswahl genutzt. Kein Ausschnitt wird dabei mehrfach ausgewählt.
            if len(x) > MAX_TRAIN_PATCHES:

                selected = np.random.choice(
                    len(x),
                    MAX_TRAIN_PATCHES,
                    replace=False,
                )

                x = x[selected]

            x = x.to(device)

            target = torch.tensor(
                [y[idx]],
                dtype=torch.long,
                device=device,
            )

            # Vor jedem Optimierungsschritt werden die Gradienten zurückgesetzt.
            optimizer.zero_grad()

            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits.unsqueeze(0),
                target,
            )

            loss.backward()
            optimizer.step()

    return model


def evaluate(model, indices, return_predictions=False):
    """
    Bewertet das Modell auf den angegebenen Slides ohne weitere Lernschritte.
    Verwendet bags und y aus dem Modul und prüft jedes Slide mit allen
    Patches. Das Modell wird in den Evaluationsmodus versetzt .
    """

    model.eval()

    predictions = []
    targets = []

    # Bei der Evaluation werden keine Gradienten benötigt.
    with torch.no_grad():

        for idx in indices:

            x = bags[idx].to(device)

            logits = model(x)

            prediction = torch.argmax(
                logits
            ).item()

            predictions.append(prediction)
            targets.append(y[idx])

    balanced_acc = balanced_accuracy_score(
        targets,
        predictions,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    if return_predictions:
        return balanced_acc, macro_f1, np.asarray(predictions, dtype=int)

    return balanced_acc, macro_f1


cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=N_REPEATS,
    random_state=SEED,
)

results = []
oof_frames = []


for fold_number, (train_idx, test_idx) in enumerate(
    cv.split(np.zeros(len(y)), y),
    start=1,
):
    set_seed(SEED)

    mean_model = MeanMLP(INPUT_DIM)
    mean_model = train_model(
        mean_model,
        train_idx,
    )

    mean_balacc, mean_f1, mean_predictions = evaluate(
        mean_model,
        test_idx,
        return_predictions=True,
    )

    attention_model = AttentionMLP(INPUT_DIM)
    attention_model = train_model(
        attention_model,
        train_idx,
    )

    attention_balacc, attention_f1, attention_predictions = evaluate(
        attention_model,
        test_idx,
        return_predictions=True,
    )


    # Die laufende Nummer wird in Durchlauf und Prüfgruppe umgerechnet.
    repeat = (fold_number - 1) // 5 + 1
    fold = (fold_number - 1) % 5 + 1

    oof_frames.append(pd.DataFrame({
        "model": MODEL_NAME, "method": "mean",
        "repeat": repeat,
        "fold": fold,
        "sample_index": test_idx,
        "he_id": labels.iloc[test_idx]["he_id"].to_numpy(),
        "y_true": y[test_idx],
        "y_pred": mean_predictions,
    }))

    oof_frames.append(pd.DataFrame({
        "model": MODEL_NAME, "method": "attention",
        "repeat": repeat,
        "fold": fold,
        "sample_index": test_idx,
        "he_id": labels.iloc[test_idx]["he_id"].to_numpy(),
        "y_true": y[test_idx],
        "y_pred": attention_predictions,
    }))

    results.append({
        "repeat": repeat,
        "fold": fold,

        "mean_accuracy": accuracy_score(y[test_idx], mean_predictions),
        "mean_balacc": mean_balacc,
        "mean_f1": mean_f1,

        "attention_accuracy": accuracy_score(y[test_idx], attention_predictions),
        "attention_balacc": attention_balacc,
        "attention_f1": attention_f1,
    })


oof_df = pd.concat(oof_frames, ignore_index=True)
oof_df.to_csv(
    OUT_DIR / f"{MODEL_NAME}_{N_REPEATS}x5_oof_predictions.csv", index=False,
)

# Erst alle Prüfvorhersagen eines Durchlaufs zusammennehmen, dann bewerten.
repeat_rows = []
for repeat, repeat_oof in oof_df.groupby("repeat"):
    row = {"repeat": repeat}
    for method in ["mean", "attention"]:
        method_oof = repeat_oof[repeat_oof["method"] == method]
        row[f"{method}_accuracy"] = accuracy_score(
            method_oof["y_true"], method_oof["y_pred"],
        )
        row[f"{method}_balacc"] = balanced_accuracy_score(
            method_oof["y_true"], method_oof["y_pred"],
        )
        row[f"{method}_f1"] = f1_score(
            method_oof["y_true"], method_oof["y_pred"],
            average="macro", zero_division=0,
        )
    repeat_rows.append(row)

repeat_df = pd.DataFrame(repeat_rows)
repeat_df.to_csv(
    OUT_DIR / f"{MODEL_NAME}_{N_REPEATS}x5_repeat_metrics.csv", index=False,
)

results = pd.DataFrame(results)

results.to_csv(
    OUT_DIR / f"{MODEL_NAME}_{N_REPEATS}x5_raw.csv",
    index=False,
)


# Mittelwert und Streuung der drei OOF-Kennzahlen über alle Durchläufe.
for method, title in [("mean", "Mean + MLP"), ("attention", "Attention + MLP")]:
    print(f"\n{title}")
    for metric, label in [
        ("accuracy", "Accuracy"),
        ("balacc", "Balanced Accuracy"),
        ("f1", "Macro F1"),
    ]:
        values = repeat_df[f"{method}_{metric}"]
        print(f"{label}: {values.mean():.3f} ± {values.std():.3f}")
