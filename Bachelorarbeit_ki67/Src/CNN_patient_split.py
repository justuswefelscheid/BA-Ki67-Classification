"""Trainiert eine ResNet-18 zur binären Ki-67-Zellklassifikation.

Die Zellbilder werden slide-basiert in Trainings-, Validierungs- und Testdaten
aufgeteilt, sodass Zellen desselben Slides nicht in mehreren Datensätzen
vorkommen. Das Modell wird mit gewichteter binärer Kreuzentropie trainiert.

Während des Trainings werden Loss und Balanced Accuracy für Training und
Validierung aufgezeichnet. Das Modell mit dem niedrigsten Validation Loss
wird gespeichert und anschließend auf dem Testdatensatz ausgewertet.
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


DATA_DIR = Path(
    "/Users/justuswefelscheid/Documents/Studium/"
    "Bachelorarbeit/ki67_QuPath_Project/cell_patches"
)

RESULT_DIR = Path(
    "/Users/justuswefelscheid/Documents/Studium/"
    "Bachelorarbeit/Ki67_Pipeline/results"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = DATA_DIR / "best_resnet18.pth"

IMAGE_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


image_paths = []
labels = []
groups = []

for class_name, label in [
    ("negative", 0),
    ("positive", 1),
]:

    class_dir = DATA_DIR / class_name

    for image_path in class_dir.rglob("*"):

        if image_path.suffix.lower() not in {
            ".png", ".jpg", ".jpeg", ".tif", ".tiff"
        }:
            continue

        image_paths.append(image_path)
        labels.append(label)

        # Der erste Unterordner entspricht dem Slide und wird für
        # die gruppenbasierte Datenaufteilung verwendet.
        relative_path = image_path.relative_to(class_dir)
        groups.append(relative_path.parts[0])


image_paths = np.array(image_paths)
labels = np.array(labels)
groups = np.array(groups)


# Zunächst werden 20 % der Slides als unabhängiger Testdatensatz abgetrennt.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED,
)

train_val_idx, test_idx = next(
    splitter.split(
        image_paths,
        labels,
        groups,
    )
)

# Aus den verbleibenden Slides werden weitere 20 % für die Validierung genutzt.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED,
)

train_relative, val_relative = next(
    splitter.split(
        image_paths[train_val_idx],
        labels[train_val_idx],
        groups[train_val_idx],
    )
)

train_idx = train_val_idx[train_relative]
val_idx = train_val_idx[val_relative]


# Augmentation wird ausschließlich auf die Trainingsbilder angewendet.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class CellDataset(Dataset):
    """Lädt einzelne Zellbilder und ihre binären Ki-67-Labels."""

    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image = Image.open(self.paths[index]).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32,
        )

        return image, label


train_dataset = CellDataset(
    image_paths[train_idx],
    labels[train_idx],
    train_transform,
)

val_dataset = CellDataset(
    image_paths[val_idx],
    labels[val_idx],
    eval_transform,
)

test_dataset = CellDataset(
    image_paths[test_idx],
    labels[test_idx],
    eval_transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# Vortrainierte ResNet-18 mit einem einzelnen Ausgang für binäre Klassifikation.
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    1,
)

model = model.to(DEVICE)


# Die positive Klasse wird entsprechend ihrer Häufigkeit im Trainingssatz gewichtet.
number_negative = np.sum(labels[train_idx] == 0)
number_positive = np.sum(labels[train_idx] == 1)

positive_weight = number_negative / number_positive

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [positive_weight],
        dtype=torch.float32,
        device=DEVICE,
    )
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


best_val_loss = float("inf")
patience = 7
epochs_without_improvement = 0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0
    train_predictions = []
    train_targets = []

    for images, targets in train_loader:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images).squeeze(1)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).int()

        train_loss += loss.item() * images.size(0)

        train_predictions.extend(
            predictions.cpu().numpy()
        )
        train_targets.extend(
            targets.cpu().numpy()
        )

    train_loss /= len(train_dataset)

    train_accuracy = balanced_accuracy_score(
        train_targets,
        train_predictions,
    )


    model.eval()

    val_loss = 0
    val_predictions = []
    val_targets = []

    # Während der Validierung werden keine Gradienten berechnet.
    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(images).squeeze(1)
            loss = criterion(logits, targets)

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).int()

            val_loss += loss.item() * images.size(0)

            val_predictions.extend(
                predictions.cpu().numpy()
            )
            val_targets.extend(
                targets.cpu().numpy()
            )

    val_loss /= len(val_dataset)

    val_accuracy = balanced_accuracy_score(
        val_targets,
        val_predictions,
    )


    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    # Das Modell mit dem niedrigsten Validation Loss wird gespeichert.
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

    else:

        epochs_without_improvement += 1

        # Training wird nach sieben Epochen ohne Verbesserung beendet.
        if epochs_without_improvement >= patience:
            break


epochs = range(1, len(train_losses) + 1)


plt.figure()

plt.plot(
    epochs,
    train_losses,
    label="Training",
)

plt.plot(
    epochs,
    val_losses,
    label="Validation",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "loss_curve.png",
    dpi=300,
)

plt.close()


plt.figure()

plt.plot(
    epochs,
    train_accuracies,
    label="Training",
)

plt.plot(
    epochs,
    val_accuracies,
    label="Validation",
)

plt.xlabel("Epoch")
plt.ylabel("Balanced Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "accuracy_curve.png",
    dpi=300,
)

plt.close()


# Für die Testauswertung wird das Modell mit dem besten Validation Loss geladen.
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

model.eval()

test_probabilities = []
test_predictions = []
test_targets = []

with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(DEVICE)

        logits = model(images).squeeze(1)
        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).int()

        test_probabilities.extend(
            probabilities.cpu().numpy()
        )

        test_predictions.extend(
            predictions.cpu().numpy()
        )

        test_targets.extend(
            targets.numpy()
        )


accuracy = accuracy_score(
    test_targets,
    test_predictions,
)

balanced_accuracy = balanced_accuracy_score(
    test_targets,
    test_predictions,
)

auc = roc_auc_score(
    test_targets,
    test_probabilities,
)


print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC-AUC:           {auc:.4f}")

print(f"\nLoss curve:     {RESULT_DIR / 'loss_curve.png'}")
print(f"Accuracy curve: {RESULT_DIR / 'accuracy_curve.png'}")