from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_FILE = Path("results/mean_baseline/repeat_metrics.csv")
OUT_FILE = Path("results/mean_baseline/train_vs_test_balacc.png")


df = pd.read_csv(CSV_FILE)

model_order = ["gigapath", "uni_v2", "virchow2"]

df = df[df["model"].isin(model_order)].copy()


summary = (
    df.groupby("model")
    .agg(
        train_balacc_mean=("train_balanced_accuracy", "mean"),
        train_balacc_std=("train_balanced_accuracy", "std"),
        test_balacc_mean=("balanced_accuracy", "mean"),
        test_balacc_std=("balanced_accuracy", "std"),
    )
    .reindex(model_order)
    .reset_index()
)


x = np.arange(len(summary))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    summary["train_balacc_mean"],
    width,
    yerr=summary["train_balacc_std"],
    capsize=5,
    label="Train Balanced Accuracy",
)

plt.bar(
    x + width / 2,
    summary["test_balacc_mean"],
    width,
    yerr=summary["test_balacc_std"],
    capsize=5,
    label="Test Balanced Accuracy",
)

plt.xticks(x, summary["model"])
plt.ylabel("Balanced Accuracy")
plt.ylim(0, 1.05)
plt.title("Train vs. Test Balanced Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(OUT_FILE, dpi=300)
plt.show()

print(f"Plot gespeichert unter: {OUT_FILE}")