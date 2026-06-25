import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("training_history_2026-06-19_08-46-40.csv")

# Plot Loss
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(df["epoch"], df["loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot Accuracy + Jaccard
plt.subplot(1,2,2)
plt.plot(df["epoch"], df["accuracy"], label="Train Accuracy")
plt.plot(df["epoch"], df["val_accuracy"], label="Val Accuracy")
plt.plot(df["epoch"], df["jaccard_coef"], label="Train Jaccard")
plt.plot(df["epoch"], df["val_jaccard_coef"], label="Val Jaccard")
plt.title("Accuracy & Jaccard over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()

plt.tight_layout()
plt.savefig("evaluation_metrics.png")  # saves PNG
plt.show()
