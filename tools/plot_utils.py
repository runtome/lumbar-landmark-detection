import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_curve(csv_path, save_dir):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()

    print(f"📈 Loss curve saved to {save_path}")
