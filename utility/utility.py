# utility/utility.py
import os
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# ===============================
# GLOBAL PATHS
# ===============================
KAGGLE_ROOT = "/kaggle/input/lumbar-coordinate-pretraining-dataset"
CSV_PATH = f"{KAGGLE_ROOT}/coords_pretrain.csv"
IMAGE_ROOT = f"{KAGGLE_ROOT}/data"

OUT_ROOT = "datasets/lumbar"


# ===============================
# IMAGE PATH UTILITY
# ===============================
def get_image_path(row):
    """
    Construct image path from source + filename
    processed_xxx_jpgs -> xxx is source
    """
    folder = f"processed_{row.source}_jpgs"
    return os.path.join(IMAGE_ROOT, folder, row.filename)


# ===============================
# DEBUG VISUALIZATION
# ===============================
def show_random_sample(n=3, split="train"):
    """
    Show random samples with lumbar keypoints
    """
    csv_path = f"{OUT_ROOT}/splits/{split}.csv"
    df = pd.read_csv(csv_path)

    samples = random.sample(list(df.filename.unique()), n)

    for fname in samples:
        rows = df[df.filename == fname]
        row0 = rows.iloc[0]

        img_path = get_image_path(row0)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"{fname} | source={row0.source}")

        for _, r in rows.iterrows():
            plt.scatter(int(r.x), int(r.y), s=40, label=r.level)

        plt.legend()
        plt.axis("off")
        plt.show()
