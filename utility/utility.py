# utility/utility.py
import os
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch

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
def get_image_path(row, split):
    """
    Return absolute image path for a CSV row
    """
    image_dir = f"{OUT_ROOT}/images/{split}"
    return os.path.join(image_dir, row.filename)



# ===============================
# DEBUG VISUALIZATION
# ===============================
def show_random_sample(n=3, split="train"):
    """
    Show random samples with GT lumbar landmarks
    """
    csv_path = f"{OUT_ROOT}/splits/{split}.csv"
    df = pd.read_csv(csv_path)

    samples = random.sample(list(df.filename.unique()), n)

    for fname in samples:
        rows = df[df.filename == fname]
        row0 = rows.iloc[0]

        img_path = get_image_path(row0, split)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ Failed to load {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"{fname} | split={split} | source={row0.source}")

        for _, r in rows.iterrows():
            plt.scatter(int(r.x), int(r.y), s=50, label=r.level)

        plt.legend()
        plt.axis("off")
        plt.show()

# def show_random_sample(n=3, split="train"):
#     """
#     Show random samples with lumbar keypoints
#     """
#     csv_path = f"{OUT_ROOT}/splits/{split}.csv"
#     df = pd.read_csv(csv_path)

#     samples = random.sample(list(df.filename.unique()), n)

#     for fname in samples:
#         rows = df[df.filename == fname]
#         row0 = rows.iloc[0]

#         img_path = get_image_path(row0)
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         plt.figure(figsize=(5, 5))
#         plt.imshow(img)
#         plt.title(f"{fname} | source={row0.source}")

#         for _, r in rows.iterrows():
#             plt.scatter(int(r.x), int(r.y), s=40, label=r.level)

#         plt.legend()
#         plt.axis("off")
#         plt.show()

# ===============================
# VISUALIZATION OF PREDICTIONS
# ===============================

def show_prediction_vs_gt(
    img,
    gt_coords,
    pred_coords,
    title="Prediction vs Ground Truth",
):
    """
    img        : Tensor [1, H, W] or [H, W]
    gt_coords  : [N, 2]
    pred_coords: [N, 2]
    """

    if torch.is_tensor(img):
        img = img.squeeze().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.title(title)

    # Ground Truth (green)
    plt.scatter(
        gt_coords[:, 0],
        gt_coords[:, 1],
        c="lime",
        s=50,
        label="GT",
        marker="o",
    )

    # Prediction (red X)
    plt.scatter(
        pred_coords[:, 0],
        pred_coords[:, 1],
        c="red",
        s=60,
        label="Pred",
        marker="x",
    )

    # Draw error lines
    for i in range(len(gt_coords)):
        plt.plot(
            [gt_coords[i, 0], pred_coords[i, 0]],
            [gt_coords[i, 1], pred_coords[i, 1]],
            "y--",
            linewidth=1,
        )

    plt.legend()
    plt.axis("off")
    plt.show()
