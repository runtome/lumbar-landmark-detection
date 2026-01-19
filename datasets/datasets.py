import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
KAGGLE_ROOT = "/kaggle/input/lumbar-coordinate-pretraining-dataset"
CSV_PATH = f"{KAGGLE_ROOT}/coords_pretrain.csv"
IMAGE_ROOT = f"{KAGGLE_ROOT}/data"

OUT_ROOT = "datasets/lumbar"
TRAIN_RATIO = 0.8
SEED = 42

LEVEL_ORDER = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

# ===============================
# UTILS
# ===============================
def mkdirs():
    for split in ["train", "val"]:
        os.makedirs(f"{OUT_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_ROOT}/labels/{split}", exist_ok=True)
    os.makedirs(f"{OUT_ROOT}/splits", exist_ok=True)


def get_image_path(row):
    """
    Construct image path from source + filename
    processed_xxx_jpgs -> xxx is source
    """
    folder = f"processed_{row.source}_jpgs"
    return os.path.join(IMAGE_ROOT, folder, row.filename)


# ===============================
# DATASET PREPARATION
# ===============================
def prepare_dataset():
    print("📥 Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    # Ensure consistent lumbar order
    df["level"] = pd.Categorical(
        df["level"], categories=LEVEL_ORDER, ordered=True
    )

    # Unique image-level split (important!)
    img_df = (
        df[["filename", "source"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print("🔀 Stratified split by source (80/20)...")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - TRAIN_RATIO,
        random_state=SEED,
    )

    train_idx, val_idx = next(
        splitter.split(img_df["filename"], img_df["source"])
    )

    train_imgs = img_df.iloc[train_idx]
    val_imgs = img_df.iloc[val_idx]

    train_files = set(train_imgs.filename)
    val_files = set(val_imgs.filename)

    df_train = df[df.filename.isin(train_files)]
    df_val = df[df.filename.isin(val_files)]

    # Save split CSVs
    df_train.to_csv(f"{OUT_ROOT}/splits/train.csv", index=False)
    df_val.to_csv(f"{OUT_ROOT}/splits/val.csv", index=False)

    print("🖼️ Copying images...")
    for split, split_df in zip(
        ["train", "val"], [train_imgs, val_imgs]
    ):
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            src = get_image_path(row)
            dst = f"{OUT_ROOT}/images/{split}/{row.filename}"

            if not os.path.exists(dst):
                shutil.copy(src, dst)

    print("📄 Writing label files (keypoints)...")
    for split, split_df in zip(
        ["train", "val"], [df_train, df_val]
    ):
        for fname, group in tqdm(split_df.groupby("filename")):
            label_path = f"{OUT_ROOT}/labels/{split}/{fname.replace('.jpg','.txt')}"

            # YOLO-Pose style: cls x y w h kpts...
            # We use dummy box (full image) since this is landmark-only
            line = ["0", "0.5", "0.5", "1.0", "1.0"]

            for level in LEVEL_ORDER:
                row = group[group.level == level]
                if len(row) == 0:
                    line += ["0", "0", "0"]  # missing
                else:
                    line += [
                        f"{row.relative_x.values[0]:.6f}",
                        f"{row.relative_y.values[0]:.6f}",
                        "2",  # visibility
                    ]

            with open(label_path, "w") as f:
                f.write(" ".join(line))

    print("✅ Dataset preparation complete!")


# ===============================
# DEBUG VISUALIZATION
# ===============================
def show_random_sample(n=3):
    import random
    import cv2
    import matplotlib.pyplot as plt

    df = pd.read_csv(f"{OUT_ROOT}/splits/train.csv")
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
            x = int(r.x)
            y = int(r.y)
            plt.scatter(x, y, s=40, label=r.level)

        plt.legend()
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    mkdirs()
    prepare_dataset()
    show_random_sample(n=2)
