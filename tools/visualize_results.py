import os
import yaml
import torch
import random
import matplotlib.pyplot as plt

from datasets.lumbar_dataset import LumbarDataset
from models.vit_coord import ViTCoordRegressor

LEVEL_ORDER = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


# -------------------------------------------------
# Model Loader
# -------------------------------------------------
def load_model(exp_name, device="cuda"):
    cfg_path = f"configs/{exp_name}.yaml"
    weight_path = f"runs/{exp_name}/best.pt"

    cfg = yaml.safe_load(open(cfg_path))

    model = ViTCoordRegressor(
        num_landmarks=cfg["model"]["num_landmarks"],
        pretrained=False,
        in_chans=1,
    ).to(device)

    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, cfg


# -------------------------------------------------
# Visualization
# -------------------------------------------------
def visualize_gt_vs_pred(img, gt, pred, title):
    img = img.squeeze().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.title(title)

    for i, level in enumerate(LEVEL_ORDER):
        plt.scatter(gt[i, 0], gt[i, 1], c="lime", s=50, label="GT" if i == 0 else "")
        plt.scatter(pred[i, 0], pred[i, 1], c="red", s=60, marker="x", label="Pred" if i == 0 else "")
        plt.plot([gt[i, 0], pred[i, 0]], [gt[i, 1], pred[i, 1]], "y--", linewidth=1)

    plt.legend()
    plt.axis("off")
    plt.show()


# -------------------------------------------------
# Main callable function
# -------------------------------------------------
def show_val_results(
    exp_name="vit_coord",
    split="val",
    n_samples=3,
    device="cuda",
):
    model, cfg = load_model(exp_name, device)

    dataset = LumbarDataset(
        csv_path=f"datasets/lumbar/splits/{split}.csv",
        image_root=f"datasets/lumbar/images/{split}",
        img_size=tuple(cfg["data"]["img_size"]),
        mode="coord",
    )

    indices = random.sample(range(len(dataset)), n_samples)

    for idx in indices:
        img, gt = dataset[idx]

        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))[0].cpu()

        visualize_gt_vs_pred(
            img,
            gt,
            pred,
            title=f"{exp_name} | {split} | sample {idx}",
        )
