# tools\visualize_results.py
import os
import yaml
import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms

from datasets.lumbar_dataset import LumbarDataset
from models.vit_coord import ViTCoordRegressor

LEVEL_ORDER = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

# -------------------------------------------------
# Coordinate Conversion
# -------------------------------------------------
def to_pixel_coords(coords, img_size):
    """
    coords: [N, 2] in [0,1]
    img_size: (H, W)
    """
    H, W = img_size
    coords_px = coords.clone()
    coords_px[:, 0] *= W   # x
    coords_px[:, 1] *= H   # y
    return coords_px



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
def visualize_gt_vs_pred(
    img,
    gt,
    pred,
    title,
    img_size,
    save_path=None,
):
    img = img.squeeze().cpu().numpy()

    # 🔧 convert to pixel space
    gt = to_pixel_coords(gt, img_size)
    pred = to_pixel_coords(pred, img_size)

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.title(title)

    for i, level in enumerate(LEVEL_ORDER):
        plt.scatter(
            gt[i, 0], gt[i, 1],
            c="lime", s=50,
            label="GT" if i == 0 else ""
        )
        plt.scatter(
            pred[i, 0], pred[i, 1],
            c="red", s=60, marker="x",
            label="Pred" if i == 0 else ""
        )
        plt.plot(
            [gt[i, 0], pred[i, 0]],
            [gt[i, 1], pred[i, 1]],
            "y--", linewidth=1
        )

    plt.legend()
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()


# -------------------------------------------------
# Main callable function
# -------------------------------------------------
def show_val_results(
    exp_name="vit_coord",
    split="val",
    n_samples=3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_image=False,
):
    model, cfg = load_model(exp_name, device)
    
    # --------------------------------------------------
    # Transform & Dataset
    # --------------------------------------------------
    
    vit_transform = transforms.Compose([
        transforms.ToPILImage(),        # cv2 → PIL
        transforms.Resize((224, 224)),
        transforms.ToTensor(),          # [1, H, W]
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = LumbarDataset(
        csv_path=f"datasets/lumbar/splits/{split}.csv",
        image_root=f"datasets/lumbar/images/{split}",
        img_size=tuple(cfg["data"]["img_size"]),
        mode="coord",
        transform=vit_transform,
    )
    
    # --------------------------------------------------
    # Output directory (from YAML)
    # --------------------------------------------------
    save_dir = cfg["logging"]["save_dir"]
    out_dir = os.path.join(save_dir, "inference_result")
    if save_image:
        os.makedirs(out_dir, exist_ok=True)
        
        

    indices = random.sample(range(len(dataset)), n_samples)

    for idx in indices:
        img, gt = dataset[idx]

        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))[0].cpu()

        save_path = None
        if save_image:
            save_path = os.path.join(
                out_dir,
                f"{split}_sample_{idx}.png"
            )

        visualize_gt_vs_pred(
            img,
            gt,
            pred,
            title=f"{exp_name} | {split} | sample {idx}",
            img_size=cfg["data"]["img_size"],   # 🔧 ADD THIS
            save_path=save_path,
        )