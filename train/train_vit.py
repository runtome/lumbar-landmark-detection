import os
import yaml
import torch
from torch.utils.data import DataLoader

from models.vit_coord import ViTCoordRegressor
from models.losses import NormalizedL2Loss
from datasets.lumbar_dataset import LumbarDataset


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def train(cfg, train_loader, val_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ViTCoordRegressor(
        num_landmarks=cfg["model"]["num_landmarks"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    criterion = NormalizedL2Loss(cfg["data"]["img_size"])

    save_dir = cfg["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        # ------------------
        # TRAIN
        # ------------------
        model.train()
        train_loss = 0

        for img, gt in train_loader:
            img, gt = img.to(device), gt.to(device)

            pred = model(img)
            loss = criterion(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.5f}")

        # ------------------
        # SAVE LAST
        # ------------------
        if epoch % cfg["logging"]["log_interval"] == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                os.path.join(save_dir, "last.pt"),
            )

        # ------------------
        # VALIDATION
        # ------------------
        if epoch % cfg["logging"]["val_interval"] == 0:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for img, gt in val_loader:
                    img, gt = img.to(device), gt.to(device)
                    pred = model(img)
                    loss = criterion(pred, gt)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.5f}")

            # ------------------
            # SAVE BEST
            # ------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    os.path.join(save_dir, "best.pt"),
                )
                print("✅ Best model updated!")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/vit_coord.yaml"))

    # TODO: replace with your actual Dataset
    train_dataset = LumbarDataset(
        csv_path="datasets/lumbar/splits/train.csv",
        image_root="datasets/lumbar/images/train",
        img_size=(224, 224),
        mode="coord"
    )

    val_dataset = LumbarDataset(
        csv_path="datasets/lumbar/splits/val.csv",
        image_root="datasets/lumbar/images/val",
        img_size=(224, 224),
        mode="coord"
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    train(cfg, train_loader, val_loader)