import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import csv

from models.vit_coord import ViTCoordRegressor
from models.losses import NormalizedL2Loss
from datasets.lumbar_dataset import LumbarDataset
from tools.logger import create_writer
from tools.early_stopping import EarlyStopping



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


def train_vit(cfg):
    #=========================
    # CONFIGURATION
    #=========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = cfg["logging"]["save_dir"]
    log_csv_path = os.path.join(save_dir, "training_log.csv")
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    csv_file = open(log_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])


    # =========================
    # TRANSFORMS
    # =========================
    vit_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # =========================
    # DATASETS
    # =========================
    train_dataset = LumbarDataset(
        csv_path=cfg["data"]["train_csv"],
        image_root=cfg["data"]["train_image_root"],
        img_size=tuple(cfg["data"]["img_size"]),
        mode="coord",
        transform=vit_transform,
    )

    val_dataset = LumbarDataset(
        csv_path=cfg["data"]["val_csv"],
        image_root=cfg["data"]["val_image_root"],
        img_size=tuple(cfg["data"]["img_size"]),
        mode="coord",
        transform=vit_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    # =========================
    # MODEL
    # =========================
    model = ViTCoordRegressor(
        num_landmarks=cfg["model"]["num_landmarks"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    criterion = NormalizedL2Loss(cfg["data"]["img_size"])

    # =========================
    # 🔧 NEW: LR SCHEDULER
    # =========================
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["training"]["scheduler"]["factor"],
        patience=cfg["training"]["scheduler"]["patience"],
        min_lr=cfg["training"]["scheduler"]["min_lr"],
    )

    # =========================
    # LOGGING
    # =========================
    
    os.makedirs(save_dir, exist_ok=True)
    writer = create_writer(save_dir)


    best_val_loss = float("inf")
    early_stopper = EarlyStopping(patience=cfg["training"]["early_stopping"]["patience"])
    
    
    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_loss = 0.0

        for img, gt in train_loader:
            img, gt = img.to(device), gt.to(device)

            pred = model(img)
            loss = criterion(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # 🔧 NEW: get LR
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss:.5f} | "
            f"LR: {current_lr:.6f}"
        )

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
            val_loss = 0.0

            with torch.no_grad():
                for img, gt in val_loader:
                    img, gt = img.to(device), gt.to(device)
                    pred = model(img)
                    loss = criterion(pred, gt)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            
            writer.add_scalar("Loss/val", val_loss, epoch)


            print(
                f"[Epoch {epoch}] "
                f"Val Loss: {val_loss:.5f}"
            )

            # 🔧 NEW: scheduler step AFTER validation
            scheduler.step(val_loss)

            # 🔧 NEW: print LR after scheduler
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != current_lr:
                print(f"🔻 LR reduced: {current_lr:.6f} → {new_lr:.6f}")
                early_stopper.reset()   # 🔥 IMPORTANT to reset Early stopping

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
            
            # ------------------
            # LOG TO CSV
            # ------------------
            current_lr = optimizer.param_groups[0]["lr"]

            csv_writer.writerow([
                epoch,
                train_loss,
                val_loss if epoch % cfg["logging"]["val_interval"] == 0 else None,
                current_lr,
            ])
            csv_file.flush()

            # 🔧 NEW: early stopping check
            if early_stopper.step(val_loss):
                csv_file.close()
                writer.flush()
                writer.close()
                print("⏹️ Early stopping triggered!")
                break
        # ------------------
        # VISUALIZE AT FINAL EPOCH
        # ------------------
        if epoch == cfg["training"]["epochs"]:
            from tools.visualize_results import show_val_results
            show_val_results("vit_coord", n_samples=3)
            csv_file.close()
            writer.flush()
            writer.close()