import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from models.vit_coord import ViTCoordRegressor
from models.losses import NormalizedL2Loss
from datasets.lumbar_dataset import LumbarDataset
from tools.logger import create_writer
from tools.early_stopping import EarlyStopping
from tools.visualization_utils import draw_landmarks
from tools.metrics import per_landmark_mae
from tools.metrics import pixel_error_per_level
from tools.plotting import plot_pixel_error_per_level




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
    LEVELS = ["L1", "L2", "L3", "L4", "L5"]



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

    # criterion = NormalizedL2Loss(cfg["data"]["img_size"])
    # criterion = torch.nn.MSELoss()
    
    if cfg["loss"]["type"] == "normalized_l2":
        criterion = NormalizedL2Loss(cfg["data"]["img_size"])
    elif cfg["loss"]["type"] == "smooth_l1":
        criterion = torch.nn.SmoothL1Loss(beta=cfg["loss"]["beta"])


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

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg['training']['epochs']} [Train]",
            leave=False
        )

        for img, gt in pbar:
            img, gt = img.to(device), gt.to(device)

            pred = model(img)
            loss = criterion(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

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
            pbar = tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{cfg['training']['epochs']} [Val]",
                leave=False
            )
                        
            mae_sum = torch.zeros(cfg["model"]["num_landmarks"], device=device)
            count = 0
            
            pixel_errors = {i: [] for i in range(cfg["model"]["num_landmarks"])}

            with torch.no_grad():
                for img, gt in pbar:
                    img, gt = img.to(device), gt.to(device)
                    pred = model(img)
                    loss = criterion(pred, gt)
                    val_loss += loss.item()
                    
                    batch_mae = per_landmark_mae(pred, gt)
                    mae_sum += batch_mae
                    count += 1
                    
                    batch_pixel_errors = pixel_error_per_level(
                        pred, gt
                    )

                    for i, errs in batch_pixel_errors.items():
                        pixel_errors[i].extend(errs)
                        
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_loss /= len(val_loader)
            mae_avg = mae_sum / count
            
            mae = torch.mean(torch.abs(pred - gt))
            writer.add_scalar("Val/MAE_pixels", mae.item(), epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            for i, mae in enumerate(mae_avg):
                writer.add_scalar(f"MAE/Landmark_{i+1}", mae.item(), epoch)
            
            # 📊 Pixel Error Plotting
            mean_err = []
            std_err = []
            
            for i, lvl in enumerate(LEVELS):
                vals = np.array(pixel_errors[i])

                mean = vals.mean()
                std = vals.std()

                mean_err.append(mean)
                std_err.append(std)

                # TensorBoard scalars
                writer.add_scalar(f"PixelErrorMean/{lvl}", mean, epoch)
                writer.add_scalar(f"PixelErrorStd/{lvl}", std, epoch)

            fig = plot_pixel_error_per_level(LEVELS, mean_err, std_err)
            writer.add_figure("PixelError/PerLevel", fig, epoch)
            
            #--------------------
            # Save plot as PNG
            # --------------------
            plot_dir = os.path.join(save_dir, "pixel_error_plots")
            os.makedirs(plot_dir, exist_ok=True)

            fig_path = os.path.join(
                plot_dir,
                f"pixel_error_epoch_{epoch}.png"
            )

            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            
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
            
            # =========================
            # 📊 TensorBoard Image Overlay
            # =========================
            if epoch % cfg["logging"]["val_interval"] == 0:
                img_vis, gt_vis = next(iter(val_loader))
                img_vis = img_vis.to(device)
                gt_vis = gt_vis.to(device)

                with torch.no_grad():
                    pred_vis = model(img_vis)

                overlay = draw_landmarks(
                    image=img_vis[0],
                    gt=gt_vis[0],
                    pred=pred_vis[0]
                )
                
                for i in range(min(3, img_vis.size(0))):
                    overlay = draw_landmarks(
                        image=img_vis[i],
                        gt=gt_vis[i],
                        pred=pred_vis[i]
                    )

                    writer.add_image(
                        f"Validation/GT_vs_Pred/epoch_{epoch}_sample_{i}",
                        overlay,
                        epoch,
                        dataformats="HWC"
                )
                        

            # 🔧 early stopping check
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