#Modifyed train_vit.py to create train_vit_heatmap.py for training ViT model for heatmap regression
# train/train_vit_heatmap.py
import os
import csv
from tools.metrics import per_landmark_mae, pixel_error_per_level
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.vit_heatmap import ViTHeatmap
from datasets.lumbar_dataset import LumbarDataset
from tools.logger import create_writer
from tools.early_stopping import EarlyStopping
from tools.plotting import plot_pixel_error_per_level
from tools.visualization_utils import draw_heatmaps_on_image, draw_landmarks

import torch.nn.functional as F


def soft_argmax_2d(heatmaps):
    """
    heatmaps: [B, N, H, W]
    return:   [B, N, 2]  (normalized 0–1)
    """
    B, N, H, W = heatmaps.shape
    heatmaps = heatmaps.view(B, N, -1)
    heatmaps = F.softmax(heatmaps, dim=-1)

    xs = torch.linspace(0, 1, W, device=heatmaps.device)
    ys = torch.linspace(0, 1, H, device=heatmaps.device)
    xs = xs.repeat(H)
    ys = ys.repeat_interleave(W)

    x = torch.sum(heatmaps * xs, dim=-1)
    y = torch.sum(heatmaps * ys, dim=-1)

    return torch.stack([x, y], dim=-1)




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


def train_vit_heatmap(cfg):
    #=========================
    # CONFIGURATION
    #=========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = cfg["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    log_csv_path = os.path.join(save_dir, "training_log.csv")
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
        heatmap_size=cfg["model"]["heatmap_size"],
        img_size=tuple(cfg["data"]["img_size"]),
        mode="heatmap",
        transform=vit_transform,
    )

    val_dataset = LumbarDataset(
        csv_path=cfg["data"]["val_csv"],
        image_root=cfg["data"]["val_image_root"],
        img_size=tuple(cfg["data"]["img_size"]),
        heatmap_size=cfg["model"]["heatmap_size"],
        mode="heatmap",
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
    model = ViTHeatmap(
        num_landmarks=cfg["model"]["num_landmarks"],
        heatmap_size=cfg["model"]["heatmap_size"],
        in_chans=cfg["model"]["in_channels"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    
    criterion = torch.nn.MSELoss()


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
            leave=True
        )

        for img, gt_hm in pbar:
            img = img.to(device)
            gt_hm = gt_hm.to(device)

            pred = model(img)
            loss = criterion(pred, gt_hm)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

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
            running_abs_error = 0.0
            running_count = 0
            
            pbar = tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{cfg['training']['epochs']} [Val]",
                leave=True
            )
                        
            mae_sum = torch.zeros(cfg["model"]["num_landmarks"], device=device)
            count = 0
            
            pixel_errors = {i: [] for i in range(cfg["model"]["num_landmarks"])}

            with torch.no_grad():
                for img, gt_hm in pbar:
                    img = img.to(device)
                    gt_hm = gt_hm.to(device)
                    
                    pred_hm = model(img)
                    loss = criterion(pred_hm, gt_hm)
                    val_loss += loss.item()
                    

                    
                    # ----------------------------------
                    # 🔥 Decode heatmaps → coordinates
                    # ----------------------------------
                    pred_xy = soft_argmax_2d(pred_hm)
                    gt_xy = soft_argmax_2d(gt_hm)
                    
                    H, W = cfg["data"]["img_size"]
                    
                    pred_xy_px = pred_xy.clone()
                    gt_xy_px   = gt_xy.clone()
                    
                    pred_xy_px[..., 0] *= W
                    pred_xy_px[..., 1] *= H
                    gt_xy_px[..., 0] *= W
                    gt_xy_px[..., 1] *= H
                    
                    # ----------------------------------
                    # 🔹 Pixel MAE
                    # ----------------------------------
                    abs_err = torch.norm(pred_xy_px - gt_xy_px, dim=-1)
                    running_abs_error += abs_err.sum().item()
                    running_count += abs_err.numel()
                    

                    running_mae = running_abs_error / running_count
                        
                    pbar.set_postfix(
                        val_loss=f"{loss.item():.5f}",
                        mae=f"{running_mae:.3f}px"
                    )

            val_loss /= len(val_loader)
            final_mae_pixels = running_abs_error / running_count
            
            mae = torch.mean(torch.abs(pred_xy - gt_xy))
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Val/MAE_pixels", final_mae_pixels, epoch)
            
            # 📊 Pixel Error Plotting
            mean_err = []
            std_err = []
            
            for i, lvl in enumerate(LEVELS):
                vals = np.array(pixel_errors[i])

                if len(vals) == 0:
                    mean = 0.0
                    std = 0.0
                    
                else:
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
                f"Val Loss: {val_loss:.5f} | "
                f"MAE: {final_mae_pixels:.3f} px"
            )

            # 🔧 NEW: scheduler step AFTER validation
            scheduler.step(val_loss)

            # 🔧 NEW: print LR after scheduler
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != current_lr:
                print(f"🔻 LR reduced: {current_lr:.6f} → {new_lr:.6f}")

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
                
                early_stopper.reset() # reset early stopper on improvement
            
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
                    
                    
                # 🔥 decode heatmaps → coordinates
                pred_xy = soft_argmax_2d(pred_vis)
                gt_xy   = soft_argmax_2d(gt_vis)
                
                H, W = cfg["data"]["img_size"]
                pred_xy[..., 0] *= W
                pred_xy[..., 1] *= H
                gt_xy[..., 0]   *= W
                gt_xy[..., 1]   *= H
                
                #Overlay Heatmaps and log to TensorBoard
                overlay_pred_hm = draw_heatmaps_on_image(
                    image=img_vis[i],
                    heatmaps=pred_vis[i],
                    alpha=0.45
                )

                writer.add_image(
                    f"Validation/Pred_Heatmap/epoch_{epoch}_sample_{i}",
                    overlay_pred_hm,
                    epoch,
                    dataformats="HWC"
                )


                # Overlay and log to TensorBoard
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