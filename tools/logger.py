from torch.utils.tensorboard import SummaryWriter
import os

def create_writer(save_dir):
    log_dir = os.path.join(save_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)
