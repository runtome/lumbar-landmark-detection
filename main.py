import argparse
import yaml

from train.train_vit import train_vit
from train.train_vit_heatmap import train_vit_heatmap
# from train.train_cnn import train_cnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    
    if cfg["model"]["name"] == "vit_heatmap":
        train_vit_heatmap(cfg)
    elif cfg["model"]["name"] == "vit_coord":
        train_vit(cfg)
    
    else:
        raise ValueError(f"Unknown model name: {cfg['model']['name']}")


if __name__ == "__main__":
    main()
