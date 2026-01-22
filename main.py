import argparse
import yaml

from train.train_vit import train_vit
from train.train_vit_heatmap import train_vit_heatmap
# from train.train_cnn import train_cnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["vit", "vit_heatmap"], required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if args.model == "vit":
        train_vit(cfg)
    elif args.model == "vit_heatmap":
        train_vit_heatmap(cfg)

if __name__ == "__main__":
    main()
