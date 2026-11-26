import argparse

import yaml
from torch.utils.data import DataLoader

from src.experiments.lane_segmentation_ppo.lane_segmentation.dataset import LaneSegDataset, build_index
from src.experiments.lane_segmentation_ppo.lane_segmentation.train import (
    DEFAULT_CONFIG, build_model, evaluate, get_device, print_results)


def main():
    parser = argparse.ArgumentParser(description="Per-track evaluation of TwinLiteNet+ weights")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--tracks", nargs="*", default=None, help="defaults to the config's tracks")
    parser.add_argument("--split", choices=["val", "all"], default="val")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]

    chunks, train_items, val_items = build_index(
        d["root"], args.tracks or d["tracks"], d["val_fraction"], d["val_only_tracks"],
        [] if args.tracks else d["exclude_tracks"])
    items = val_items if args.split == "val" else train_items + val_items

    device = get_device(args.device)
    model = build_model(cfg["model"]["config"], args.weights, device)
    loader = DataLoader(LaneSegDataset(chunks, items), batch_size=args.batch_size, num_workers=2)
    print_results(evaluate(model, loader, device))


if __name__ == "__main__":
    main()