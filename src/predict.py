import argparse
import csv
from pathlib import Path

from PIL import Image

from src.config import load_config, resolve_path
from src.data import list_images
from src.labels import ARTISTS
from src.model import PainterModel


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the painter of every image in a folder.")
    parser.add_argument("folder", type=Path, help="Folder of images to classify.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override config/infer.yaml.")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config("infer")
    checkpoint = args.checkpoint or resolve_path(config["checkpoint"])
    batch_size = args.batch_size or config["batch_size"]
    folder = args.folder
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    paths = list_images(folder)
    if not paths:
        raise SystemExit(f"No images found in {folder}")

    predictor = PainterModel(checkpoint)
    rows = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = [Image.open(path) for path in batch_paths]
        probabilities = predictor.predict_batch(images)
        for path, image, scores in zip(batch_paths, images, probabilities):
            image.close()
            confidence, class_id = scores.max(dim=0)
            artist = ARTISTS[int(class_id)]
            row = {
                "file": path.name,
                "artist": artist,
                "confidence": f"{float(confidence):.6f}",
            }
            rows.append(row)
            print(f"{row['file']}\t{row['artist']}\t{row['confidence']}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file", "artist", "confidence"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
