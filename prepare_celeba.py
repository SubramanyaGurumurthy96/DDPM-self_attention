import argparse
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision import transforms


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a filtered CelebA subset.")
    parser.add_argument("--images", required=True, help="Directory containing CelebA JPG files")
    parser.add_argument("--attributes", required=True, help="Path to list_attr_celeba.csv")
    parser.add_argument("--output", default="data/celeba64/person")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    attributes = pd.read_csv(args.attributes)
    filtered = attributes[
        (attributes["Young"] == 1)
        & (attributes["Smiling"] == 1)
        & (attributes["Male"] == -1)
        & (attributes["No_Beard"] == 1)
    ]
    candidates = [name for name in filtered["image_id"] if (source_dir / name).exists()]
    if len(candidates) < args.count:
        raise ValueError(f"Requested {args.count} images, but only {len(candidates)} are available")

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.count)
    resize = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64)])
    for name in selected:
        destination = output_dir / name
        with Image.open(source_dir / name) as image:
            resize(image.convert("RGB")).save(destination, quality=95)
    print(f"Prepared {len(selected)} images in {output_dir}")


if __name__ == "__main__":
    main()
