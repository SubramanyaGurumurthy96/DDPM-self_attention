from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloader(
    data_dir: str | Path,
    image_size: int = 64,
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = datasets.ImageFolder(Path(data_dir), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
