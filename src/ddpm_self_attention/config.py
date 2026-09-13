from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    data_dir: str = "data/celeba64"
    output_dir: str = "outputs"
    image_size: int = 64
    channels: int = 3
    base_channels: int = 128
    time_dim: int = 128
    diffusion_steps: int = 500
    beta_start: float = 1e-4
    beta_end: float = 1e-2
    batch_size: int = 8
    epochs: int = 150
    learning_rate: float = 2e-4
    ema_decay: float = 0.999
    sample_every: int = 10
    num_workers: int = 4
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
