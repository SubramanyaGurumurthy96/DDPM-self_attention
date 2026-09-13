import random
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def update_ema(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float) -> None:
    for parameter, ema_parameter in zip(model.parameters(), ema_model.parameters()):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)


def save_sample_grid(samples: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    samples = (samples.clamp(-1, 1) + 1) / 2
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(samples, path, nrow=nrow)
