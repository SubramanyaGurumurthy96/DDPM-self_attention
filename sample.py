import argparse
from pathlib import Path

import torch

from ddpm_self_attention import DDPM, ExperimentConfig, UNet64
from ddpm_self_attention.utils import save_sample_grid, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images from a DDPM checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output", default="outputs/generated_samples.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    seed_everything(config.seed)
    model = UNet64(config.channels, config.base_channels, config.time_dim).to(device)
    model.load_state_dict(checkpoint["ema_model_state"])
    model.eval()
    diffusion = DDPM(
        config.diffusion_steps, config.beta_start, config.beta_end, device
    )
    samples = diffusion.sample(model, args.count, config.channels, config.image_size)
    save_sample_grid(samples, Path(args.output), nrow=max(1, int(args.count**0.5)))
    print(f"Saved samples to {args.output}")


if __name__ == "__main__":
    main()
