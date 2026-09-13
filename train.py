import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ddpm_self_attention import DDPM, ExperimentConfig, UNet64
from ddpm_self_attention.data import create_dataloader
from ddpm_self_attention.utils import save_sample_grid, seed_everything, update_ema


def train(config: ExperimentConfig) -> None:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loader = create_dataloader(
        config.data_dir, config.image_size, config.batch_size, config.num_workers
    )
    model = UNet64(config.channels, config.base_channels, config.time_dim).to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    diffusion = DDPM(
        config.diffusion_steps, config.beta_start, config.beta_end, device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_history = []

    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for clean_images, _ in progress:
            clean_images = clean_images.to(device, non_blocking=True)
            timesteps = torch.randint(
                0, diffusion.steps, (clean_images.shape[0],), device=device
            )
            noise = torch.randn_like(clean_images)
            noisy_images = diffusion.add_noise(clean_images, timesteps, noise)
            predicted_velocity = model(noisy_images, timesteps)
            target_velocity = diffusion.velocity_target(clean_images, noise, timesteps)
            loss = F.mse_loss(predicted_velocity, target_velocity)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_ema(model, ema_model, config.ema_decay)
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(loader)
        loss_history.append(average_loss)
        print(f"Average loss: {average_loss:.4f}")

        should_save = epoch % config.sample_every == 0 or epoch == config.epochs - 1
        if should_save:
            checkpoint_path = checkpoint_dir / f"ddpm_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "config": config.to_dict(),
                    "model_state": model.state_dict(),
                    "ema_model_state": ema_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss_history": loss_history,
                },
                checkpoint_path,
            )
            ema_model.eval()
            samples = diffusion.sample(
                ema_model, count=16, channels=config.channels, size=config.image_size
            )
            save_sample_grid(samples, sample_dir / f"epoch_{epoch:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the self-attention DDPM.")
    parser.add_argument("--config", default="configs/celeba64.yaml")
    args = parser.parse_args()
    train(ExperimentConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
