import torch


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    batch = timesteps.shape[0]
    return values.gather(0, timesteps).view(batch, *((1,) * (len(shape) - 1)))


class DDPM:
    def __init__(
        self,
        steps: int = 500,
        beta_start: float = 1e-4,
        beta_end: float = 1e-2,
        device: str | torch.device = "cpu",
    ):
        self.steps = steps
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self, clean_images: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = extract(self.alpha_bars, timesteps, clean_images.shape)
        return alpha_bar.sqrt() * clean_images + (1.0 - alpha_bar).sqrt() * noise

    def velocity_target(
        self, clean_images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = extract(self.alpha_bars, timesteps, clean_images.shape)
        return alpha_bar.sqrt() * noise - (1.0 - alpha_bar).sqrt() * clean_images

    @torch.inference_mode()
    def sample(self, model: torch.nn.Module, count: int, channels: int = 3, size: int = 64):
        images = torch.randn(count, channels, size, size, device=self.device)
        for index in reversed(range(self.steps)):
            timesteps = torch.full((count,), index, device=self.device, dtype=torch.long)
            velocity = model(images, timesteps)
            alpha_bar = self.alpha_bars[index]
            predicted_noise = alpha_bar.sqrt() * velocity + (1.0 - alpha_bar).sqrt() * images
            images = (
                images
                - (1.0 - self.alphas[index])
                / (1.0 - alpha_bar).sqrt()
                * predicted_noise
            ) / self.alphas[index].sqrt()
            if index > 0:
                images = images + self.betas[index].sqrt() * torch.randn_like(images)
        return images
