import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.float()
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timestep.device)
            / (half - 1)
        )
        angles = timestep[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(x)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = F.silu(self.norm2(self.conv2(hidden)))
        return hidden + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        hidden = self.norm(x)
        query = self.query(hidden).reshape(batch, channels, height * width)
        key = self.key(hidden).reshape(batch, channels, height * width)
        value = self.value(hidden).reshape(batch, channels, height * width)
        weights = torch.softmax(
            query.transpose(1, 2) @ key / math.sqrt(channels), dim=-1
        )
        attended = (value @ weights.transpose(1, 2)).reshape(
            batch, channels, height, width
        )
        return x + self.projection(attended)


class UNet64(nn.Module):
    """U-Net denoiser for 64 x 64 RGB images using v-prediction."""

    def __init__(self, channels: int = 3, base_channels: int = 128, time_dim: int = 128):
        super().__init__()
        ch = base_channels
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.input_conv = nn.Conv2d(channels, ch, 3, padding=1)
        self.down1 = ResidualBlock(ch, ch, time_dim)
        self.down2 = ResidualBlock(ch, ch * 2, time_dim)
        self.down3 = ResidualBlock(ch * 2, ch * 4, time_dim)
        self.pool = nn.AvgPool2d(2)
        self.middle = ResidualBlock(ch * 4, ch * 4, time_dim)
        self.attention = SelfAttention(ch * 4)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up2 = ResidualBlock(ch * 6, ch * 2, time_dim)
        self.up1 = ResidualBlock(ch * 3, ch, time_dim)
        self.output_conv = nn.Conv2d(ch, channels, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(timestep)
        skip1 = self.down1(self.input_conv(x), time_embedding)
        skip2 = self.down2(self.pool(skip1), time_embedding)
        hidden = self.down3(self.pool(skip2), time_embedding)
        hidden = self.attention(self.middle(hidden, time_embedding))
        hidden = self.up2(
            torch.cat((self.upsample(hidden), skip2), dim=1), time_embedding
        )
        hidden = self.up1(
            torch.cat((self.upsample(hidden), skip1), dim=1), time_embedding
        )
        return self.output_conv(hidden)
