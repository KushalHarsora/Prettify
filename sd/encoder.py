import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # creating convolution by merging rgba values into one and storing the matrix (Convolution Visualizer)
            # (batch_size, channel(rgba), height, width) -> (Batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # Creating residue by giving input and output
            # (batch_size, channel(rgba), height, width) -> (Batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batch_size, channel(rgba), height, width) -> (Batch_size, 128, height / 2, width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (batch_size, 128, height / 2, width / 2) -> (Batch_size, 256, height / 2, width / 2)
            VAE_ResidualBlock(128, 256),
            # (batch_size, 256, height / 2, width / 2) -> (Batch_size, 256, height / 2, width / 2)
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, height / 2, width / 2) -> (Batch_size, 256, height / 4, width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (batch_size, 256, height / 4, width / 4) -> (Batch_size, 512, height / 4, width / 4)
            VAE_ResidualBlock(256, 512),

            # (batch_size, 512, height / 4, width / 4) -> (Batch_size, 512, height / 4, width / 4)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height, width) -> (Batch_size, 512, height / 8, width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # creating relation between pixels from first to last
            VAE_AttentionBlock(512),

            # (batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)
            nn.GroupNorm(32, 512),

            # (batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)
            nn.SELU(),

            # (batch_size, 512, height / 8, width / 8) -> (Batch_size, 8, height / 8, width / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (batch_size, 8, height / 8, width / 8) -> (Batch_size, 8, height / 8, width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x -> (Batch_size, Channel, height, width)
        # noise -> (Bath_size, output_channel, height / 8, width / 8)

        for module in self:
            # add pixels in left and bottom
            x = F.pad(x, (0, 1, 0, 1))
        
        x = module(x)

        # (Bath_size, output_channel, height / 8, width / 8) -> Two Tensors of Shape (Batch_size, 4, height / 8, width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # (Bath_size, output_channel, height / 8, width / 8) -> Two Tensors of Shape (Batch_size, 4, height / 8, width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Bath_size, output_channel, height / 8, width / 8) -> Two Tensors of Shape (Batch_size, 4, height / 8, width / 8)
        variance = log_variance.exp()
        # (Bath_size, output_channel, height / 8, width / 8) -> Two Tensors of Shape (Batch_size, 4, height / 8, width / 8)
        standard_deviation = variance.sqrt()

        # Z=N(0, 1) -> N(mean, variance)=x?
        x = mean + standard_deviation + noise
        # scale output by constant
        x *= 0.18215

        return x