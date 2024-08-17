import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (Batch_size, Seq_Len, dim)

        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        intermin_shape = (batch_size, sequence_length, self.n_heads, self.d_heads)

        # (Batch_size, Seq_Len, Dim) -> (Batch_size, Seq_Len, Dim = 3) -> 3 Tensors of Shape (Batch_size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=1)

        # (Batch_size, Seq_Len, Dim) -> (Batch_size, Seq_Len, H, Dim / H) -> (Batch_size, H, Seq_Len, Dim / H)
        q = q.view(intermin_shape).transpose(1, 2)
        k = k.view(intermin_shape).transpose(1, 2)
        v = v.view(intermin_shape).transpose(1, 2)

        # (Batch_size, H, Seq_Len, Dim)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # Mask where upper triangle is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.mask_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_heads)

        weight = F.softmax(weight, dim=1)

        # (Batch_size, H, Seq_Len, Seq_Len) @ (batch_size, H, Seq_Len, Dim / H) -> (Batch_size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_size, H, Seq_Len, Dim / H) -> (Batch_size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (Batch_size, Seq_Len, dim)
        return output
