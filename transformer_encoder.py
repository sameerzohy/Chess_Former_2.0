import torch 
import torch.nn as nn 
from attention import RelativeMultiHeadAttention


class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = RelativeMultiHeadAttention(d_model, nhead, dropout=dropout)

        # Feedforward block
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, src_mask=None):
        """
        src: (B,64,d_model)
        src_mask: optional (B,64,64) or (64,64)
        """
        # Self-attention + residual + norm
        attn_out = self.self_attn(src, attn_mask=src_mask)    # (B,64,d_model)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # Feedforward + residual + norm
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)

        return src
