import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Standard projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # --- Shaw-style 2D relative bias (for 8x8 board) ---
        # 15 x 15 possible (dr, df) pairs
        self.num_rel = 15 * 15
        # Bias per relation *per head*
        self.rel_bias = nn.Embedding(self.num_rel, nhead)

        # Precompute relative index matrix for squares 0..63
        rel_index = self._build_rel_index_8x8()  # (64,64)
        self.register_buffer("rel_index", rel_index, persistent=False)

    @staticmethod
    def _build_rel_index_8x8():
        # squares: 0..63, FEN order (rank 8->1, file a->h)
        idx = torch.arange(64)
        r = idx // 8
        f = idx % 8

        # (64,1) and (1,64) to broadcast
        r_i = r.view(64, 1)
        r_j = r.view(1, 64)
        f_i = f.view(64, 1)
        f_j = f.view(1, 64)

        dr = r_i - r_j  # (64,64) in [-7,7]
        df = f_i - f_j  # (64,64) in [-7,7]

        rel_index = (dr + 7) * 15 + (df + 7)  # map to 0..224
        rel_index = rel_index.long()
        return rel_index  # (64,64)

    def forward(self, x, attn_mask=None):
        """
        x: (B, 64, d_model)
        attn_mask: optional (B, 64, 64) or (64,64) mask
        """
        B, S, D = x.shape
        assert S == 64, "This relative encoding assumes exactly 64 squares"
        assert D == self.d_model

        # Project to Q, K, V
        q = self.q_proj(x)  # (B,64,D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split heads: (B, nhead, 64, head_dim)
        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_logits = torch.matmul(q, k.transpose(-2, -1))  # (B, nhead, 64, 64)
        attn_logits = attn_logits / math.sqrt(self.head_dim)

        # --- Add Shaw relative bias ---
        # self.rel_index: (64,64)
        # rel_bias: (64,64,nhead) -> (nhead,64,64)
        rel_bias = self.rel_bias(self.rel_index.view(-1))  # (64*64, nhead)
        rel_bias = rel_bias.view(64, 64, self.nhead).permute(2, 0, 1)  # (nhead,64,64)

        attn_logits = attn_logits + rel_bias.unsqueeze(0)  # (B,nhead,64,64)

        # Optional mask
        if attn_mask is not None:
            # attn_mask should be broadcastable to (B,nhead,64,64)
            attn_logits = attn_logits.masked_fill(attn_mask, float("-inf"))

        # Attention weights
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, v)  # (B,nhead,64,head_dim)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, S, D)  # (B,64,D)

        # Final linear
        out = self.out_proj(out)
        return out
