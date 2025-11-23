import torch
import torch.nn as nn
# import torch.nn.functional as F 
from transformer_encoder import RelativeTransformerEncoderLayer

NUM_PIECE_TYPES = 12 
HISTORY_STEPS = 8
PIECE_HIST_DIM = HISTORY_STEPS * NUM_PIECE_TYPES
META_DIM = 16
RAW_FEAT_DIM = PIECE_HIST_DIM + META_DIM 
MODEL_DIM = 256 

class ChessFormer(nn.Module):
    def __init__(
        self,
        d_model: int = MODEL_DIM,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(RAW_FEAT_DIM, d_model)
        self.square_embed = nn.Embedding(64, d_model)

        # --- Use our relative encoder layer instead of nn.TransformerEncoderLayer ---
        # encoder_layer = RelativeTransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        # )
        self.encoder = nn.ModuleList([
            RelativeTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])


        self.policy_q = nn.Linear(d_model, d_model)
        self.policy_k = nn.Linear(d_model, d_model)

        self.promo_head = nn.Linear(d_model, 5)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, raw_feats, src_mask=None):
        B, S, F = raw_feats.shape
        assert S == 64
        assert F == RAW_FEAT_DIM

        x = self.feature_proj(raw_feats)  # (B,64,d_model)

        square_ids = torch.arange(64, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.square_embed(square_ids)

        # Pass through N relative transformer layers
        for layer in self.encoder:
            x = layer(x, src_mask)

        q = self.policy_q(x)
        k = self.policy_k(x)

        policy_logits = torch.matmul(q, k.transpose(-2, -1))  # (B,64,64)
        promo_logits = self.promo_head(x)

        x_global = x.mean(dim=1)
        value_logits = self.value_head(x_global)

        return policy_logits, promo_logits, value_logits
