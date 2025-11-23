# chess_former.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm  # <-- progress bars

from model import ChessFormer
from loader import train_loader, val_loader

# ---- DEVICE ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- MODEL + OPTIMIZER ----
num_epochs = 10
model = ChessFormer().to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(num_epochs):
    # =======================
    #        TRAIN
    # =======================
    model.train()
    total_train_loss = 0.0
    total_train_policy_loss = 0.0
    total_train_value_loss = 0.0
    total_train_samples = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]")

    for x, from_sq, to_sq, promo, value in train_bar:
        x = x.to(device)
        from_sq = from_sq.to(device)
        to_sq = to_sq.to(device)
        promo = promo.to(device)
        value = value.to(device)

        policy_logits, promo_logits, value_logits = model(x)

        B = x.size(0)

        # ---- Policy loss ----
        policy_logits_flat = policy_logits.view(B, -1)  # (B, 4096)
        policy_target = from_sq * 64 + to_sq            # (B,)
        loss_policy = F.cross_entropy(policy_logits_flat, policy_target)

        # ---- Value loss ----
        loss_value = F.cross_entropy(value_logits, value)

        # ---- Total loss ----
        loss = loss_policy + loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Accumulate stats ----
        total_train_loss += loss.item() * B
        total_train_policy_loss += loss_policy.item() * B
        total_train_value_loss += loss_value.item() * B
        total_train_samples += B

        # Update tqdm bar
        avg_batch_loss = total_train_loss / total_train_samples
        train_bar.set_postfix(
            loss=f"{avg_batch_loss:.4f}",
            pol=f"{loss_policy.item():.4f}",
            val=f"{loss_value.item():.4f}"
        )

    avg_train_loss = total_train_loss / total_train_samples
    avg_train_policy_loss = total_train_policy_loss / total_train_samples
    avg_train_value_loss = total_train_value_loss / total_train_samples

    # =======================
    #      VALIDATION
    # =======================
    model.eval()
    total_val_loss = 0.0
    total_val_policy_loss = 0.0
    total_val_value_loss = 0.0
    total_val_samples = 0

    # For top-1 move accuracy
    total_correct_moves = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]")

    with torch.no_grad():
        for x, from_sq, to_sq, promo, value in val_bar:
            x = x.to(device)
            from_sq = from_sq.to(device)
            to_sq = to_sq.to(device)
            promo = promo.to(device)
            value = value.to(device)

            policy_logits, promo_logits, value_logits = model(x)

            B = x.size(0)

            # ---- Policy loss ----
            policy_logits_flat = policy_logits.view(B, -1)
            policy_target = from_sq * 64 + to_sq
            loss_policy = F.cross_entropy(policy_logits_flat, policy_target)

            # ---- Value loss ----
            loss_value = F.cross_entropy(value_logits, value)

            # ---- Total loss ----
            loss = loss_policy + loss_value

            # ---- Top-1 move accuracy ----
            pred_moves = policy_logits_flat.argmax(dim=-1)  # (B,)
            correct = (pred_moves == policy_target).sum().item()
            total_correct_moves += correct

            total_val_loss += loss.item() * B
            total_val_policy_loss += loss_policy.item() * B
            total_val_value_loss += loss_value.item() * B
            total_val_samples += B

            # Update tqdm bar
            current_acc = total_correct_moves / total_val_samples if total_val_samples > 0 else 0.0
            val_bar.set_postfix(
                loss=f"{(total_val_loss / total_val_samples):.4f}",
                acc=f"{current_acc*100:.2f}%"
            )

    avg_val_loss = total_val_loss / total_val_samples
    avg_val_policy_loss = total_val_policy_loss / total_val_samples
    avg_val_value_loss = total_val_value_loss / total_val_samples
    val_policy_acc = total_correct_moves / total_val_samples

    # =======================
    #      EPOCH SUMMARY
    # =======================
    print(
        f"Epoch {epoch+1}/{num_epochs} "
        f"| train_loss={avg_train_loss:.4f} "
        f"(pol={avg_train_policy_loss:.4f}, val={avg_train_value_loss:.4f}) "
        f"| val_loss={avg_val_loss:.4f} "
        f"(pol={avg_val_policy_loss:.4f}, val={avg_val_value_loss:.4f}) "
        f"| val_policy_acc={val_policy_acc*100:.2f}%"
    )
