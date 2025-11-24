# chess_former.py

import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import ChessFormer
from dataset import PGN_Moves
from loader import get_train_val_files

# -------------------
#      CONFIG
# -------------------

NUM_EPOCHS = 1             # single full pass over train_files
BATCH_SIZE = 64
MAX_MOVES_PER_FILE = 100_000  # optional cap per file; set None to use all moves

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_chessformer.pt")
TRAIN_STATE_PATH = os.path.join(CHECKPOINT_DIR, "training_checkpoint.pt")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

VALIDATE_EVERY_N_FILES = 50   # 🔥 run val + save model after each 50 train files

# -------------------
#      DEVICE
# -------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------
#   MODEL + OPTIM
# -------------------

model = ChessFormer().to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

best_val_loss = float("inf")
start_epoch = 0
start_file_idx = 0  # index in train_files to resume from within an epoch

# -------------------
#   RESUME LOGIC
# -------------------

if os.path.exists(TRAIN_STATE_PATH):
    print("🔄 Found existing training checkpoint, trying to resume...")
    checkpoint = torch.load(TRAIN_STATE_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    start_epoch = checkpoint.get("epoch", 0)
    start_file_idx = checkpoint.get("file_idx", 0)

    print(
        f"Resuming from epoch {start_epoch}/{NUM_EPOCHS}, "
        f"file index {start_file_idx}, best_val_loss={best_val_loss:.4f}"
    )
else:
    print("🚀 No training checkpoint found, starting from scratch.")

# -------------------
#   TRAIN / VAL FILES
# -------------------

train_files, val_files = get_train_val_files(block_size=50, val_per_block=9)
print(f"Train files: {len(train_files)} | Val files: {len(val_files)}")


# -------------------
#   VALIDATION HELPER
# -------------------

def run_validation(model, val_files, epoch, step_label=""):
    """
    Run full validation over all val_files and print summary.
    Returns: (avg_val_loss, avg_val_policy_loss, avg_val_value_loss, val_policy_acc)
    """
    model.eval()
    total_val_loss = 0.0
    total_val_policy_loss = 0.0
    total_val_value_loss = 0.0
    total_val_samples = 0
    total_correct_moves = 0

    print(f"\n[VAL] Epoch {epoch + 1} {step_label}")
    for v_idx, v_pgn_path in enumerate(val_files, start=1):
        print(f"  -> Validating on file {v_idx}/{len(val_files)}: {os.path.basename(v_pgn_path)}")

        val_dataset = PGN_Moves(
            v_pgn_path,
            max_games=None,
            max_moves=MAX_MOVES_PER_FILE,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1} [val] file {v_idx}/{len(val_files)}",
        )

        with torch.no_grad():
            for x, from_sq, to_sq, promo, value in val_bar:
                x = x.to(device)
                from_sq = from_sq.to(device)
                to_sq = to_sq.to(device)
                promo = promo.to(device)
                value = value.to(device)

                policy_logits, promo_logits, value_logits = model(x)
                B = x.size(0)

                policy_logits_flat = policy_logits.view(B, -1)
                policy_target = from_sq * 64 + to_sq
                loss_policy = F.cross_entropy(policy_logits_flat, policy_target)
                loss_value = F.cross_entropy(value_logits, value)
                loss = loss_policy + loss_value

                pred_moves = policy_logits_flat.argmax(dim=-1)
                correct = (pred_moves == policy_target).sum().item()
                total_correct_moves += correct

                total_val_loss += loss.item() * B
                total_val_policy_loss += loss_policy.item() * B
                total_val_value_loss += loss_value.item() * B
                total_val_samples += B

                current_acc = (
                    total_correct_moves / total_val_samples
                    if total_val_samples > 0
                    else 0.0
                )
                val_bar.set_postfix(
                    loss=f"{(total_val_loss / total_val_samples):.4f}",
                    acc=f"{current_acc * 100:.2f}%",
                )

        del val_dataset, val_loader
        torch.cuda.empty_cache()

    avg_val_loss = total_val_loss / total_val_samples
    avg_val_policy_loss = total_val_policy_loss / total_val_samples
    avg_val_value_loss = total_val_value_loss / total_val_samples
    val_policy_acc = total_correct_moves / total_val_samples

    print(
        f"[VAL SUMMARY] Epoch {epoch + 1} {step_label} "
        f"| val_loss={avg_val_loss:.4f} "
        f"(pol={avg_val_policy_loss:.4f}, val={avg_val_value_loss:.4f}) "
        f"| val_policy_acc={val_policy_acc * 100:.2f}%"
    )

    return avg_val_loss, avg_val_policy_loss, avg_val_value_loss, val_policy_acc


# -------------------
#   TRAINING LOOP
# -------------------

for epoch in range(start_epoch, NUM_EPOCHS):
    print("\n" + "#" * 60)
    print(f"### EPOCH {epoch + 1}/{NUM_EPOCHS} ###")
    print("#" * 60)

    model.train()
    total_train_loss = 0.0
    total_train_policy_loss = 0.0
    total_train_value_loss = 0.0
    total_train_samples = 0

    print("\n[TRAIN]")
    for file_idx, pgn_path in enumerate(train_files):
        # Skip files we've already processed if resuming mid-epoch
        if epoch == start_epoch and file_idx < start_file_idx:
            continue

        display_idx = file_idx + 1  # human-readable (1-based)
        print(f"  -> Training on file {display_idx}/{len(train_files)}: {os.path.basename(pgn_path)}")

        train_dataset = PGN_Moves(
            pgn_path,
            max_games=None,
            max_moves=MAX_MOVES_PER_FILE,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1} [train] file {display_idx}/{len(train_files)}",
        )

        for x, from_sq, to_sq, promo, value in train_bar:
            x = x.to(device)
            from_sq = from_sq.to(device)
            to_sq = to_sq.to(device)
            promo = promo.to(device)
            value = value.to(device)

            policy_logits, promo_logits, value_logits = model(x)
            B = x.size(0)

            policy_logits_flat = policy_logits.view(B, -1)
            policy_target = from_sq * 64 + to_sq
            loss_policy = F.cross_entropy(policy_logits_flat, policy_target)
            loss_value = F.cross_entropy(value_logits, value)
            loss = loss_policy + loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * B
            total_train_policy_loss += loss_policy.item() * B
            total_train_value_loss += loss_value.item() * B
            total_train_samples += B

            avg_batch_loss = total_train_loss / total_train_samples
            train_bar.set_postfix(
                loss=f"{avg_batch_loss:.4f}",
                pol=f"{loss_policy.item():.4f}",
                val=f"{loss_value.item():.4f}",
            )

        del train_dataset, train_loader
        torch.cuda.empty_cache()

        # ---- VALIDATE + SAVE MODEL AFTER EACH 50 FILES OR AT THE END ----
        if (display_idx % VALIDATE_EVERY_N_FILES == 0) or (display_idx == len(train_files)):
            step_label = f"(after {display_idx} train files)"
            avg_val_loss, _, _, _ = run_validation(model, val_files, epoch, step_label=step_label)

            # 1) Save best model (based on val loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"✔ Model improved — saved BEST checkpoint to {BEST_MODEL_PATH}")

            # 2) Save training state so we can resume from here
            if display_idx == len(train_files):
                # Finished this epoch: next time start at next epoch, file_idx=0
                next_epoch = epoch + 1
                next_file_idx_to_start = 0
            else:
                # In the middle of this epoch: resume from next file
                next_epoch = epoch
                next_file_idx_to_start = display_idx  # since display_idx = file_idx+1

            torch.save(
                {
                    "epoch": next_epoch,
                    "file_idx": next_file_idx_to_start,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                TRAIN_STATE_PATH,
            )
            print(
                f"💾 Saved training state to {TRAIN_STATE_PATH} "
                f"(next_epoch={next_epoch}, next_file_idx={next_file_idx_to_start})"
            )

    # (Optional) epoch-level train summary
    if total_train_samples > 0:
        avg_train_loss = total_train_loss / total_train_samples
        avg_train_policy_loss = total_train_policy_loss / total_train_samples
        avg_train_value_loss = total_train_value_loss / total_train_samples
        print(
            f"\n[TRAIN SUMMARY] Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"| train_loss={avg_train_loss:.4f} "
            f"(pol={avg_train_policy_loss:.4f}, val={avg_train_value_loss:.4f})"
        )

print("\n✅ Training complete.")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model weights saved at: {BEST_MODEL_PATH}")