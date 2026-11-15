"""Training loop.

One epoch = one pass over the training set. For each batch:
    forward (logits) -> loss -> backward (gradients) -> optimizer step.
After each epoch the model is evaluated on the validation set. The best model
(lowest validation loss) is saved; training stops early when the validation
loss has not improved for PATIENCE epochs.

Measured on the RTX 3050 Ti: ~80 s per epoch with bf16 mixed precision
(~2 min in full fp32), ~2 GB of GPU memory.

Run from the repository root (after dataset.py):
    python src/train.py
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PAD_ID, load_or_train_tokenizer, load_split, make_dataloaders
from model import Text2SQLTransformer, count_parameters

CHECKPOINT_DIR = Path("checkpoints")
SEED = 0

BATCH_SIZE = 64
MAX_EPOCHS = 15
PATIENCE = 3  # epochs without validation improvement before stopping
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000  # about one epoch
GRAD_CLIP = 1.0

MODEL_CONFIG = {
    "d_model": 256,
    "nhead": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 1024,
    "dropout": 0.1,
}


def set_seed(seed: int) -> None:
    """Make the run reproducible (same initial weights, same batch order)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Send a batch to the device and return (src, decoder_input, labels)."""
    return tuple(
        batch[key].to(device, non_blocking=True)
        for key in ("src", "decoder_input", "labels")
    )


def autocast(device: torch.device) -> torch.autocast:
    """bf16 mixed precision on GPU: ~1.6x faster on this card, same loss.

    Matrix products run in bfloat16 (16 bits) while the weights stay in
    float32. bf16 keeps the exponent range of float32, so unlike float16 it
    needs no loss scaling. Disabled on CPU.
    """
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> LambdaLR:
    """Learning rate schedule: linear warmup, then linear decay to 0.

        lr
        ^      /\\
        |     /   \\
        |    /       \\
        |   /           \\
        +--+---------------+--> step
          WARMUP          total_steps

    Warmup: at the start the weights are random and the gradients are large
    and noisy; a small learning rate avoids destabilising the transformer.
    Decay: at the end, smaller steps let the model settle into a minimum.
    The lambda returns a factor in [0, 1] that multiplies LEARNING_RATE.
    """

    def factor(step: int) -> float:
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        return max(0.0, (total_steps - step) / max(1, total_steps - WARMUP_STEPS))

    return LambdaLR(optimizer, factor)


def save_checkpoint(model: nn.Module, vocab_size: int, path: Path) -> None:
    """Save the weights and everything needed to rebuild the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": MODEL_CONFIG,
            "vocab_size": vocab_size,
        },
        path,
    )


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the non-padding target tokens.

    logits: (batch, tgt_len, vocab_size), labels: (batch, tgt_len).

    Padding is ignored: predicting <pad> after <eos> is trivial, and these
    many easy tokens would lower the loss without the SQL getting any better.
    The loss is computed in float32 for numerical stability under autocast.
    """
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=PAD_ID,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> float:
    """Train for one epoch and return the mean training loss over the batches.

    zero_grad() is called for every batch because PyTorch adds the new
    gradients to the existing ones instead of replacing them.
    """
    model.train()  # dropout ON
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        src, decoder_input, labels = move_batch(batch, device)

        optimizer.zero_grad()
        with autocast(device):
            loss = compute_loss(model(src, decoder_input), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()  # the schedule counts batches, not epochs

        total_loss += loss.item()  # a float, not the tensor and its graph

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return the mean loss over the batches of loader, without training.

    Dropout is turned off: it randomly drops neurons to fight overfitting
    during training, but would make the evaluation noisy and pessimistic.
    """
    model.eval()  # dropout OFF
    total_loss = 0.0

    for batch in tqdm(loader, desc="eval", leave=False):
        src, decoder_input, labels = move_batch(batch, device)
        with autocast(device):
            total_loss += compute_loss(model(src, decoder_input), labels).item()

    return total_loss / len(loader)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TF32: faster float32 matrix products on Ampere GPUs, negligible precision loss.
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Device: {device}")

    tokenizer = load_or_train_tokenizer(load_split("train"))
    vocab_size = tokenizer.get_vocab_size()
    train_loader, val_loader, _ = make_dataloaders(tokenizer, batch_size=BATCH_SIZE)

    model = Text2SQLTransformer(vocab_size=vocab_size, **MODEL_CONFIG).to(device)
    print(f"Trainable parameters: {count_parameters(model) / 1e6:.1f} M")

    # AdamW = Adam with a weight decay applied directly to the weights (a light
    # L2-like regularisation). betas=(0.9, 0.98) as in the original paper.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = make_scheduler(optimizer, total_steps=MAX_EPOCHS * len(train_loader))

    best_val_loss = math.inf
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val_loss
        print(
            f"Epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - start:.0f}s"
            + ("  <- best, saved" if improved else "")
        )

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, vocab_size, CHECKPOINT_DIR / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs: early stopping.")
                break

    # Keep the curves to plot them in the README.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (CHECKPOINT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
