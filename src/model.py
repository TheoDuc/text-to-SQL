"""Encoder-decoder transformer that maps (schema + question) to SQL.

Architecture (same family as "Attention is all you need" and T5):

    src ids ──> embedding + positions ──> ENCODER ──> memory
                                                        │ (cross-attention)
    decoder_input ids ──> embedding + positions ──> DECODER ──> linear ──> logits

    - The encoder reads the whole source at once: every token can look at
      every other token. Nothing is predicted on this side, so no causal mask.
    - The decoder produces the SQL. At position t it may only look at the
      target tokens 0..t (causal mask), plus the whole encoder output.
    - The final linear layer turns each decoder vector into one score per
      vocabulary token (the logits). The predicted token is the argmax.

nn.Transformer provides the layers; this module builds everything around
them: embeddings, positional encoding, masks and output layer.

Run from the repository root (after dataset.py):
    python src/model.py
"""

import math
import warnings

import torch
from torch import nn

from dataset import MAX_SRC_LEN, MAX_TGT_LEN, PAD_ID

# nn.Transformer uses an internal speed-up (nested tensors) in eval mode that
# prints a harmless "prototype stage" warning.
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors")


class PositionalEncoding(nn.Module):
    """Add a fixed sinusoidal position signal to the token embeddings.

    Attention has no notion of order: without this, "SELECT a FROM b" and
    "FROM b SELECT a" would look the same to the model. Each position gets a
    unique vector built with sines and cosines of different frequencies:

        PE[pos, 2i]     = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i + 1] = cos(pos / 10000^(2i / d_model))

    Why not add the raw position (0, 1, ..., 127)? Its scale would drown the
    embeddings (values around 1). Sines and cosines stay in [-1, 1], give a
    unique vector per position, and PE[pos + k] is a fixed rotation of PE[pos],
    which makes relative offsets easy to learn.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        two_i = torch.arange(0, d_model, 2, dtype=torch.float)
        angles = position / 10000 ** (two_i / d_model)  # (max_len, d_model / 2)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        # A buffer is saved with the model and moved to the GPU with it, but it
        # is not trained. Shape (1, max_len, d_model) to broadcast over the batch.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> same shape, with positions added."""
        return x + self.pe[:, : x.size(1)]


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Mask that prevents each target position from attending to future positions.

    Returns a (size, size) boolean tensor where True means "forbidden" (row =
    position that looks, column = position looked at). For size = 4:

        [[False,  True,  True,  True],
         [False, False,  True,  True],
         [False, False, False,  True],
         [False, False, False, False]]

    Without it, teacher forcing would let position t read token t + 1, which
    is exactly the label it must predict: the model would learn to copy it,
    and fail at generation time when the future does not exist yet.
    """
    return torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1
    )


class Text2SQLTransformer(nn.Module):
    """Encoder-decoder transformer that maps (schema + question) to SQL."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # One embedding shared by source and target: same tokenizer, and the
        # SQL copies many words from the source (column names, values).
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.positional_encoding = PositionalEncoding(
            d_model, max(MAX_SRC_LEN, MAX_TGT_LEN)
        )
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Weight tying: the output layer reuses the embedding matrix. Fewer
        # parameters, and one vector per token for both input and output.
        self.output = nn.Linear(d_model, vocab_size)
        self.output.weight = self.embedding.weight
        # nn.Embedding draws its weights from N(0, 1). Shared with the output
        # layer, this gives huge logits at the start (initial loss ~85 instead
        # of ~9). With std = 1/sqrt(d_model) the logits start small, and
        # embed() multiplies by sqrt(d_model) to bring the inputs back to ~1.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**-0.5)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].zero_()

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (batch, seq_len) -> (batch, seq_len, d_model)."""
        x = self.embedding(ids) * math.sqrt(self.d_model)
        return self.dropout(self.positional_encoding(x))

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Run the encoder only: (batch, src_len) -> memory (batch, src_len, d_model)."""
        return self.transformer.encoder(
            self.embed(src), src_key_padding_mask=(src == PAD_ID)
        )

    def decode(
        self, decoder_input: torch.Tensor, memory: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor:
        """Run the decoder and the output layer: -> logits (batch, tgt_len, vocab)."""
        out = self.transformer.decoder(
            self.embed(decoder_input),
            memory,
            tgt_mask=causal_mask(decoder_input.size(1), decoder_input.device),
            tgt_key_padding_mask=(decoder_input == PAD_ID),
            memory_key_padding_mask=(src == PAD_ID),
        )
        return self.output(out)

    def forward(self, src: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """Training pass (teacher forcing).

        src:           (batch, src_len)  token ids of schema + question
        decoder_input: (batch, tgt_len)  <bos> + SQL token ids (without the last)
        returns:       (batch, tgt_len, vocab_size) logits

        Generation (step 5) calls encode() once, then decode() once per new
        token, reusing the same memory.
        """
        return self.decode(decoder_input, self.encode(src), src)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    from dataset import load_or_train_tokenizer, load_split, make_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"\nCausal mask (size 4):\n{causal_mask(4, device)}")

    tokenizer = load_or_train_tokenizer(load_split("train"))
    train_loader, _, _ = make_dataloaders(tokenizer)

    model = Text2SQLTransformer(vocab_size=tokenizer.get_vocab_size()).to(device)
    print(f"\nTrainable parameters: {count_parameters(model) / 1e6:.1f} M")

    batch = next(iter(train_loader))
    src = batch["src"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    labels = batch["labels"].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(src, decoder_input)
    print(f"logits shape: {tuple(logits.shape)}  (batch, tgt_len, vocab_size)")

    # With random weights the model is roughly uniform over the vocabulary,
    # so the loss should be close to ln(vocab_size).
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=PAD_ID
    )
    print(
        f"Initial loss: {loss.item():.2f}  "
        f"(ln(vocab_size) = {math.log(logits.size(-1)):.2f})"
    )


if __name__ == "__main__":
    main()
