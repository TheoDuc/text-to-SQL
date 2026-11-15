"""Tokenisation and DataLoader.

Turn each (context, question, answer) example into tensors of token ids that
the transformer can consume, grouped into padded batches.

Pipeline:
    1. Build the source text: schema (context) + question. The schema is
       required: the model cannot guess table and column names (e.g.
       `table_11869952_1`, `born_state`) from the question alone.
    2. Train a byte-level BPE tokenizer (like GPT-2) on the training set only,
       sources and SQL answers. BPE splits rare words into frequent sub-words,
       so a column name never seen in training can still be written. With a
       word-level vocabulary, 98% of the test answers would contain at least
       one unknown word.
    3. A Dataset that encodes one example:
         source -> [ids...]                 (encoder input, read all at once)
         target -> [<bos>, ids..., <eos>]   (generated token by token, so the
                                             decoder needs a start signal and
                                             must learn when to stop)
    4. A collate function that pads a batch and builds the decoder input /
       labels pair used for teacher forcing.

Run from the repository root (after data_split.py):
    python src/dataset.py
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SPLIT_DIR = Path("data/splits")
TOKENIZER_PATH = Path("data/tokenizer.json")

VOCAB_SIZE = 8000
# The longest source and target of the test set are ~100 tokens long, so 128
# truncates nothing.
MAX_SRC_LEN = 128
MAX_TGT_LEN = 128

# The position in this list is the token id.
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(len(SPECIAL_TOKENS))


def load_split(name: str) -> pd.DataFrame:
    """Load data/splits/{name}.json (columns: question, context, answer)."""
    return pd.read_json(SPLIT_DIR / f"{name}.json")


def build_source(context: str, question: str) -> str:
    """Build the encoder input from the schema and the question.

    "CREATE TABLE head (age INTEGER)", "How many heads are older than 56?"
    -> "CREATE TABLE head (age INTEGER) | How many heads are older than 56?"
    """
    return f"{context} | {question}"


def build_sources(df: pd.DataFrame) -> list[str]:
    """Build the encoder input of every row of a DataFrame."""
    return [build_source(c, q) for c, q in zip(df["context"], df["question"])]


def train_tokenizer(train_df: pd.DataFrame) -> Tokenizer:
    """Train a byte-level BPE tokenizer on the sources and SQL answers of train.

    Only train is used: fitting the tokenizer on val/test would leak their
    vocabulary, like fitting a scaler on the test set.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # Byte-level: every text is first converted to bytes, so any character can
    # be encoded. The decoder undoes this step.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    texts = build_sources(train_df) + train_df["answer"].tolist()
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def load_or_train_tokenizer(train_df: pd.DataFrame) -> Tokenizer:
    """Load the tokenizer from disk, or train and save it the first time.

    Delete data/tokenizer.json to force a new training.
    """
    if TOKENIZER_PATH.exists():
        return Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer = train_tokenizer(train_df)
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_PATH))
    return tokenizer


class Text2SQLDataset(Dataset):
    """One item = the token ids of one (source, target) pair.

    All the examples are tokenised once at creation (encode_batch runs in
    parallel), instead of once per epoch in __getitem__.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: Tokenizer) -> None:
        sources = tokenizer.encode_batch(build_sources(df))
        targets = tokenizer.encode_batch(df["answer"].tolist())

        self.src = [
            torch.tensor(enc.ids[:MAX_SRC_LEN], dtype=torch.long) for enc in sources
        ]
        # Truncate before adding <bos>/<eos> so that <eos> is never cut off.
        self.tgt = [
            torch.tensor(
                [BOS_ID] + enc.ids[: MAX_TGT_LEN - 2] + [EOS_ID], dtype=torch.long
            )
            for enc in targets
        ]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"src": self.src[idx], "tgt": self.tgt[idx]}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a list of examples into a batch, and prepare teacher forcing.

    Teacher forcing: during training, the decoder receives the TRUE previous
    tokens and must predict the next one at every position.
        tgt           = <bos> SELECT name FROM head <eos>
        decoder_input = <bos> SELECT name FROM head          (without the last token)
        labels        =       SELECT name FROM head <eos>    (without the first token)
    At position t, the decoder sees decoder_input[:t+1] and must predict labels[t].
    Without the shift, the model would only learn to copy its input.
    """
    src = pad_sequence(
        [item["src"] for item in batch], batch_first=True, padding_value=PAD_ID
    )
    tgt = pad_sequence(
        [item["tgt"] for item in batch], batch_first=True, padding_value=PAD_ID
    )
    return {"src": src, "decoder_input": tgt[:, :-1], "labels": tgt[:, 1:]}


def make_dataloaders(
    tokenizer: Tokenizer, batch_size: int = 64
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the train, val and test DataLoaders (only train is shuffled)."""
    return tuple(
        DataLoader(
            Text2SQLDataset(load_split(name), tokenizer),
            batch_size=batch_size,
            shuffle=(name == "train"),
            collate_fn=collate_fn,
        )
        for name in ["train", "val", "test"]
    )


def main() -> None:
    # Byte-level tokens contain characters such as "Ġ" (a space before a
    # word), which the default Windows console encoding cannot print.
    sys.stdout.reconfigure(encoding="utf-8")

    train_df = load_split("train")
    tokenizer = load_or_train_tokenizer(train_df)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    source = build_source(train_df["context"].iloc[0], train_df["question"].iloc[0])
    encoding = tokenizer.encode(source)
    print(f"\nSource: {source}")
    print(f"Tokens: {encoding.tokens[:20]} ...")
    print(f"Decoded back: {tokenizer.decode(encoding.ids)}")

    train_loader, val_loader, test_loader = make_dataloaders(tokenizer)
    print(
        f"\nBatches: train={len(train_loader)}, val={len(val_loader)}, "
        f"test={len(test_loader)}"
    )

    batch = next(iter(train_loader))
    for key, tensor in batch.items():
        print(f"{key:14s} shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    print("\nFirst example of the batch:")
    for key in ["decoder_input", "labels"]:
        text = tokenizer.decode(batch[key][0].tolist(), skip_special_tokens=False)
        print(f"{key:14s} {text}")


if __name__ == "__main__":
    main()
