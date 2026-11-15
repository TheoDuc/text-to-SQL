"""Generate SQL on the test set and measure the results.

Training used teacher forcing: the decoder always received the TRUE previous
tokens. At generation time there is no true answer: the model must feed its
OWN previous predictions back as input, one token at a time (autoregressive
decoding). Greedy decoding keeps, at each step, the most likely token.

    step 0:  <bos>                              -> SELECT
    step 1:  <bos> SELECT                       -> name
    step 2:  <bos> SELECT name                  -> FROM
    ...                                         -> <eos>  (stop)

Metrics:
    - exact match: the predicted SQL equals the expected SQL (after
      normalising case and spaces). Strict: `points = 13` vs `points = "13"`
      counts as wrong.
    - valid SQL:   the prediction runs without error on an empty SQLite
      database built from the schema (right syntax, existing tables and
      columns). It does not mean the answer is right.
Both are reported overall and separately for single- and multi-table queries.

Run from the repository root (after train.py):
    python src/evaluation.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    BOS_ID,
    EOS_ID,
    MAX_TGT_LEN,
    PAD_ID,
    Text2SQLDataset,
    collate_fn,
    load_or_train_tokenizer,
    load_split,
)
from model import Text2SQLTransformer
from train import CHECKPOINT_DIR, autocast

BATCH_SIZE = 128
PREDICTIONS_PATH = CHECKPOINT_DIR / "test_predictions.csv"


def load_model(path: Path, device: torch.device) -> Text2SQLTransformer:
    """Rebuild the model from a checkpoint saved by train.py."""
    checkpoint = torch.load(path, map_location=device)
    model = Text2SQLTransformer(checkpoint["vocab_size"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def ids_to_sql(ids: list[int], tokenizer) -> str:
    """Cut a generated sequence at the first <eos> and decode it to text."""
    if EOS_ID in ids:
        ids = ids[: ids.index(EOS_ID)]
    return tokenizer.decode(ids)


@torch.no_grad()
def greedy_decode(model: Text2SQLTransformer, src: torch.Tensor) -> torch.Tensor:
    """Generate the SQL token ids of a whole batch, one token at a time.

    src:     (batch, src_len)
    returns: (batch, generated_len) generated ids, WITHOUT the leading <bos>.
             After its <eos>, a sequence is filled with <pad>.

    The whole batch is generated in parallel: at each step, every sequence
    gets one new token. Sequences that have already produced <eos> keep
    receiving <pad> until every sequence is finished.

    The source never changes, so it is encoded once and the memory is reused
    at every step. The decoder, however, recomputes the whole prefix at each
    step and only the last position is used: the earlier positions predict
    tokens that were already chosen (and, thanks to the causal mask, their
    outputs are the same as at the previous steps). Only the last position
    predicts the NEW token to append. Caching those past states (KV cache)
    would avoid the recomputation; nn.Transformer does not provide it.
    """
    batch_size, device = src.size(0), src.device
    memory = model.encode(src)

    generated = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(MAX_TGT_LEN - 1):
        logits = model.decode(generated, memory, src)  # (batch, cur_len, vocab)
        next_token = logits[:, -1].argmax(dim=-1).masked_fill(finished, PAD_ID)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished |= next_token == EOS_ID
        if finished.all():
            break

    return generated[:, 1:]


def normalize_sql(sql: str) -> str:
    """Lowercase and collapse whitespace, so that formatting does not count.

    "SELECT  name\nFROM Head" -> "select name from head"
    """
    return " ".join(sql.lower().split())


def is_valid_sql(context: str, sql: str) -> bool:
    """True if `sql` runs without error on an empty database built from `context`.

    This only checks that the query is executable (syntax, existing tables and
    columns), not that it answers the question: the tables are empty, so there
    is no result to compare. For "How many heads are older than 56?",
    `SELECT COUNT(*) FROM head WHERE age < 56` is valid but wrong.
    """
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(context)
        connection.execute(sql).fetchall()
        return True
    except sqlite3.Error:
        return False
    finally:
        connection.close()


def predict(
    model: Text2SQLTransformer, df: pd.DataFrame, tokenizer, device
) -> list[str]:
    """Generate the SQL of every row of df."""
    loader = DataLoader(
        Text2SQLDataset(df, tokenizer), batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    predictions = []
    for batch in tqdm(loader, desc="generate"):
        with autocast(device):
            generated = greedy_decode(model, batch["src"].to(device))
        predictions.extend(ids_to_sql(ids, tokenizer) for ids in generated.tolist())
    return predictions


def report(df: pd.DataFrame) -> None:
    """Print the metrics overall and by number of tables."""
    df = df.assign(
        query_type=df["context"]
        .str.count("CREATE TABLE")
        .gt(1)
        .map({False: "single-table", True: "multi-table"})
    )
    overall = df[["exact_match", "valid_sql"]].mean().to_frame("overall").T
    overall["n"] = len(df)
    by_type = df.groupby("query_type").agg(
        exact_match=("exact_match", "mean"),
        valid_sql=("valid_sql", "mean"),
        n=("exact_match", "size"),
    )
    print(pd.concat([overall, by_type]).round(3), end="\n\n")


def show_errors(df: pd.DataFrame, n: int = 8) -> None:
    """Print a few wrong predictions, to understand what the model gets wrong."""
    errors = df[~df["exact_match"]]
    for _, row in errors.sample(min(n, len(errors)), random_state=0).iterrows():
        print(f"Q:    {row['question']}")
        print(f"gold: {row['answer']}")
        print(f"pred: {row['prediction']}\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_or_train_tokenizer(load_split("train"))
    model = load_model(CHECKPOINT_DIR / "best_model.pt", device)
    test_df = load_split("test")

    test_df["prediction"] = predict(model, test_df, tokenizer, device)
    test_df["exact_match"] = [
        normalize_sql(pred) == normalize_sql(gold)
        for pred, gold in zip(test_df["prediction"], test_df["answer"])
    ]
    test_df["valid_sql"] = [
        is_valid_sql(context, pred)
        for context, pred in zip(test_df["context"], test_df["prediction"])
    ]

    report(test_df)
    show_errors(test_df)

    test_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"All predictions saved to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
