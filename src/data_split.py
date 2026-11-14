"""Split the dataset into train / validation / test sets.

Goal: evaluate the model on schemas it has never seen during training.

A random split would leak information: many questions share the same tables,
so the model would see a table in train and again in test, and the test score
would measure memorisation rather than generalisation to new schemas.

Strategy:
    1. Extract the table names used by each example (from the `context`).
    2. Build groups: two examples are in the same group if they share a table,
       directly or through a chain of other examples (union-find).
    3. Split with StratifiedGroupKFold into 10 folds of ~10%:
         - a whole group always lands in a single fold;
         - the share of multi-table queries is kept as even as possible.
    4. Keep the 2 folds closest to the global multi-table share for test and
       validation, the 8 others for train (~80 / 10 / 10).

Run from the repository root:
    python src/data_split.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_PATH = Path("data/sql_create_context_v4.json")
SPLIT_DIR = Path("data/splits")
N_FOLDS = 10
SEED = 0

TABLE_PATTERN = re.compile(r"CREATE TABLE (\w+)", flags=re.IGNORECASE)


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset and remove duplicated questions."""
    return pd.read_json(path).drop_duplicates(subset="question", ignore_index=True)


def extract_table_names(context: str) -> list[str]:
    """Return the sorted, de-duplicated table names found in a context.

    "CREATE TABLE head (age INTEGER); CREATE TABLE management (x VARCHAR)"
    -> ["head", "management"]
    """
    return sorted(set(TABLE_PATTERN.findall(context)))


class UnionFind:
    """Disjoint sets: merges items that are linked together.

    Each item points to a parent. The root of a chain (an item that is its own
    parent) represents the whole group.
        - find(x):     follow the parents up to the root of x's group.
        - union(a, b): attach the root of b under the root of a.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            # Path compression: point to the grandparent to shorten the chain.
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def build_groups(tables_per_row: pd.Series) -> pd.Series:
    """Give each example the id of its group (the name of the root table)."""
    uf = UnionFind()
    for tables in tables_per_row:
        for table in tables[1:]:
            uf.union(tables[0], table)
    return tables_per_row.map(lambda tables: uf.find(tables[0]))


def add_split_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns used by the split: tables, group, n_tables, multi_table.

    We stratify on multi_table (0/1) rather than on n_tables because the
    categories with 3, 4 or 5 tables contain too few examples.
    """
    df = df.copy()
    df["tables"] = df["context"].map(extract_table_names)
    df["group"] = build_groups(df["tables"])
    df["n_tables"] = df["tables"].str.len()
    df["multi_table"] = (df["n_tables"] > 1).astype(int)
    return df


def describe_groups(df: pd.DataFrame, top: int = 5) -> None:
    """Print the number of groups and the biggest ones."""
    stats = (
        df.groupby("group")
        .agg(size=("question", "size"), multi_share=("multi_table", "mean"))
        .sort_values("size", ascending=False)
    )
    print(f"{len(stats)} groups")
    print(stats.head(top).round(3), end="\n\n")


def make_folds(df: pd.DataFrame) -> list[np.ndarray]:
    """Split the row positions into N_FOLDS folds that respect the groups.

    StratifiedGroupKFold.split() yields (train_idx, test_idx) pairs like a
    cross-validation. The N_FOLDS test parts form a partition of the data, so
    we only keep them.
    """
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return [test_idx for _, test_idx in sgkf.split(df, df["multi_table"], df["group"])]


def choose_val_test_folds(df: pd.DataFrame, folds: list[np.ndarray]) -> tuple[int, int]:
    """Return (val_fold, test_fold): the 2 folds closest to the global multi-table share.

    Stratification is only approximate with groups, so some folds hold far
    more multi-table queries than others. The closest fold goes to test, so
    that the final evaluation is the most representative of the dataset; the
    second closest goes to validation.
    """
    global_share = df["multi_table"].mean()
    fold_stats = pd.DataFrame(
        {
            "size": [len(idx) for idx in folds],
            "multi_share": [df["multi_table"].iloc[idx].mean() for idx in folds],
        }
    )
    fold_stats["gap"] = (fold_stats["multi_share"] - global_share).abs()

    print(f"Global multi-table share: {global_share:.4f}")
    print(fold_stats.round(4), end="\n\n")

    test_fold, val_fold = fold_stats["gap"].nsmallest(2).index
    return int(val_fold), int(test_fold)


def assign_split(
    df: pd.DataFrame, folds: list[np.ndarray], val_fold: int, test_fold: int
) -> pd.DataFrame:
    """Add a `split` column with the values "train", "val" or "test"."""
    split = np.full(len(df), "train", dtype=object)
    split[folds[val_fold]] = "val"
    split[folds[test_fold]] = "test"
    return df.assign(split=split)


def check_split(df: pd.DataFrame) -> None:
    """Check that no group is shared between splits and print split statistics."""
    splits_per_group = df.groupby("group")["split"].nunique()
    assert (splits_per_group == 1).all(), "Some groups appear in several splits"

    stats = df.groupby("split").agg(n=("question", "size"), multi_share=("multi_table", "mean"))
    stats["share"] = stats["n"] / len(df)
    print(stats.loc[["train", "val", "test"], ["n", "share", "multi_share"]].round(4), end="\n\n")


def save_splits(df: pd.DataFrame, out_dir: Path) -> None:
    """Save train.json, val.json and test.json (question, context, answer only)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        part = df.loc[df["split"] == name, ["question", "context", "answer"]]
        path = out_dir / f"{name}.json"
        part.to_json(path, orient="records", indent=2, force_ascii=False)
        print(f"Saved {len(part)} rows to {path}")


def main() -> None:
    df = add_split_columns(load_data(DATA_PATH))
    describe_groups(df)

    folds = make_folds(df)
    val_fold, test_fold = choose_val_test_folds(df, folds)
    df = assign_split(df, folds, val_fold, test_fold)

    check_split(df)
    save_splits(df, SPLIT_DIR)


if __name__ == "__main__":
    main()
