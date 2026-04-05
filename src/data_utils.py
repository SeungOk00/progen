import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class ProteinDataset(Dataset):
    def __init__(self, lines: list[str], tokenizer: Tokenizer):
        self.lines = lines
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode(self.lines[idx])
        input_ids = torch.tensor(encoding.ids, dtype=torch.long)
        attention_mask = torch.tensor(encoding.attention_mask, dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_sequence_lines(file_path: str) -> tuple[list[str], list[str]]:
    lines = []
    prefixes = set()

    with open(file_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            prefix = re.match(r"<\|.*\|>", line).group(0)
            prefixes.add(prefix)
            lines.append(line)

    return lines, sorted(prefixes)


def load_tokenizer(model_path: str, max_length: int = 1024) -> Tokenizer:
    tokenizer_path = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_pretrained(model_path)

    tokenizer.no_padding()
    tokenizer.enable_padding(
        direction="right",
        pad_id=0,
        pad_token="<|pad|>",
        length=max_length,
    )
    tokenizer.enable_truncation(max_length=max_length)
    return tokenizer


def build_datasets(
    train_file: str,
    val_file: str,
    tokenizer: Tokenizer,
) -> tuple[ProteinDataset, ProteinDataset, list[str], int]:
    train_lines, train_prefixes = load_sequence_lines(train_file)
    val_lines, val_prefixes = load_sequence_lines(val_file)

    if train_prefixes != val_prefixes:
        raise ValueError("Prefixes in train and validation data must be the same")

    num_added_tokens = tokenizer.add_tokens(train_prefixes)
    return (
        ProteinDataset(train_lines, tokenizer),
        ProteinDataset(val_lines, tokenizer),
        train_prefixes,
        num_added_tokens,
    )


def build_dataset_from_lines(
    lines: list[str],
    tokenizer: Tokenizer,
    prefixes: list[str] | None = None,
) -> tuple[ProteinDataset, list[str], int]:
    if prefixes is None:
        prefixes = sorted({re.match(r"<\|.*\|>", line).group(0) for line in lines})

    num_added_tokens = tokenizer.add_tokens(prefixes)
    return ProteinDataset(lines, tokenizer), prefixes, num_added_tokens


def make_kfold_splits(
    lines: list[str],
    num_folds: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    if len(lines) < num_folds:
        raise ValueError("Number of folds cannot exceed number of sequences")

    shuffled = list(lines)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    fold_indices = np.array_split(np.arange(len(shuffled)), num_folds)

    folds = []
    for fold_idx in range(num_folds):
        val_index_set = set(fold_indices[fold_idx].tolist())
        train_lines = [line for idx, line in enumerate(shuffled) if idx not in val_index_set]
        val_lines = [line for idx, line in enumerate(shuffled) if idx in val_index_set]
        folds.append((train_lines, val_lines))
    return folds
