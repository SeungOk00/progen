import os
from dataclasses import dataclass

import numpy as np
from Bio import SeqIO


@dataclass
class PreparedSplit:
    train_sequences: list[str]
    test_sequences: list[str]


def fasta_to_tagged_sequences(
    input_file_name: str,
    bidirectional: bool = False,
) -> list[str]:
    base_name = os.path.basename(input_file_name).lower()
    label = os.path.splitext(base_name)[0]

    parsed_sequences = []
    with open(input_file_name, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence = str(record.seq)
            parsed_sequences.append(f"<|{label}|>1{sequence}2")
            if bidirectional:
                parsed_sequences.append(f"<|{label}|>2{sequence[::-1]}1")

    return parsed_sequences


def split_sequences(
    sequences: list[str],
    train_split_ratio: float,
    rng: np.random.Generator,
) -> PreparedSplit:
    if not 0 <= train_split_ratio <= 1:
        raise ValueError("Train-test split ratio must be between 0 and 1.")

    shuffled = list(sequences)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_split_ratio)
    return PreparedSplit(
        train_sequences=shuffled[:split_idx],
        test_sequences=shuffled[split_idx:],
    )


def prepare_dataset_splits(
    input_files: list[str],
    train_split_ratio: float,
    seed: int,
    bidirectional: bool = False,
) -> PreparedSplit:
    rng = np.random.default_rng(seed)
    train_sequences = []
    test_sequences = []

    for input_file in input_files:
        sequences = fasta_to_tagged_sequences(input_file, bidirectional=bidirectional)
        split = split_sequences(sequences, train_split_ratio=train_split_ratio, rng=rng)
        train_sequences.extend(split.train_sequences)
        test_sequences.extend(split.test_sequences)

    rng.shuffle(train_sequences)
    rng.shuffle(test_sequences)
    return PreparedSplit(
        train_sequences=train_sequences,
        test_sequences=test_sequences,
    )


def write_sequences(output_file: str, sequences: list[str]) -> None:
    with open(output_file, "w") as handle:
        for sequence in sequences:
            handle.write(sequence + "\n")
