import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


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


# ─────────────────────────────────────────
# 헤더 버전별 태깅 (v1-v4)
# ─────────────────────────────────────────

def _parse_pfam_header(record: SeqRecord) -> tuple[str, str, str]:
    """
    download_pfam.py 출력 헤더 파싱.
    형식: ACCESSION|PF00000(loc)|PROTEIN_NAME
    반환: (accession, pfam_id, protein_name)
    """
    parts = record.description.split("|")
    accession = parts[0].strip() if parts else record.id
    pfam_id = ""
    if len(parts) > 1:
        m = re.search(r"(PF\d+)", parts[1])
        pfam_id = m.group(1) if m else parts[1].split("(")[0].strip()
    name = parts[2].strip() if len(parts) > 2 else ""
    return accession, pfam_id, name


def _sanitize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def fasta_to_tagged_sequences_versioned(
    input_file_name: str,
    version: str = "v1",
    bidirectional: bool = False,
    annotations: Optional[dict[str, dict[str, str]]] = None,
) -> list[str]:
    """
    FASTA 파일을 버전별 태그 형식으로 변환.

    version:
      v1  <|accession_pfamid_proteinname|>  (기본 - 전체 헤더)
      v2  <|proteinname|>                   (단백질 이름만)
      v3  <|molecular_function|>            (GO 분자 기능, annotations 필요)
      v4  <|ec_number|>                     (EC 번호, annotations 필요)

    annotations: fetch_annotations.py 가 생성한 딕셔너리
      {accession: {'molecular_function': ..., 'ec_number': ...}}
    """
    base_name = os.path.basename(input_file_name).lower()
    filename_label = os.path.splitext(base_name)[0]  # fallback 레이블

    parsed_sequences: list[str] = []
    with open(input_file_name, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence = str(record.seq).upper().replace("-", "")
            accession, pfam_id, name = _parse_pfam_header(record)

            if version == "v1":
                ann = (annotations or {}).get(accession, {})
                ec = ann.get("ec_number", "")
                parts = [p for p in [accession, pfam_id, name, ec] if p]
                label = _sanitize("_".join(parts)) if parts else filename_label

            elif version == "v2":
                label = _sanitize(name) if name else filename_label

            elif version == "v3":
                ann = (annotations or {}).get(accession, {})
                mf = ann.get("molecular_function", "")
                label = _sanitize(mf) if mf else (_sanitize(name) if name else filename_label)

            elif version == "v4":
                ann = (annotations or {}).get(accession, {})
                ec = ann.get("ec_number", "")
                label = _sanitize(ec) if ec else (_sanitize(name) if name else filename_label)

            else:
                raise ValueError(f"알 수 없는 버전: {version}. v1/v2/v3/v4 중 선택하세요.")

            parsed_sequences.append(f"<|{label}|>1{sequence}2")
            if bidirectional:
                parsed_sequences.append(f"<|{label}|>2{sequence[::-1]}1")

    return parsed_sequences
