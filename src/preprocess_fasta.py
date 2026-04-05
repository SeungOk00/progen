"""
preprocess_fasta.py

FASTA 전처리 파이프라인:
  1. 정확한 중복 서열 제거
  2. cd-hit 클러스터링 (설치된 경우)
  3. 패밀리 간 데이터 균형 조정
  4. 헤더 버전별(v1-v4) 학습용 텍스트 파일 생성

사용 예시:
  python preprocess_fasta.py \
    --input_files downloads/PF02763.fasta downloads/PF09009.fasta downloads/PF03494.fasta \
    --main_families PF02763 PF09009 \
    --n_per_family 500 \
    --output_dir data \
    --cdhit_threshold 0.9 \
    --bidirectional

v3/v4 헤더를 사용하려면 먼저 fetch_annotations.py 를 실행하세요:
  python fetch_annotations.py --input_files downloads/*.fasta --output annotations.tsv
  python preprocess_fasta.py ... --annotations annotations.tsv
"""

import argparse
import logging
import os
import re
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 헤더 파싱
# ─────────────────────────────────────────

def parse_header(record: SeqRecord) -> tuple[str, str, str]:
    """
    download_pfam.py 출력 형식 파싱.
    헤더: ACCESSION|PF00000(start...end)|PROTEIN_NAME
    반환: (accession, pfam_id, protein_name)
    """
    desc = record.description
    parts = desc.split("|")
    accession = parts[0].strip() if len(parts) > 0 else record.id

    pfam_id = ""
    if len(parts) > 1:
        # 첫 번째 Pfam 항목만 추출, 위치 정보 제거
        m = re.search(r"(PF\d+)", parts[1])
        pfam_id = m.group(1) if m else parts[1].split("(")[0].strip()

    name = parts[2].strip() if len(parts) > 2 else ""
    return accession, pfam_id, name


def sanitize_label(s: str) -> str:
    """소문자 변환 후 특수문자를 언더스코어로 치환."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def extract_domain_region(record: SeqRecord) -> SeqRecord:
    """
    헤더의 위치 정보(start...end)를 파싱해 catalytic domain 구간만 추출.

    헤더 형식: ACCESSION|PF00000(start...end)|PROTEIN_NAME
    위치가 여러 fragment인 경우 모든 구간을 이어붙임.
    위치 정보가 없으면 전체 서열 반환.

    예시:
      PF02763(10...350)         → seq[9:350]
      PF02763(10...100,150...300) → seq[9:100] + seq[149:300]  (멀티 fragment)
    """
    desc = record.description
    parts = desc.split("|")
    if len(parts) < 2:
        return record  # 위치 정보 없음 → 전체 반환

    domain_part = parts[1]  # e.g., "PF02763(10...350)" or "PF02763(10...100,150...300)"

    # 괄호 안 위치 정보 추출
    loc_match = re.search(r"\(([^)]+)\)", domain_part)
    if not loc_match:
        return record

    loc_str = loc_match.group(1)  # e.g., "10...350" or "10...100,150...300"
    fragments = loc_str.split(",")

    full_seq = str(record.seq)
    domain_seq = ""
    for frag in fragments:
        coords = re.findall(r"(\d+)\.\.\.(\d+)", frag)
        for start_str, end_str in coords:
            start = int(start_str) - 1  # 1-based → 0-based
            end = int(end_str)
            domain_seq += full_seq[start:end]

    if not domain_seq:
        return record  # 파싱 실패 시 전체 반환

    new_record = record[:]
    new_record.seq = type(record.seq)(domain_seq)
    return new_record


# ─────────────────────────────────────────
# 중복 제거
# ─────────────────────────────────────────

def remove_exact_duplicates(records: list[SeqRecord]) -> tuple[list[SeqRecord], int]:
    """동일한 아미노산 서열을 가진 레코드 제거 (첫 번째 발생 유지)."""
    seen: set[str] = set()
    unique: list[SeqRecord] = []
    dup_count = 0
    for rec in records:
        seq_str = str(rec.seq).upper().replace("-", "")
        if seq_str not in seen:
            seen.add(seq_str)
            unique.append(rec)
        else:
            dup_count += 1
    return unique, dup_count


# ─────────────────────────────────────────
# cd-hit 클러스터링
# ─────────────────────────────────────────

def run_cdhit(
    records: list[SeqRecord],
    output_prefix: str,
    threshold: float = 0.9,
    word_size: int = 5,
    threads: int = 0,
) -> tuple[list[SeqRecord] | None, str | None]:
    """
    cd-hit 실행. 대표 서열 목록과 클러스터 파일 경로 반환.
    cd-hit 미설치 시 None, None 반환.
    word_size 가이드: threshold >= 0.7 → 5, 0.6~0.7 → 4, 0.5~0.6 → 3
    """
    tmp_fasta = output_prefix + "_input.fasta"
    tmp_out = output_prefix + "_cdhit"
    clstr_file = tmp_out + ".clstr"

    with open(tmp_fasta, "w") as f:
        SeqIO.write(records, f, "fasta")

    cmd = [
        "cd-hit",
        "-i", tmp_fasta,
        "-o", tmp_out,
        "-c", str(threshold),
        "-n", str(word_size),
        "-d", "0",
        "-M", "16000",
        "-T", str(threads),
    ]

    try:
        logger.info(f"cd-hit 실행: threshold={threshold}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"cd-hit 오류:\n{result.stderr}")

        reps = list(SeqIO.parse(tmp_out, "fasta"))
        logger.info(f"cd-hit 완료: {len(records)} → {len(reps)} 대표 서열")
        return reps, clstr_file

    except FileNotFoundError:
        logger.warning("cd-hit 가 설치되어 있지 않습니다. 클러스터링을 건너뜁니다.")
        logger.warning("설치 방법: conda install -c bioconda cd-hit  또는  sudo apt install cd-hit")
        if os.path.exists(tmp_fasta):
            os.remove(tmp_fasta)
        return None, None

    except RuntimeError as e:
        logger.error(str(e))
        if os.path.exists(tmp_fasta):
            os.remove(tmp_fasta)
        return None, None


# ─────────────────────────────────────────
# 데이터 균형 조정
# ─────────────────────────────────────────

def balance_families(
    family_records: dict[str, list[SeqRecord]],
    main_families: list[str],
    n_per_family: int,
    rng: np.random.Generator,
) -> dict[str, list[SeqRecord]]:
    """
    메인 패밀리는 최대 n_per_family 개씩 샘플링.
    보조 패밀리(나머지)는 메인 패밀리 총 개수만큼 균등 분배.
    """
    balanced: dict[str, list[SeqRecord]] = {}

    use_all = (n_per_family == 0)

    # 메인 패밀리 처리
    total_main = 0
    for fam in main_families:
        recs = list(family_records.get(fam, []))
        rng.shuffle(recs)
        sampled = recs if use_all else recs[:n_per_family]
        balanced[fam] = sampled
        total_main += len(sampled)
        logger.info(f"{fam}: {len(family_records.get(fam, []))} → {len(sampled)} 서열")

    # 보조 패밀리 처리
    aux_families = [f for f in family_records if f not in main_families]
    if aux_families:
        for fam in aux_families:
            recs = list(family_records.get(fam, []))
            rng.shuffle(recs)
            if use_all:
                sampled = recs
            else:
                quota_per_aux = max(1, total_main // len(aux_families))
                sampled = recs[:quota_per_aux]
            balanced[fam] = sampled
            logger.info(f"{fam} (보조): {len(family_records.get(fam, []))} → {len(sampled)} 서열")

    return balanced


# ─────────────────────────────────────────
# 어노테이션 로드
# ─────────────────────────────────────────

def load_annotations(annotation_file: str) -> dict[str, dict[str, str]]:
    """
    fetch_annotations.py 출력 TSV 로드.
    형식: accession\tmolecular_function\tec_number
    반환: {accession: {'molecular_function': ..., 'ec_number': ...}}
    """
    annotations: dict[str, dict[str, str]] = {}
    with open(annotation_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                acc = parts[0].strip()
                annotations[acc] = {
                    "molecular_function": parts[1].strip(),
                    "ec_number": parts[2].strip(),
                }
    return annotations


# ─────────────────────────────────────────
# 학습용 텍스트 생성
# ─────────────────────────────────────────

def make_training_lines(
    record: SeqRecord,
    label: str,
    bidirectional: bool = False,
) -> list[str]:
    seq = str(record.seq).upper().replace("-", "")
    lines = [f"<|{label}|>1{seq}2"]
    if bidirectional:
        lines.append(f"<|{label}|>2{seq[::-1]}1")
    return lines


def records_to_training_lines(
    records: list[SeqRecord],
    version: str,
    filename_label: str,
    annotations: dict[str, dict[str, str]] | None = None,
    bidirectional: bool = False,
) -> list[str]:
    """
    버전별로 레코드를 학습 텍스트 형식으로 변환.

    v1: <|accession_pfamid_proteinname|>
    v2: <|proteinname|>
    v3: <|molecular_function|>  ← annotations 필요
    v4: <|ec_number|>           ← annotations 필요
    """
    lines = []
    missing_annotation = 0

    for rec in records:
        accession, pfam_id, name = parse_header(rec)

        if version == "v1":
            ann = (annotations or {}).get(accession, {})
            ec = ann.get("ec_number", "")
            parts = [p for p in [accession, pfam_id, name, ec] if p]
            label = sanitize_label("_".join(parts)) if parts else filename_label

        elif version == "v2":
            label = sanitize_label(name) if name else filename_label

        elif version == "v3":
            ann = (annotations or {}).get(accession, {})
            mf = ann.get("molecular_function", "")
            if mf:
                label = sanitize_label(mf)
            else:
                missing_annotation += 1
                label = sanitize_label(name) if name else filename_label

        elif version == "v4":
            ann = (annotations or {}).get(accession, {})
            ec = ann.get("ec_number", "")
            if ec:
                label = sanitize_label(ec)
            else:
                missing_annotation += 1
                label = sanitize_label(name) if name else filename_label

        else:
            label = filename_label

        lines.extend(make_training_lines(rec, label, bidirectional))

    if missing_annotation > 0:
        logger.warning(
            f"버전 {version}: {missing_annotation}개 레코드에 어노테이션 없음 → 단백질 이름으로 대체"
        )
    return lines


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main(args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    clean_dir = os.path.join(args.output_dir, "cleaned")
    os.makedirs(clean_dir, exist_ok=True)

    # 어노테이션 로드 (v3/v4용)
    annotations = None
    if args.annotations:
        annotations = load_annotations(args.annotations)
        logger.info(f"어노테이션 로드: {len(annotations)}개 레코드")

    # 패밀리별 레코드 로드
    family_records: dict[str, list[SeqRecord]] = {}
    family_labels: dict[str, str] = {}

    for fasta_path in args.input_files:
        base = os.path.basename(fasta_path)
        label = os.path.splitext(base)[0].lower()  # e.g., "pf02763"
        fam_id = label.upper()                       # e.g., "PF02763"

        with open(fasta_path) as f:
            records = list(SeqIO.parse(f, "fasta"))
        logger.info(f"로드: {fasta_path} → {len(records)}개 서열")

        # 0. catalytic domain 구간 추출 (옵션)
        if args.extract_domain:
            records = [extract_domain_region(r) for r in records]
            # 너무 짧은 서열 제거 (도메인 파싱 실패 또는 비정상 서열)
            before = len(records)
            records = [r for r in records if len(r.seq) >= args.min_domain_len]
            dropped = before - len(records)
            logger.info(f"{fam_id}: 도메인 추출 완료 (최소 길이 {args.min_domain_len}aa 미만 {dropped}개 제거)")

        # 1. 중복 제거
        unique, n_dup = remove_exact_duplicates(records)
        logger.info(f"{fam_id}: 중복 {n_dup}개 제거 → {len(unique)}개 유니크 서열")

        family_records[fam_id] = unique
        family_labels[fam_id] = label

    # 2. cd-hit 클러스터링
    if not args.skip_cdhit:
        for fam_id in list(family_records.keys()):
            prefix = os.path.join(clean_dir, family_labels[fam_id])
            reps, clstr_file = run_cdhit(
                family_records[fam_id],
                output_prefix=prefix,
                threshold=args.cdhit_threshold,
                word_size=args.cdhit_word_size,
            )
            if reps is not None:
                family_records[fam_id] = reps
                logger.info(f"{fam_id} cd-hit 클러스터 파일: {clstr_file}")
    else:
        logger.info("cd-hit 클러스터링 건너뜀 (--skip_cdhit)")

    # 3. 데이터 균형 조정
    main_families = [f.upper() for f in args.main_families]
    balanced = balance_families(family_records, main_families, args.n_per_family, rng)

    # 정제된 FASTA 저장
    for fam_id, recs in balanced.items():
        out_fasta = os.path.join(clean_dir, f"{fam_id.lower()}_clean.fasta")
        with open(out_fasta, "w") as f:
            SeqIO.write(recs, f, "fasta")
        logger.info(f"정제된 FASTA 저장: {out_fasta} ({len(recs)}개 서열)")

    # 4. 버전별 학습 데이터 생성
    versions = args.versions
    needs_annotations = {"v1", "v3", "v4"}
    if args.annotations is None and any(v in needs_annotations for v in versions):
        missing = [v for v in versions if v in needs_annotations]
        logger.warning(
            f"{missing} 버전은 --annotations 파일이 있어야 EC 번호/분자기능이 포함됩니다. "
            "fetch_annotations.py 를 먼저 실행하거나, "
            "없으면 단백질 이름으로 대체됩니다."
        )

    for version in versions:
        version_dir = os.path.join(args.output_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        all_lines: list[str] = []
        for fam_id, recs in balanced.items():
            label = family_labels.get(fam_id, fam_id.lower())
            lines = records_to_training_lines(
                recs, version, label, annotations, args.bidirectional
            )
            all_lines.extend(lines)

        # 셔플 후 train/val 분리
        rng.shuffle(all_lines)
        split_idx = int(len(all_lines) * args.train_split_ratio)
        train_lines = all_lines[:split_idx]
        val_lines = all_lines[split_idx:]

        train_path = os.path.join(version_dir, "train.txt")
        val_path = os.path.join(version_dir, "val.txt")

        with open(train_path, "w") as f:
            f.write("\n".join(train_lines) + "\n")
        with open(val_path, "w") as f:
            f.write("\n".join(val_lines) + "\n")

        logger.info(
            f"[{version}] train: {len(train_lines)}개, val: {len(val_lines)}개 "
            f"→ {version_dir}"
        )

    logger.info("전처리 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FASTA 전처리: 중복 제거, cd-hit 클러스터링, 데이터 균형, 버전별 학습 데이터 생성"
    )
    parser.add_argument(
        "--input_files", nargs="+", required=True,
        help="입력 FASTA 파일 목록 (예: downloads/PF02763.fasta ...)"
    )
    parser.add_argument(
        "--main_families", nargs="+", required=True,
        help="메인 패밀리 ID 목록 (예: PF02763 PF09009). 나머지는 보조로 처리."
    )
    parser.add_argument(
        "--output_dir", default="data",
        help="출력 디렉토리. 기본값: data"
    )
    parser.add_argument(
        "--n_per_family", type=int, default=0,
        help="메인 패밀리당 최대 서열 수. 0이면 전체 사용. 기본값: 0 (전체)"
    )
    parser.add_argument(
        "--cdhit_threshold", type=float, default=0.9,
        help="cd-hit 시퀀스 동일성 임계값 (0~1). 기본값: 0.9"
    )
    parser.add_argument(
        "--cdhit_word_size", type=int, default=5,
        help="cd-hit word size. threshold>=0.7→5, 0.6~0.7→4, 0.5~0.6→3. 기본값: 5"
    )
    parser.add_argument(
        "--skip_cdhit", action="store_true",
        help="cd-hit 클러스터링 건너뜀"
    )
    parser.add_argument(
        "--versions", nargs="+", default=["v1", "v2"],
        choices=["v1", "v2", "v3", "v4"],
        help="생성할 헤더 버전 목록. v3/v4는 --annotations 필요. 기본값: v1 v2"
    )
    parser.add_argument(
        "--annotations",
        help="fetch_annotations.py 출력 TSV 파일 경로 (v3/v4 헤더용)"
    )
    parser.add_argument(
        "--bidirectional", "-b", action="store_true",
        help="역방향 서열도 학습 데이터에 포함"
    )
    parser.add_argument(
        "--train_split_ratio", type=float, default=0.8,
        help="학습/검증 분리 비율. 기본값: 0.8"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드. 기본값: 42"
    )
    parser.add_argument(
        "--extract_domain", action="store_true",
        help="헤더의 위치 정보(start...end)를 이용해 catalytic domain 구간만 추출. 기본값: False (전체 서열 사용)"
    )
    parser.add_argument(
        "--min_domain_len", type=int, default=10,
        help="도메인 추출 후 최소 서열 길이 (aa). 이보다 짧으면 제거. 기본값: 10"
    )
    args = parser.parse_args()
    main(args)
