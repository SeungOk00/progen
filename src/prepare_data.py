import argparse
import logging
from raw_fasta_utils import (
    prepare_dataset_splits,
    write_sequences,
    fasta_to_tagged_sequences_versioned,
    split_sequences,
    PreparedSplit,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_annotations(annotation_file: str) -> dict:
    """fetch_annotations.py 출력 TSV 로드."""
    annotations = {}
    with open(annotation_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                annotations[parts[0].strip()] = {
                    "molecular_function": parts[1].strip(),
                    "ec_number": parts[2].strip(),
                }
    return annotations


def main(args: argparse.Namespace):
    output_file_val = args.output_file_val or args.output_file_test
    if output_file_val is None:
        raise ValueError("Provide --output_file_val. --output_file_test is kept only as a compatibility alias.")

    # 헤더 버전 v1 이상 사용 시 버전별 태깅
    if args.header_version != "default":
        annotations = None
        if args.annotations:
            annotations = load_annotations(args.annotations)
            logging.info(f"어노테이션 로드: {len(annotations)}개 레코드")
        elif args.header_version in ("v3", "v4"):
            logging.warning(
                f"헤더 버전 {args.header_version}은 --annotations 파일이 필요합니다. "
                "fetch_annotations.py 를 먼저 실행하세요."
            )

        import numpy as np
        rng = np.random.default_rng(args.seed)
        train_sequences = []
        test_sequences = []

        for input_file in args.input_files:
            sequences = fasta_to_tagged_sequences_versioned(
                input_file,
                version=args.header_version,
                bidirectional=args.bidirectional,
                annotations=annotations,
            )
            split = split_sequences(sequences, train_split_ratio=args.train_split_ratio, rng=rng)
            train_sequences.extend(split.train_sequences)
            test_sequences.extend(split.test_sequences)
            logging.info(f"로드: {input_file} ({len(sequences)}개 서열, 버전={args.header_version})")

        rng.shuffle(train_sequences)
        rng.shuffle(test_sequences)
        splits = PreparedSplit(train_sequences=train_sequences, test_sequences=test_sequences)

    else:
        # 기존 방식 (파일명을 레이블로 사용)
        splits = prepare_dataset_splits(
            input_files=args.input_files,
            train_split_ratio=args.train_split_ratio,
            seed=args.seed,
            bidirectional=args.bidirectional,
        )

    if args.bidirectional:
        logging.info("Data is bidirectional. Each sequence will be stored in both directions.")

    for input_file in args.input_files:
        logging.info(f"Loaded raw FASTA from {input_file}")

    logging.info(f"Train data: {len(splits.train_sequences)} sequences")
    logging.info(f"Validation data: {len(splits.test_sequences)} sequences")

    logging.info(f"Saving training data to {args.output_file_train}")
    write_sequences(args.output_file_train, splits.train_sequences)

    logging.info(f"Saving validation data to {output_file_val}")
    write_sequences(output_file_val, splits.test_sequences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_files", type=str, nargs="+", required=True, help="Input fasta files."
    )
    parser.add_argument(
        "--output_file_train", type=str, default="train_data.txt", help="Output file for the train data split. Default: train_data.txt"
    )
    parser.add_argument(
        "--output_file_val", type=str, help="Output file for validation split. Default is validation_data.txt when not using the compatibility alias."
    )
    parser.add_argument(
        "--output_file_test", type=str, help="Compatibility alias for --output_file_val."
    )
    parser.add_argument(
        "--bidirectional",
        "-b",
        action="store_true",
        help="Whether to store also the reverse of the sequences. Default: False.",
    )
    parser.add_argument(
        "--train_split_ratio",
        "-s",
        type=float,
        default=0.8,
        help="Train-test split ratio. Default: 0.8",
    )
    parser.add_argument(
        "--seed", type=int, default=69, help="Random seed",
    )
    parser.add_argument(
        "--header_version",
        type=str,
        default="default",
        choices=["default", "v1", "v2", "v3", "v4"],
        help=(
            "헤더 버전 선택. "
            "default: 파일명을 레이블로 사용 (기존 방식). "
            "v1: accession_pfamid_proteinname. "
            "v2: 단백질 이름만. "
            "v3: 분자 기능 (--annotations 필요). "
            "v4: EC 번호 (--annotations 필요). "
            "기본값: default"
        ),
    )
    parser.add_argument(
        "--annotations",
        type=str,
        help="fetch_annotations.py 출력 TSV 경로 (v3/v4 헤더용)",
    )
    args = parser.parse_args()
    if args.output_file_val is None and args.output_file_test is None:
        args.output_file_val = "validation_data.txt"
    main(args)
