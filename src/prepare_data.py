import argparse
import logging
from raw_fasta_utils import prepare_dataset_splits, write_sequences

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    output_file_val = args.output_file_val or args.output_file_test
    if output_file_val is None:
        raise ValueError("Provide --output_file_val. --output_file_test is kept only as a compatibility alias.")

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
    args = parser.parse_args()
    if args.output_file_val is None and args.output_file_test is None:
        args.output_file_val = "validation_data.txt"
    main(args)
