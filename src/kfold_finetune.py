import argparse
import json
import os

import numpy as np
import torch

from data_utils import (
    build_dataset_from_lines,
    load_sequence_lines,
    load_tokenizer,
    make_kfold_splits,
)
from finetune import evaluate, get_lr_schedule, init_new_embeddings, train
from hf_utils import configure_hf_auth, load_model

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_fold(
    fold_idx: int,
    train_lines: list[str],
    val_lines: list[str],
    prefixes: list[str],
    args: argparse.Namespace,
):
    tokenizer = load_tokenizer(args.model)
    train_dataset, _, _ = build_dataset_from_lines(train_lines, tokenizer, prefixes)
    val_dataset, _, _ = build_dataset_from_lines(val_lines, tokenizer, prefixes)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Fold {fold_idx}: loading model {args.model}")
    model = load_model(args.model, device=device)
    init_new_embeddings(model, prefixes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_steps = args.epochs * max(
        1, len(train_dataset) // (args.batch_size * args.accumulation_steps)
    )
    scheduler = get_lr_schedule(optimizer, args, training_steps)

    if args.eval_before_train:
        logger.info(f"Fold {fold_idx}: validation before training")
        evaluate(model, val_dataset, args, before_train=True)

    fold_job_id = f"fold{fold_idx}"
    _, train_losses, val_losses = train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        job_id=fold_job_id,
    )
    return {
        "fold": fold_idx,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": min(val_losses) if val_losses else None,
    }


def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env_path = configure_hf_auth()
    if env_path is not None:
        logger.info(f"Loaded Hugging Face credentials from {env_path}")

    lines, prefixes = load_sequence_lines(args.data_file)
    logger.info(f"Loaded {len(lines)} sequences from {args.data_file}")
    logger.info(f"Found prefixes: {prefixes}")

    folds = make_kfold_splits(lines, num_folds=args.num_folds, seed=args.seed)
    fold_results = []
    for fold_idx, (train_lines, val_lines) in enumerate(folds, start=1):
        logger.info(
            f"Starting fold {fold_idx}/{args.num_folds} with train={len(train_lines)}, val={len(val_lines)}"
        )
        fold_results.append(run_fold(fold_idx, train_lines, val_lines, prefixes, args))

    best_losses = [result["best_val_loss"] for result in fold_results if result["best_val_loss"] is not None]
    summary = {
        "model": args.model,
        "data_file": args.data_file,
        "num_folds": args.num_folds,
        "mean_best_val_loss": float(np.mean(best_losses)) if best_losses else None,
        "std_best_val_loss": float(np.std(best_losses)) if best_losses else None,
        "folds": fold_results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "kfold_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info(f"Saved k-fold summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hugohrban/progen2-small")
    parser.add_argument("--data_file", type=str, required=True, help="Preprocessed text file used for k-fold splitting.")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds. Default: 5")
    parser.add_argument("--output_dir", type=str, default="checkpoints/kfold", help="Directory for k-fold summary output.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--checkpoint_rate", type=int, default=5)
    parser.add_argument(
        "--decay",
        type=str,
        choices=["cosine", "linear", "exponential", "constant"],
        default="cosine",
    )
    parser.add_argument("--save_optimizer", action="store_true", default=False)
    parser.add_argument("--eval_before_train", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    main(args)
