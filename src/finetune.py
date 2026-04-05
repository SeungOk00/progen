import sys
import os
import argparse
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from data_utils import ProteinDataset, build_datasets, load_tokenizer
from models.progen.modeling_progen import ProGenForCausalLM
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def init_new_embeddings(model: ProGenForCausalLM, prefixes: list[str]):
    if len(prefixes) <= 2:
        logger.info("No new embeddings to initialize.")
        return
    num_new_tokens = len(prefixes) - 2
    new_embs = torch.zeros((num_new_tokens, model.config.embed_dim), device=model.device)
    new_lm_head = torch.zeros((num_new_tokens, model.config.embed_dim), device=model.device)

    unk_token_emb: torch.Tensor = model.transformer.wte.weight[-1].detach()
    unk_token_lm_head: torch.Tensor = model.lm_head.weight[-1].detach()
    mean_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.mean()
    std_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.std()
    mean_unk_lm_head = torch.zeros_like(new_lm_head) + unk_token_lm_head.mean()
    std_unk_lm_head = torch.zeros_like(new_lm_head) + unk_token_lm_head.std()

    # initialize new embeddings with normal distribution same as untrained embeddings
    torch.normal(mean_unk_emb, std_unk_emb, out=new_embs)
    torch.normal(mean_unk_lm_head, std_unk_lm_head, out=new_lm_head)
    new_embs = torch.cat([model.transformer.wte.weight, new_embs], dim=0)
    new_lm_head = torch.cat([model.lm_head.weight, new_lm_head], dim=0)
    logger.debug(f"New embeddings shape: {new_embs.shape}")
    model.transformer.wte.weight = torch.nn.Parameter(new_embs, requires_grad=True)
    model.lm_head.weight = torch.nn.Parameter(new_lm_head, requires_grad=True)
    model.config.vocab_size_emb = new_embs.shape[0]
    model.config.vocab_size_lm_head = new_lm_head.shape[0]


def get_lr_schedule(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace, train_steps: int
):
    if args.decay == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=train_steps,
        )
    elif args.decay == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=train_steps,
        )
    elif args.decay == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9, last_epoch=-1
        )
    elif args.decay == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
        )
    else:
        raise ValueError(
            f"Invalid learning rate decay type. Must be 'cosine', 'linear', 'exponential', or 'constant'. Got: {args.decay}"
        )
    return scheduler


def train_epoch(
    model: ProGenForCausalLM,
    dataset: ProteinDataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    args: argparse.Namespace,
):
    model.train()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_loss = 0
    total_updates = max(1, (len(dataloader) + args.accumulation_steps - 1) // args.accumulation_steps)
    pbar = tqdm(total=total_updates)
    batch: dict[str, torch.Tensor]
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss: torch.Tensor = model(**batch).loss
        loss = loss / args.accumulation_steps
        loss.backward()
        total_loss = total_loss + loss.item()
        # using gradient accumulation to save memory
        if (i + 1) % args.accumulation_steps == 0 or i + 1 == len(dataloader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pbar.update()
    pbar.close()
    logger.info(f"TRAIN epoch {epoch}: loss: {total_loss / len(dataloader)}")
    logger.debug(f"Last learning rate: {scheduler.get_last_lr()}")
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: ProGenForCausalLM,
    dataset: ProteinDataset,
    args: argparse.Namespace,
    before_train: bool = False,
):
    model.eval()
    total_loss = 0
    eval_batch_size = 1 if before_train else args.batch_size * 4
    dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=True)
    total_length = len(dataloader)
    pbar = tqdm(total=total_length)
    batch: dict[str, torch.Tensor]
    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss: torch.Tensor = model(**batch).loss
        total_loss += loss.item()
        pbar.update()
    pbar.close()
    logger.info(f"EVAL loss: {total_loss / total_length}")
    return total_loss / total_length


def train(
    model: ProGenForCausalLM,
    tokenizer: Tokenizer,
    train_dataset: ProteinDataset,
    val_dataset: ProteinDataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
    job_id: str,
):
    train_losses = []
    eval_losses = []
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Start time of epoch {epoch}: {datetime.now()}")
        train_loss = train_epoch(model, train_dataset, optimizer, scheduler, epoch, args)
        train_losses.append(train_loss)

        logger.info(f"Running validation after {epoch} epochs:")
        eval_loss = evaluate(model, val_dataset, args)
        eval_losses.append(eval_loss)

        model_name = (job_id + "-" if job_id is not None else "") + args.model.strip(os.sep).split(os.sep)[-1]
        if epoch % args.checkpoint_rate == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join("checkpoints", f"{model_name}-finetuned", f"e{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            model.save_pretrained(checkpoint_path)
            tokenizer.save(os.path.join(checkpoint_path, "tokenizer.json"), pretty=True)

            if args.save_optimizer:
                logger.info("Saving optimizer and scheduler...")
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

            logger.info(f"Model saved at: {checkpoint_path}")
    return model, train_losses, eval_losses


def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        logger.debug(f"Slurm job id: {job_id}")
    else:
        logger.warning("No Slurm job ID found.")

    # loading data and tokenizer
    tokenizer = load_tokenizer(args.model)
    val_file = args.val_file or args.test_file
    if val_file is None:
        raise ValueError("Provide --val_file. --test_file is kept only as a compatibility alias.")

    train_data, val_data, prefixes, num_added_tokens = build_datasets(
        train_file=args.train_file,
        val_file=val_file,
        tokenizer=tokenizer,
    )
    logger.info(f"Found prefixes: {prefixes}")
    logger.info(f"Added {num_added_tokens} family tokens to tokenizer.")
    logger.debug(f"Train data size: {len(train_data)}")
    logger.debug(f"Validation data size: {len(val_data)}")

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU. Please consider using a GPU for training.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    logger.debug(f"hyperparameters: effective batch={args.batch_size * args.accumulation_steps}, {args.batch_size=}, {args.accumulation_steps=}, {args.epochs=}, {args.lr=}, {args.warmup_steps=}, {args.checkpoint_rate=}")

    # loading model
    logger.info(f"Loading model: {args.model}...")
    model = ProGenForCausalLM.from_pretrained(args.model).to(device)
    logger.info(f"Model loaded. Parameter count: {model.num_parameters() // 1e6} M")
    init_new_embeddings(model, prefixes)

    # creating optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_steps = max(
        1,
        args.epochs * len(train_data) // (args.batch_size * args.accumulation_steps),
    )
    logger.debug(f"Weight updates per epoch: {training_steps / args.epochs}")
    logger.debug(f"Total weight updates: {training_steps}")
    scheduler = get_lr_schedule(optimizer, args, training_steps)

    if args.eval_before_train:
        logger.info("Running validation before training...")
        evaluate(model, val_data, args, before_train=True)

    # training loop
    model, train_losses, val_losses = train(
        model,
        tokenizer,
        train_data,
        val_data,
        optimizer,
        scheduler,
        args,
        job_id,
    )

    logger.info("Finetuning finished.")
    logger.info(f"Train losses: {train_losses}")
    logger.info(f"Validation losses: {val_losses}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="hugohrban/progen2-small",
        help="Name of the model checkpoint to be finetuned.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on. Default: \"cuda\"",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to training data file. Must contain preprocessed data (includes prefixes and one protein per line, e.g. not fasta format).",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        help="Path to validation data file. Must contain preprocessed data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Compatibility alias for --val_file. Repeated evaluation during training uses validation data, not a final test set.",
    )
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="How many steps to accumulate gradients before updating weights. Default: 4",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate. Check out also the '--decay' argument. Default: 1e-4",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=200,
        help="Number of warmup steps for learning rate scheduler. Linearly increasing form 0 to --lr. Default: 200",
    )
    parser.add_argument(
        "--checkpoint_rate", type=int, default=5, help="Save model checkpoint every n epochs. Default: 5"
    )
    parser.add_argument(
        "--decay",
        type=str,
        choices=["cosine", "linear", "exponential", "constant"],
        default="cosine",
        help="Learning rate decay. Default: \"cosine\"",
    )
    parser.add_argument(
        "--save_optimizer",
        action="store_true",
        default=False,
        help="Should we also save the optimizer and scheduler at every checkpoint",
    )
    parser.add_argument(
        "--eval_before_train",
        action="store_true",
        default=False,
        help="Run validation before training. default: False",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging level.",
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    main(args)
