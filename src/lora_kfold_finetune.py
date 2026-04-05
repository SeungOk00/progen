"""
lora_kfold_finetune.py

LoRA + K-Fold 교차검증 파인튜닝 스크립트.
각 fold마다 모델을 새로 로드하고 LoRA를 적용해 독립적으로 학습.
전체 결과는 kfold_summary.json 및 kfold_loss_curve.png 로 저장.

사용 예시:
  python lora_kfold_finetune.py \
    --data_file ../data/v1/train.txt \
    --num_folds 5 \
    --output_dir ../checkpoints/lora_kfold_v1 \
    --epochs 10
"""

import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from data_utils import build_dataset_from_lines, load_sequence_lines, load_tokenizer, make_kfold_splits
from hf_utils import configure_hf_auth, load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 한글 폰트
# ─────────────────────────────────────────

def setup_korean_font():
    candidates = ["Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


setup_korean_font()


# ─────────────────────────────────────────
# LoRA 적용
# ─────────────────────────────────────────

def apply_lora(model, args):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def init_new_embeddings(model, prefixes: list[str]):
    if len(prefixes) <= 2:
        return
    num_new = len(prefixes) - 2
    base = model.base_model.model if hasattr(model, "base_model") else model
    unk_emb = base.transformer.wte.weight[-1].detach()
    mean, std = unk_emb.mean().item(), unk_emb.std().item()

    with torch.no_grad():
        new_embs = torch.normal(mean, std, size=(num_new, base.config.embed_dim)).to(base.transformer.wte.weight.device)
        base.transformer.wte.weight = torch.nn.Parameter(
            torch.cat([base.transformer.wte.weight, new_embs], dim=0), requires_grad=True
        )
        new_lm = torch.normal(mean, std, size=(num_new, base.config.embed_dim)).to(base.lm_head.weight.device)
        base.lm_head.weight = torch.nn.Parameter(
            torch.cat([base.lm_head.weight, new_lm], dim=0), requires_grad=True
        )
        base.config.vocab_size_emb = base.transformer.wte.weight.shape[0]
        base.config.vocab_size_lm_head = base.lm_head.weight.shape[0]


# ─────────────────────────────────────────
# 학습 / 평가
# ─────────────────────────────────────────

def train_epoch(model, dataset, optimizer, scheduler, epoch, args):
    model.train()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_loss = 0.0
    total_updates = max(1, (len(dataloader) + args.accumulation_steps - 1) // args.accumulation_steps)
    pbar = tqdm(total=total_updates, desc=f"Epoch {epoch} train")
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        loss = model(**batch).loss / args.accumulation_steps
        loss.backward()
        total_loss += loss.item()

        if (i + 1) % args.accumulation_steps == 0 or i + 1 == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pbar.update()

    pbar.close()
    avg = total_loss / len(dataloader)
    logger.info(f"TRAIN epoch {epoch}: loss={avg:.4f}")
    return avg


@torch.no_grad()
def evaluate(model, dataset, args):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=args.batch_size * 2, shuffle=False)
    total_loss = 0.0
    pbar = tqdm(total=len(dataloader), desc="Validation")
    for batch in dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        total_loss += model(**batch).loss.item()
        pbar.update()
    pbar.close()
    avg = total_loss / len(dataloader)
    logger.info(f"EVAL loss={avg:.4f}")
    return avg


# ─────────────────────────────────────────
# 그래프
# ─────────────────────────────────────────

def save_fold_loss_plot(train_losses, val_losses, fold_idx, output_path):
    """단일 fold loss 곡선."""
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", linewidth=2, label="Train Loss", color="#2196F3")
    ax.plot(epochs, val_losses, marker="s", linewidth=2, label="Validation Loss", color="#F44336", linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Fold {fold_idx} 학습 곡선", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    min_val = min(val_losses)
    min_ep = val_losses.index(min_val) + 1
    ax.annotate(
        f"최소 val loss\n{min_val:.4f} (epoch {min_ep})",
        xy=(min_ep, min_val),
        xytext=(min_ep + 0.3, min_val + (max(val_losses) - min_val) * 0.2),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
        color="#F44336",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"저장: {output_path}")


def save_kfold_summary_plot(fold_results, output_path):
    """전체 fold 결과 비교 그래프."""
    fold_ids = [r["fold"] for r in fold_results]
    best_vals = [r["best_val_loss"] for r in fold_results]
    mean_val = np.mean(best_vals)
    std_val = np.std(best_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: fold별 best val loss 막대 그래프
    palette = plt.cm.Set2(np.linspace(0, 1, len(fold_ids)))
    bars = axes[0].bar([f"Fold {i}" for i in fold_ids], best_vals, color=palette, edgecolor="white")
    axes[0].axhline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"평균: {mean_val:.4f}")
    axes[0].fill_between(
        [-0.5, len(fold_ids) - 0.5],
        mean_val - std_val, mean_val + std_val,
        alpha=0.15, color="red", label=f"±std: {std_val:.4f}"
    )
    for bar, val in zip(bars, best_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Best Validation Loss", fontsize=11)
    axes[0].set_title("Fold별 최소 Validation Loss", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")

    # 오른쪽: fold별 val loss 곡선 오버레이
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_results)))
    for r, color in zip(fold_results, colors):
        epochs = list(range(1, len(r["val_losses"]) + 1))
        axes[1].plot(epochs, r["val_losses"], marker="o", linewidth=1.5,
                     label=f"Fold {r['fold']}", color=color, alpha=0.8)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Validation Loss", fontsize=11)
    axes[1].set_title("Fold별 Validation Loss 곡선", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"K-Fold 교차검증 결과 (K={len(fold_ids)})  평균 val loss: {mean_val:.4f} ± {std_val:.4f}",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"저장: {output_path}")


# ─────────────────────────────────────────
# fold 단위 학습
# ─────────────────────────────────────────

def run_fold(fold_idx, train_lines, val_lines, prefixes, args):
    fold_dir = os.path.join(args.output_dir, f"fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.model)
    train_dataset, _, _ = build_dataset_from_lines(train_lines, tokenizer, prefixes)
    val_dataset, _, _ = build_dataset_from_lines(val_lines, tokenizer, prefixes)

    device = torch.device(args.device)
    logger.info(f"[Fold {fold_idx}] 모델 로드: {args.model}")
    model = load_model(args.model, device=str(device))
    init_new_embeddings(model, prefixes)
    model = apply_lora(model, args)
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = max(1, args.epochs * len(train_dataset) // (args.batch_size * args.accumulation_steps))

    if args.decay == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    elif args.decay == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps)

    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dataset, optimizer, scheduler, epoch, args)
        val_loss = evaluate(model, val_dataset, args)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # fold별 loss 곡선 갱신
        save_fold_loss_plot(
            train_losses, val_losses, fold_idx,
            output_path=os.path.join(fold_dir, "loss_curve.png"),
        )

        # 체크포인트 저장
        if epoch % args.checkpoint_rate == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(fold_dir, f"checkpoint_e{epoch}")
            os.makedirs(ckpt_path, exist_ok=True)
            model.save_pretrained(ckpt_path)
            tokenizer.save(os.path.join(ckpt_path, "tokenizer.json"), pretty=True)
            logger.info(f"[Fold {fold_idx}] 체크포인트 저장: {ckpt_path}")

    return {
        "fold": fold_idx,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": min(val_losses),
    }


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 없음 → CPU로 대체")
        args.device = "cpu"

    configure_hf_auth()

    lines, prefixes = load_sequence_lines(args.data_file)
    logger.info(f"총 {len(lines)}개 서열, 레이블: {prefixes}")

    folds = make_kfold_splits(lines, num_folds=args.num_folds, seed=args.seed)
    fold_results = []

    for fold_idx, (train_lines, val_lines) in enumerate(folds, start=1):
        logger.info(f"===== Fold {fold_idx}/{args.num_folds} | train={len(train_lines)}, val={len(val_lines)} =====")
        result = run_fold(fold_idx, train_lines, val_lines, prefixes, args)
        fold_results.append(result)

    # 전체 요약
    best_losses = [r["best_val_loss"] for r in fold_results]
    summary = {
        "model": args.model,
        "data_file": args.data_file,
        "num_folds": args.num_folds,
        "mean_best_val_loss": float(np.mean(best_losses)),
        "std_best_val_loss": float(np.std(best_losses)),
        "folds": fold_results,
    }

    summary_path = os.path.join(args.output_dir, "kfold_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"요약 저장: {summary_path}")

    # 전체 비교 그래프
    save_kfold_summary_plot(fold_results, os.path.join(args.output_dir, "kfold_loss_curve.png"))

    logger.info(f"K-Fold 완료. 평균 val loss: {np.mean(best_losses):.4f} ± {np.std(best_losses):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA + K-Fold 교차검증 파인튜닝")
    parser.add_argument("--model", default="hugohrban/progen2-small")
    parser.add_argument("--data_file", required=True, help="전처리된 학습 파일 (k-fold 분리에 사용)")
    parser.add_argument("--num_folds", type=int, default=5, help="Fold 수. 기본값: 5")
    parser.add_argument("--output_dir", default="../checkpoints/lora_kfold", help="출력 디렉토리")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--decay", default="cosine", choices=["cosine", "linear", "constant"])
    parser.add_argument("--checkpoint_rate", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=["qk_proj", "v_proj"])
    args = parser.parse_args()
    main(args)
