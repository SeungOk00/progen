"""
lora_finetune.py

LoRA를 이용한 ProGen2 파인튜닝 스크립트.
기존 finetune.py 대비 VRAM 사용량을 대폭 줄임.

사용 예시 (v1 헤더):
  python lora_finetune.py \
    --train_file ../data/v1/train.txt \
    --val_file ../data/v1/val.txt \
    --output_dir ../checkpoints/lora_v1 \
    --epochs 10 \
    --lora_r 16 \
    --lora_alpha 32
"""

import argparse
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

from data_utils import build_datasets, load_tokenizer
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
# LoRA 설정
# ─────────────────────────────────────────

def apply_lora(model, args) -> None:
    """ProGen2 모델에 LoRA 적용."""
    # ProGen2는 GPT-J 계열: query/value projection에 LoRA 적용
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────
# 임베딩 초기화 (새 패밀리 토큰)
# ─────────────────────────────────────────

def init_new_embeddings(model, prefixes: list[str]):
    """새로 추가된 패밀리 토큰 임베딩을 unk 토큰 분포로 초기화."""
    if len(prefixes) <= 2:
        return
    num_new = len(prefixes) - 2

    # LoRA 래핑된 모델에서 base model 접근
    base = model.base_model.model if hasattr(model, "base_model") else model
    unk_emb = base.transformer.wte.weight[-1].detach()
    mean, std = unk_emb.mean(), unk_emb.std()

    with torch.no_grad():
        new_embs = torch.normal(
            mean.expand(num_new, -1) if unk_emb.dim() > 0 else mean.item(),
            std.item(),
            size=(num_new, base.config.embed_dim),
        ).to(base.transformer.wte.weight.device)
        base.transformer.wte.weight = torch.nn.Parameter(
            torch.cat([base.transformer.wte.weight, new_embs], dim=0),
            requires_grad=True,
        )
        new_lm = torch.normal(mean.item(), std.item(),
                              size=(num_new, base.config.embed_dim)).to(base.lm_head.weight.device)
        base.lm_head.weight = torch.nn.Parameter(
            torch.cat([base.lm_head.weight, new_lm], dim=0),
            requires_grad=True,
        )
        base.config.vocab_size_emb = base.transformer.wte.weight.shape[0]
        base.config.vocab_size_lm_head = base.lm_head.weight.shape[0]

    logger.info(f"새 패밀리 토큰 {num_new}개 임베딩 초기화 완료.")


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
    avg_loss = total_loss / len(dataloader)
    logger.info(f"TRAIN epoch {epoch}: loss={avg_loss:.4f}")
    return avg_loss


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
    avg_loss = total_loss / len(dataloader)
    logger.info(f"EVAL loss={avg_loss:.4f}")
    return avg_loss


# ─────────────────────────────────────────
# Loss 그래프 저장
# ─────────────────────────────────────────

def save_loss_plot(train_losses: list[float], val_losses: list[float], output_path: str):
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", linewidth=2, label="Train Loss", color="#2196F3")
    ax.plot(epochs, val_losses, marker="s", linewidth=2, label="Validation Loss", color="#F44336", linestyle="--")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("LoRA 파인튜닝 학습 곡선", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    # 최소 val loss 표시
    min_val_epoch = val_losses.index(min(val_losses)) + 1
    min_val = min(val_losses)
    ax.annotate(
        f"최소 val loss\n{min_val:.4f} (epoch {min_val_epoch})",
        xy=(min_val_epoch, min_val),
        xytext=(min_val_epoch + 0.3, min_val + (max(val_losses) - min_val) * 0.15),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
        color="#F44336",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Loss 그래프 저장: {output_path}")


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 디바이스
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 없음 → CPU로 대체")
        args.device = "cpu"
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # 데이터 로드
    configure_hf_auth()
    tokenizer = load_tokenizer(args.model)
    train_data, val_data, prefixes, num_added = build_datasets(
        train_file=args.train_file,
        val_file=args.val_file,
        tokenizer=tokenizer,
    )
    logger.info(f"패밀리 토큰 {num_added}개 추가: {prefixes}")
    logger.info(f"Train: {len(train_data)}개, Val: {len(val_data)}개")

    # 모델 로드 + LoRA 적용
    logger.info(f"모델 로드: {args.model}")
    model = load_model(args.model, device=str(device))
    init_new_embeddings(model, prefixes)
    model = apply_lora(model, args)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = max(1, args.epochs * len(train_data) // (args.batch_size * args.accumulation_steps))

    if args.decay == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    elif args.decay == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps)

    # 학습 루프
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_data, optimizer, scheduler, epoch, args)
        val_loss = evaluate(model, val_data, args)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Loss 그래프 매 에폭 업데이트
        save_loss_plot(
            train_losses, val_losses,
            output_path=os.path.join(args.output_dir, "loss_curve.png"),
        )

        # 체크포인트 저장
        if epoch % args.checkpoint_rate == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_e{epoch}")
            os.makedirs(ckpt_path, exist_ok=True)
            model.save_pretrained(ckpt_path)
            tokenizer.save(os.path.join(ckpt_path, "tokenizer.json"), pretty=True)
            logger.info(f"체크포인트 저장: {ckpt_path}")

    logger.info("학습 완료.")
    logger.info(f"Train losses: {[f'{l:.4f}' for l in train_losses]}")
    logger.info(f"Val losses:   {[f'{l:.4f}' for l in val_losses]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 기반 ProGen2 파인튜닝")
    parser.add_argument("--model", default="hugohrban/progen2-small", help="베이스 모델 경로 또는 HF 이름")
    parser.add_argument("--train_file", required=True, help="학습 데이터 파일 경로")
    parser.add_argument("--val_file", required=True, help="검증 데이터 파일 경로")
    parser.add_argument("--output_dir", default="../checkpoints/lora", help="체크포인트 및 그래프 저장 디렉토리")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--decay", default="cosine", choices=["cosine", "linear", "constant"])
    parser.add_argument("--checkpoint_rate", type=int, default=5, help="N 에폭마다 체크포인트 저장")
    parser.add_argument("--seed", type=int, default=42)
    # LoRA 하이퍼파라미터
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank. 클수록 표현력↑, VRAM↑. 기본값: 16")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor. 보통 r*2. 기본값: 32")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout. 기본값: 0.05")
    parser.add_argument(
        "--lora_target_modules", nargs="+",
        default=["qk_proj", "v_proj"],
        help="LoRA 적용할 모듈 이름. 기본값: qk_proj v_proj"
    )
    args = parser.parse_args()
    main(args)
