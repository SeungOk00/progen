# 학습 방법

## 전체 파이프라인

```
data/v1/train.txt, val.txt
data/v2/train.txt, val.txt
data/v3/train.txt, val.txt
data/v4/train.txt, val.txt
      ↓ lora_finetune.py (버전별 각각 실행)
checkpoints/lora_v1/checkpoint_e{N}/
checkpoints/lora_v2/checkpoint_e{N}/
checkpoints/lora_v3/checkpoint_e{N}/
checkpoints/lora_v4/checkpoint_e{N}/
      + loss_curve.png (매 에폭 갱신)
```

---

## 베이스 모델

- **모델**: `hugohrban/progen2-small` (151M 파라미터)
- **구조**: GPT-J 계열 Causal LM (단방향 자기회귀)
- **토크나이저**: BPE 기반, 최대 길이 1024
- **출처**: Salesforce ProGen2 (Nijkamp et al., 2022)

첫 실행 시 HuggingFace에서 자동 다운로드 (~600MB):
```powershell
huggingface-cli download hugohrban/progen2-small
```

---

## 학습 방법: LoRA (Low-Rank Adaptation)

Full fine-tuning 대신 LoRA를 사용해 VRAM을 절약하고 overfitting을 방지.

### LoRA 원리

```
기존 가중치 W (고정)
      ↓
W' = W + α/r × (A × B)
     A: (d × r), B: (r × d)  ← 학습되는 부분
```

- 전체 파라미터의 ~1% 만 학습
- Attention의 `qk_proj`, `v_proj` 레이어에만 적용

### LoRA 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--lora_r` | 16 | Rank. 클수록 표현력↑, VRAM↑ |
| `--lora_alpha` | 32 | Scaling factor (보통 r×2) |
| `--lora_dropout` | 0.05 | Dropout |
| `--lora_target_modules` | `qk_proj v_proj` | LoRA 적용 레이어 |

---

## 학습 데이터 포맷

```
<label>1{서열}2          ← N→C 방향
<label>2{역방향서열}1    ← C→N 방향 (bidirectional)
```

### 헤더 버전별 학습 실험

| 버전 | label 구성 | 예시 |
|------|-----------|------|
| v1 | accession + pfam_id + protein_name + EC번호 (서열마다 다름) | `q6nk15_pf02763_diphtheria_toxin_ec_2_4_2_36` |
| v2 | 단백질 이름 고정 | `diphtheria_toxin` |
| v3 | 기능 이름 고정 | `exotoxin_a_catalytic` |
| v4 | EC번호 고정 | `nad_diphthamide_adp_ribosyltransferase_activity` |

---

## 학습 설정

### 기본 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--epochs` | 10 | 총 학습 에폭 수 |
| `--batch_size` | 8 | 미니배치 크기 |
| `--accumulation_steps` | 4 | Gradient accumulation (유효 배치 = 8×4=32) |
| `--lr` | 1e-4 | 학습률 |
| `--warmup_steps` | 100 | LR warmup 스텝 수 |
| `--decay` | cosine | LR 스케줄러 (cosine / linear / constant) |
| `--checkpoint_rate` | 5 | N 에폭마다 체크포인트 저장 |

### LR 스케줄러 (cosine)

```
lr
↑
│  /‾‾‾‾‾‾\
│ /          \___________
└──────────────────────→ step
  warmup     cosine decay
```

---

## 실행 방법

### 버전별 학습 (각각 실행)

```powershell
# v1
python lora_finetune.py --train_file ../data/v1/train.txt --val_file ../data/v1/val.txt --output_dir ../checkpoints/lora_v1 --epochs 10

# v2
python lora_finetune.py --train_file ../data/v2/train.txt --val_file ../data/v2/val.txt --output_dir ../checkpoints/lora_v2 --epochs 10

# v3
python lora_finetune.py --train_file ../data/v3/train.txt --val_file ../data/v3/val.txt --output_dir ../checkpoints/lora_v3 --epochs 10

# v4
python lora_finetune.py --train_file ../data/v4/train.txt --val_file ../data/v4/val.txt --output_dir ../checkpoints/lora_v4 --epochs 10
```

### LoRA rank 조정 (VRAM에 따라)

```powershell
# VRAM 여유 있을 때 (표현력 높이기)
python lora_finetune.py ... --lora_r 32 --lora_alpha 64

# VRAM 부족할 때
python lora_finetune.py ... --lora_r 8 --lora_alpha 16
```

---

## 출력 결과

```
checkpoints/
├── lora_v1/
│   ├── checkpoint_e5/
│   │   ├── adapter_config.json   ← LoRA 설정
│   │   ├── adapter_model.bin     ← LoRA 가중치 (소용량)
│   │   └── tokenizer.json        ← 패밀리 토큰 포함
│   ├── checkpoint_e10/
│   └── loss_curve.png            ← 학습 곡선 (매 에폭 갱신)
├── lora_v2/
├── lora_v3/
└── lora_v4/
```

---

## Loss 그래프

매 에폭마다 `loss_curve.png` 자동 저장.
- 파란 실선: Train Loss
- 빨간 점선: Validation Loss
- 최소 Validation Loss 지점 표시

---

## 손실 함수

**Cross-Entropy Loss** (언어 모델 표준)

```
L = -1/N × Σ log P(token_t | token_1, ..., token_{t-1})
```

- 패딩 토큰은 loss 계산에서 제외 (label = -100)
- 서열 전체를 예측 대상으로 사용 (헤더 토큰 포함)

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| [src/lora_finetune.py](src/lora_finetune.py) | LoRA 학습 메인 스크립트 |
| [src/finetune.py](src/finetune.py) | 원본 Full fine-tuning 스크립트 |
| [src/data_utils.py](src/data_utils.py) | 데이터셋 로딩 및 토크나이저 |
| [src/hf_utils.py](src/hf_utils.py) | HuggingFace 모델 로드 |
| [src/models/progen/modeling_progen.py](src/models/progen/modeling_progen.py) | ProGen2 모델 아키텍처 |
