# 데이터 전처리 로직

## 전체 파이프라인

```
downloads/*.fasta
      ↓ (1) fetch_annotations.py
annotations.tsv
      ↓ (2) preprocess_fasta.py
data/cleaned/*.fasta
data/v1/train.txt, val.txt
data/v2/train.txt, val.txt
data/v3/train.txt, val.txt
data/v4/train.txt, val.txt
      ↓ (3) analysis.py
figures/*.png
```

---

## 1단계: 어노테이션 수집 — `fetch_annotations.py`

v1(EC번호), v3(분자기능), v4(EC번호 단독) 헤더에 필요한 정보를 UniProt REST API에서 가져옴.

```bash
python fetch_annotations.py \
  --input_files downloads/PF02763.fasta downloads/PF09009.fasta downloads/PF03494.fasta \
  --output annotations.tsv
```

### 동작 방식

1. FASTA 헤더에서 UniProt accession 추출 (`Q6NK15|PF02763(...)| ...` → `Q6NK15`)
2. 각 accession에 대해 `https://rest.uniprot.org/uniprotkb/{accession}.json` 호출
3. 응답에서 두 가지 정보 추출:
   - **molecular_function**: GO term 중 `F:` 분류 → FUNCTION comment 순으로 추출
   - **ec_number**: `recommendedName.ecNumbers` 에서 추출 (예: `EC 2.4.2.36`)
4. rate limit 방지: 요청 간 0.2초 대기, 429 응답 시 지수 백오프

### 출력 형식 (TSV)

```
# accession	molecular_function	ec_number
Q6NK15	NAD+-diphthamide ADP-ribosyltransferase activity	EC 2.4.2.36
...
```

---

## 2단계: 전처리 — `preprocess_fasta.py`

```bash
python preprocess_fasta.py \
  --input_files downloads/PF02763.fasta downloads/PF09009.fasta downloads/PF03494.fasta \
  --main_families PF02763 PF09009 \
  --n_per_family 500 \
  --versions v1 v2 v3 v4 \
  --annotations annotations.tsv \
  --output_dir ../data \
  --bidirectional
```

### Step 1: 정확한 중복 제거

- 아미노산 서열을 대문자로 정규화하고 갭(`-`) 제거 후 비교
- 동일한 서열이 있으면 첫 번째 발생만 유지

### Step 2: cd-hit 클러스터링

- 지정한 동일성 임계값(`--cdhit_threshold`, 기본 0.9) 이상인 서열을 군집화
- 각 군집의 대표 서열만 남김 → 유사 서열 제거 효과
- cd-hit 미설치 시 이 단계 자동 건너뜀

| 임계값 | word_size | 의미 |
|--------|-----------|------|
| ≥ 0.7 | 5 | 70% 이상 유사한 서열 군집화 |
| 0.6~0.7 | 4 | |
| 0.5~0.6 | 3 | |

출력: `data/cleaned/pf02763_cdhit.clstr` (군집 정보 파일)

### Step 3: 데이터 균형 조정

- **메인 패밀리** (PF02763, PF09009): 최대 `n_per_family`(500)개 무작위 샘플링
- **보조 패밀리** (PF03494): 메인 전체 개수 ÷ 보조 패밀리 수 만큼 할당
  - 예) 메인 1000개, 보조 1개 → PF03494에서 1000개 샘플링

정제된 FASTA 저장: `data/cleaned/{family}_clean.fasta`

### Step 4: 버전별 학습 데이터 생성

FASTA 레코드를 학습 텍스트 포맷으로 변환 후 train(80%) / val(20%) 분리.

#### 학습 텍스트 포맷

```
<|레이블|>1{서열}2          ← N→C 방향
<|레이블|>2{역방향서열}1    ← C→N 방향 (--bidirectional 옵션)
```

| 기호 | 의미 |
|------|------|
| `<\|레이블\|>` | 패밀리/기능 정보 토큰 |
| `1` | N-terminus 토큰 |
| `2` | C-terminus 토큰 |

#### 헤더 버전별 레이블 구성

| 버전 | 레이블 구성 | 예시 |
|------|------------|------|
| v1 | accession + pfam_id + protein_name + EC번호 | `q6nk15_pf02763_diphtheria_toxin_ec_2_4_2_36` |
| v2 | protein_name만 | `diphtheria_toxin` |
| v3 | molecular_function (GO term) | `nad_diphthamide_adp_ribosyltransferase_activity` |
| v4 | EC번호만 | `ec_2_4_2_36` |

- v1/v3/v4는 `annotations.tsv` 필요
- 어노테이션 없는 레코드는 protein_name으로 대체

---

## 3단계: 분석 시각화 — `analysis.py`

```bash
python analysis.py \
  --input_files ../data/cleaned/pf02763_clean.fasta \
                ../data/cleaned/pf09009_clean.fasta \
                ../data/cleaned/pf03494_clean.fasta \
  --output_dir ../figures
```

### 생성 그래프

| 파일 | 내용 |
|------|------|
| `length_distribution.png` | 패밀리별 서열 길이 히스토그램 + 중앙값 표시 |
| `family_composition.png` | 패밀리별 서열 수 막대 그래프 |
| `cluster_distribution.png` | cd-hit 클러스터 크기 분포 (`--clstr_files` 제공 시) |
| `similarity_heatmap.png` | 패밀리 간 k-mer 유사도 히트맵 |

유사도 히트맵은 k-mer Jaccard 유사도 사용 (기본 k=3, 패밀리당 최대 50개 샘플).

---

## 출력 디렉토리 구조

```
data/
├── cleaned/
│   ├── pf02763_clean.fasta
│   ├── pf09009_clean.fasta
│   ├── pf03494_clean.fasta
│   ├── pf02763_cdhit.clstr     ← cd-hit 설치 시
│   └── ...
├── v1/
│   ├── train.txt
│   └── val.txt
├── v2/
│   ├── train.txt
│   └── val.txt
├── v3/
│   ├── train.txt
│   └── val.txt
└── v4/
    ├── train.txt
    └── val.txt
figures/
├── length_distribution.png
├── family_composition.png
├── cluster_distribution.png
└── similarity_heatmap.png
```

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| [src/download_pfam.py](src/download_pfam.py) | Pfam FASTA 다운로드 |
| [src/fetch_annotations.py](src/fetch_annotations.py) | UniProt 어노테이션 수집 |
| [src/preprocess_fasta.py](src/preprocess_fasta.py) | 전처리 메인 스크립트 |
| [src/raw_fasta_utils.py](src/raw_fasta_utils.py) | FASTA 파싱, 버전별 태깅 함수 |
| [src/analysis.py](src/analysis.py) | 분석 시각화 |
| [src/prepare_data.py](src/prepare_data.py) | 경량 전처리 (헤더 버전 선택만 필요할 때) |
