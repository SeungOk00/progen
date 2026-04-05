"""
analysis.py

전처리된 데이터 분석 및 시각화:
  1. 패밀리별 서열 길이 분포 (히스토그램)
  2. 패밀리 구성 비율 (막대 그래프)
  3. cd-hit 클러스터 크기 분포
  4. 패밀리별 서열 유사도 히트맵 (k-mer 기반, 패밀리마다 별도 파일)

사용 예시:
  python analysis.py \
    --input_files data/cleaned/pf02763_clean.fasta data/cleaned/pf09009_clean.fasta data/cleaned/pf03494_clean.fasta \
    --output_dir figures
"""

import argparse
import logging
import os
import re

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from Bio import SeqIO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 한글 폰트 설정
# ─────────────────────────────────────────

def setup_korean_font():
    """Windows/Mac/Linux 환경에서 한글 폰트 자동 설정."""
    candidates = [
        "Malgun Gothic",       # Windows
        "Apple SD Gothic Neo", # macOS
        "NanumGothic",         # Linux
        "DejaVu Sans",         # fallback
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


setup_korean_font()

# 남색(0) → 노랑(100) 컬러맵


# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────

def load_families(fasta_files: list[str]) -> dict[str, list[str]]:
    """FASTA 파일 목록을 {family_id: [sequences]} 딕셔너리로 로드."""
    families: dict[str, list[str]] = {}
    for path in fasta_files:
        base = os.path.basename(path)
        fam_id = os.path.splitext(base)[0].upper()
        fam_id = re.sub(r"_CLEAN$", "", fam_id)
        with open(path) as f:
            seqs = [str(rec.seq).upper().replace("-", "") for rec in SeqIO.parse(f, "fasta")]
        families[fam_id] = seqs
        logger.info(f"로드: {fam_id} → {len(seqs)}개 서열")
    return families


def parse_clstr(clstr_file: str) -> list[list[str]]:
    """cd-hit .clstr 파일 파싱. 반환: 각 클러스터의 accession 목록 리스트."""
    clusters: list[list[str]] = []
    current: list[str] = []
    with open(clstr_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                if current:
                    clusters.append(current)
                current = []
            else:
                m = re.search(r">(\S+)\.\.\.", line)
                if m:
                    current.append(m.group(1))
    if current:
        clusters.append(current)
    return clusters


# ─────────────────────────────────────────
# 시각화 함수
# ─────────────────────────────────────────

def plot_length_distribution(
    families: dict[str, list[str]],
    output_path: str,
    bins: int = 40,
):
    """패밀리별 서열 길이 분포 히스토그램."""
    palette = sns.color_palette("Set2", len(families))
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4), sharey=False)
    if len(families) == 1:
        axes = [axes]

    for ax, (fam_id, seqs), color in zip(axes, families.items(), palette):
        lengths = [len(s) for s in seqs]
        ax.hist(lengths, bins=bins, color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(fam_id, fontsize=12)
        ax.set_xlabel("서열 길이 (aa)", fontsize=10)
        ax.set_ylabel("서열 수", fontsize=10)
        median = int(np.median(lengths))
        ax.axvline(median, color="black", linestyle="--", linewidth=1.2,
                   label=f"중앙값: {median} aa")
        ax.legend(fontsize=9)

    fig.suptitle("패밀리별 서열 길이 분포", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"저장: {output_path}")


def plot_family_composition(
    families: dict[str, list[str]],
    output_path: str,
):
    """패밀리별 서열 수 막대 그래프."""
    fam_ids = list(families.keys())
    counts = [len(seqs) for seqs in families.values()]
    palette = sns.color_palette("Set2", len(fam_ids))

    fig, ax = plt.subplots(figsize=(max(6, len(fam_ids) * 2), 4))
    bars = ax.bar(fam_ids, counts, color=palette, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xlabel("Pfam 패밀리", fontsize=11)
    ax.set_ylabel("서열 수", fontsize=11)
    ax.set_title("패밀리별 서열 수", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"저장: {output_path}")


def plot_cluster_distribution(
    clstr_files: list[str],
    fasta_files: list[str],
    output_path: str,
):
    """cd-hit 클러스터 크기 분포."""
    palette = sns.color_palette("Set1", len(clstr_files))
    fig, axes = plt.subplots(1, len(clstr_files), figsize=(5 * len(clstr_files), 4))
    if len(clstr_files) == 1:
        axes = [axes]

    for ax, clstr_file, fasta_file, color in zip(axes, clstr_files, fasta_files, palette):
        fam_id = re.sub(r"_CLEAN$", "",
                        os.path.splitext(os.path.basename(fasta_file))[0].upper())
        clusters = parse_clstr(clstr_file)
        sizes = [len(c) for c in clusters]

        ax.hist(sizes, bins=max(10, len(sizes) // 5), color=color,
                edgecolor="white", linewidth=0.5)
        ax.set_title(fam_id, fontsize=12)
        ax.set_xlabel("클러스터 크기 (서열 수)", fontsize=10)
        ax.set_ylabel("클러스터 수", fontsize=10)
        ax.text(
            0.98, 0.97,
            f"총 클러스터: {len(clusters)}\n총 서열: {sum(sizes)}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    fig.suptitle("cd-hit 클러스터 크기 분포", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"저장: {output_path}")


# ─────────────────────────────────────────
# 서열 유사도 히트맵 (패밀리별 개별 파일)
# ─────────────────────────────────────────

def kmer_similarity(seq1: str, seq2: str, k: int = 3) -> float:
    """k-mer Jaccard 유사도."""
    if not seq1 or not seq2:
        return 0.0
    kmers1 = {seq1[i:i+k] for i in range(len(seq1) - k + 1)}
    kmers2 = {seq2[i:i+k] for i in range(len(seq2) - k + 1)}
    union = kmers1 | kmers2
    return len(kmers1 & kmers2) / len(union) if union else 0.0


def compute_similarity_matrix(seqs: list[str], k: int = 3) -> np.ndarray:
    """서열 간 k-mer 유사도 행렬 계산."""
    n = len(seqs)
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            sim = kmer_similarity(seqs[i], seqs[j], k)
            mat[i, j] = sim
            mat[j, i] = sim
    return mat


def plot_similarity_heatmap_per_family(
    families: dict[str, list[str]],
    output_dir: str,
    max_per_family: int = 50,
    kmer_k: int = 3,
):
    """
    패밀리별로 개별 히트맵 파일 생성.
    색상: 0(남색) → 1(노랑)
    파일명: similarity_heatmap_{family_id}.png
    """
    for fam_id, seqs in families.items():
        sampled = seqs[:max_per_family]
        n = len(sampled)
        logger.info(f"{fam_id} 히트맵 계산 중: {n}x{n} 행렬 (k={kmer_k})")

        mat = compute_similarity_matrix(sampled, k=kmer_k)

        fig_size = max(6, n // 4)
        fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))

        hm = sns.heatmap(
            mat,
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=1,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": f"k-mer 유사도 (k={kmer_k})"},
            square=True,
        )

        # 컬러바 레이블 폰트 크기
        hm.collections[0].colorbar.ax.set_ylabel(
            f"k-mer 유사도 (k={kmer_k})", fontsize=10
        )

        ax.set_title(f"{fam_id} 서열 유사도 히트맵\n(샘플 {n}개)", fontsize=13, pad=12)
        ax.set_xlabel("서열", fontsize=10)
        ax.set_ylabel("서열", fontsize=10)

        out_path = os.path.join(output_dir, f"similarity_heatmap_{fam_id}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"저장: {out_path}")


# ─────────────────────────────────────────
# 통계 요약 출력
# ─────────────────────────────────────────

def print_statistics(families: dict[str, list[str]]):
    print("\n" + "=" * 55)
    print(f"{'Family':12s} {'Sequences':>10s} {'Mean len':>10s} {'Median':>8s} {'Min':>6s} {'Max':>6s}")
    print("-" * 55)
    for fam_id, seqs in families.items():
        lengths = [len(s) for s in seqs]
        print(
            f"{fam_id:12s} {len(seqs):>10d} "
            f"{np.mean(lengths):>10.1f} {np.median(lengths):>8.1f} "
            f"{min(lengths):>6d} {max(lengths):>6d}"
        )
    print("=" * 55 + "\n")


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    families = load_families(args.input_files)

    print_statistics(families)

    # 1. 서열 길이 분포
    plot_length_distribution(
        families,
        output_path=os.path.join(args.output_dir, "length_distribution.png"),
        bins=args.bins,
    )

    # 2. 패밀리 구성
    plot_family_composition(
        families,
        output_path=os.path.join(args.output_dir, "family_composition.png"),
    )

    # 3. cd-hit 클러스터 분포
    if args.clstr_files:
        if len(args.clstr_files) != len(args.input_files):
            logger.warning("--clstr_files 수가 --input_files 수와 다릅니다. 클러스터 플롯 건너뜀.")
        else:
            plot_cluster_distribution(
                args.clstr_files,
                args.input_files,
                output_path=os.path.join(args.output_dir, "cluster_distribution.png"),
            )

    # 4. 패밀리별 유사도 히트맵 (개별 파일)
    plot_similarity_heatmap_per_family(
        families,
        output_dir=args.output_dir,
        max_per_family=args.max_per_family,
        kmer_k=args.kmer_k,
    )

    logger.info(f"모든 그래프 저장 완료: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단백질 패밀리 데이터 분석 및 시각화")
    parser.add_argument("--input_files", nargs="+", required=True, help="분석할 FASTA 파일 목록")
    parser.add_argument("--clstr_files", nargs="+", help="cd-hit .clstr 파일 목록")
    parser.add_argument("--output_dir", default="figures", help="그래프 저장 디렉토리. 기본값: figures")
    parser.add_argument("--bins", type=int, default=40, help="히스토그램 bin 수. 기본값: 40")
    parser.add_argument("--max_per_family", type=int, default=50, help="히트맵 패밀리당 최대 서열 수. 기본값: 50")
    parser.add_argument("--kmer_k", type=int, default=3, help="k-mer 크기. 기본값: 3")
    args = parser.parse_args()
    main(args)
