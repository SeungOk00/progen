"""
fetch_annotations.py

UniProt REST API를 통해 단백질 어노테이션 수집.
preprocess_fasta.py 의 v3(molecular function), v4(EC number) 헤더 생성에 사용.

출력: TSV 파일 (accession, molecular_function, ec_number)

사용 예시:
  python fetch_annotations.py \
    --input_files downloads/PF02763.fasta downloads/PF09009.fasta downloads/PF03494.fasta \
    --output annotations.tsv
"""

import argparse
import json
import logging
import time
from urllib import request
from urllib.error import HTTPError
from urllib.parse import urlencode

from Bio import SeqIO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{accession}.json"


def fetch_uniprot(accession: str, retries: int = 3, delay: float = 1.0) -> dict | None:
    """UniProt에서 단백질 정보 JSON 가져오기."""
    url = UNIPROT_API.format(accession=accession)
    for attempt in range(retries):
        try:
            req = request.Request(url, headers={"Accept": "application/json"})
            with request.urlopen(req, timeout=10) as res:
                return json.loads(res.read().decode())
        except HTTPError as e:
            if e.code == 404:
                return None  # 존재하지 않는 accession
            if e.code == 429:  # rate limit
                wait = delay * (2 ** attempt)
                logger.warning(f"Rate limit, {wait:.1f}초 대기...")
                time.sleep(wait)
            else:
                logger.warning(f"{accession} 요청 실패 (HTTP {e.code}), 재시도 {attempt+1}/{retries}")
                time.sleep(delay)
        except Exception as e:
            logger.warning(f"{accession} 요청 오류: {e}, 재시도 {attempt+1}/{retries}")
            time.sleep(delay)
    return None


def extract_ec_numbers(data: dict) -> str:
    """UniProt JSON에서 EC 번호 추출."""
    ec_numbers = []
    try:
        protein_desc = data.get("proteinDescription", {})
        recommended = protein_desc.get("recommendedName", {})
        for ec_entry in recommended.get("ecNumbers", []):
            val = ec_entry.get("value", "")
            if val:
                ec_numbers.append(f"EC {val}")
    except Exception:
        pass
    return "; ".join(ec_numbers)


def extract_molecular_function(data: dict) -> str:
    """
    UniProt JSON에서 분자 기능 설명 추출.
    우선순위: GO molecular function → FUNCTION comment → 권장명 전체
    """
    # 1. GO molecular function
    go_functions = []
    for xref in data.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "GO":
            for prop in xref.get("properties", []):
                if prop.get("key") == "GoTerm":
                    val = prop.get("value", "")
                    if val.startswith("F:"):  # F: = molecular_function
                        go_functions.append(val[2:].strip())
    if go_functions:
        return go_functions[0]  # 첫 번째 GO 분자 기능만 사용

    # 2. FUNCTION comment
    for comment in data.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                return texts[0].get("value", "")[:100]  # 100자 제한

    # 3. 대안 이름
    try:
        alt_names = (
            data.get("proteinDescription", {})
            .get("alternativeNames", [])
        )
        if alt_names:
            full_name = alt_names[0].get("fullName", {}).get("value", "")
            if full_name:
                return full_name
    except Exception:
        pass

    return ""


def collect_accessions(fasta_files: list[str]) -> list[str]:
    """FASTA 파일에서 UniProt accession 목록 추출."""
    accessions = []
    seen = set()
    for path in fasta_files:
        with open(path) as f:
            for record in SeqIO.parse(f, "fasta"):
                # 헤더: ACCESSION|...|...
                acc = record.description.split("|")[0].strip()
                if acc and acc not in seen:
                    accessions.append(acc)
                    seen.add(acc)
    return accessions


def main(args: argparse.Namespace):
    accessions = collect_accessions(args.input_files)
    logger.info(f"총 {len(accessions)}개 accession 수집")

    results = []
    for i, acc in enumerate(accessions, 1):
        if i % 50 == 0:
            logger.info(f"진행: {i}/{len(accessions)}")

        data = fetch_uniprot(acc)
        if data is None:
            results.append((acc, "", ""))
            continue

        mol_func = extract_molecular_function(data)
        ec_num = extract_ec_numbers(data)
        results.append((acc, mol_func, ec_num))

        # API 부하 방지 (초당 최대 5 요청 권장)
        time.sleep(args.delay)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# accession\tmolecular_function\tec_number\n")
        for acc, mol_func, ec_num in results:
            f.write(f"{acc}\t{mol_func}\t{ec_num}\n")

    n_mol = sum(1 for _, mf, _ in results if mf)
    n_ec = sum(1 for _, _, ec in results if ec)
    logger.info(f"저장 완료: {args.output}")
    logger.info(f"  molecular_function 수집: {n_mol}/{len(results)}")
    logger.info(f"  ec_number 수집: {n_ec}/{len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UniProt에서 분자 기능 및 EC 번호 수집 (v3/v4 헤더용)"
    )
    parser.add_argument(
        "--input_files", nargs="+", required=True,
        help="입력 FASTA 파일 목록"
    )
    parser.add_argument(
        "--output", default="annotations.tsv",
        help="출력 TSV 파일 경로. 기본값: annotations.tsv"
    )
    parser.add_argument(
        "--delay", type=float, default=0.2,
        help="API 요청 간 대기 시간(초). 기본값: 0.2"
    )
    args = parser.parse_args()
    main(args)
