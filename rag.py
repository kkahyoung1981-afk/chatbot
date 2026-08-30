"""
rag.py — PDF → FAISS 벡터 인덱스 구축 및 검색 모듈

저장 파일:
  faiss.index  — FAISS 인덱스 (벡터)
  chunks.pkl   — 청크 텍스트 + 메타데이터

CLI:
  python rag.py --build              # pdf/ 폴더 인덱싱
  python rag.py --query "딥러닝이란?" # 검색 테스트
  python rag.py --build --query "..." # 인덱싱 + 검색
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import fitz
except ImportError:
    import pymupdf as fitz

# ─────────────────────────────────────────────
# 경로 & 모델 설정
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
PDF_DIR    = BASE_DIR / "pdf"
INDEX_PATH = BASE_DIR / "faiss.index"
CHUNKS_PATH = BASE_DIR / "chunks.pkl"

# multilingual-e5-small: ~120MB, 한국어 지원, 384차원
EMBED_MODEL = "intfloat/multilingual-e5-small"

_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def _embed_passages(texts: list[str]) -> np.ndarray:
    """문서 청크 임베딩 — e5 모델은 'passage: ' 접두사 필요."""
    prefixed = ["passage: " + t for t in texts]
    vecs = _get_model().encode(prefixed, normalize_embeddings=True, show_progress_bar=True)
    return np.array(vecs, dtype="float32")

def _embed_query(text: str) -> np.ndarray:
    """쿼리 임베딩 — e5 모델은 'query: ' 접두사 필요."""
    vec = _get_model().encode(["query: " + text], normalize_embeddings=True)
    return np.array(vec, dtype="float32")

# ─────────────────────────────────────────────
# PDF 파싱
# ─────────────────────────────────────────────
def _parse_pdf(path: Path) -> str:
    doc = fitz.open(str(path))
    pages = [page.get_text("text") for page in doc if page.get_text("text").strip()]
    doc.close()
    return "\n".join(pages)

def _load_all_pdfs(pdf_dir: Path) -> list[dict]:
    pdfs = sorted(pdf_dir.glob("**/*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"{pdf_dir} 에 PDF 파일이 없습니다.")
    result = []
    for path in pdfs:
        print(f"  파싱 중: {path.name}")
        text = _parse_pdf(path)
        if text.strip():
            result.append({"filename": path.name, "text": text})
    return result

# ─────────────────────────────────────────────
# 청킹
# ─────────────────────────────────────────────
def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]+", " ", text).strip()

def _chunk(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    sentences = re.split(r"(?<=[.!?。！？\n])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= size:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if len(current) > overlap else current
            current = (tail + " " + sent).strip() if tail else sent
    if current:
        chunks.append(current)
    return chunks

# ─────────────────────────────────────────────
# 인덱스 구축 (공개 API)
# ─────────────────────────────────────────────
def build_index(
    pdf_dir: Path = PDF_DIR,
    chunk_size: int = 400,
    overlap: int = 80,
) -> int:
    """PDF를 파싱·청킹·임베딩해 FAISS 인덱스를 저장합니다."""
    print(f"\n[RAG] PDF 폴더: {pdf_dir}")
    docs = _load_all_pdfs(pdf_dir)

    all_chunks: list[dict] = []
    for doc in docs:
        text = _clean(doc["text"])
        for i, chunk in enumerate(_chunk(text, chunk_size, overlap)):
            all_chunks.append({
                "text": chunk,
                "source": doc["filename"],
                "chunk_index": i,
            })
        print(f"  {doc['filename']}: {i + 1}개 청크")

    texts = [c["text"] for c in all_chunks]
    print(f"\n[RAG] 임베딩 중 (총 {len(texts)}개)…")
    vecs = _embed_passages(texts)

    # FAISS IndexFlatIP: 내적 유사도 (정규화된 벡터 → cosine similarity)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"[RAG] 완료: {len(all_chunks)}개 청크 → {INDEX_PATH.name}, {CHUNKS_PATH.name}")
    return len(all_chunks)

# ─────────────────────────────────────────────
# 검색 (공개 API)
# ─────────────────────────────────────────────
def query_context(question: str, n_results: int = 5) -> str:
    """질문과 유사한 청크를 검색해 컨텍스트 문자열로 반환."""
    if not is_index_ready():
        return ""

    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    q_vec = _embed_query(question)
    n = min(n_results, len(chunks))
    scores, indices = index.search(q_vec, n)

    parts = []
    for score, idx in zip(scores[0], indices[0]):
        c = chunks[idx]
        parts.append(f"[출처: {c['source']} | 유사도: {score:.2f}]\n{c['text']}")

    return "\n\n---\n\n".join(parts)

def is_index_ready() -> bool:
    return INDEX_PATH.exists() and CHUNKS_PATH.exists()

def load_chunks_df():
    """visualize.py용 — 청크 메타데이터 + 임베딩을 반환."""
    if not is_index_ready():
        return None, None
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(str(INDEX_PATH))
    # FAISS에서 전체 벡터 추출
    vecs = np.zeros((index.ntotal, index.d), dtype="float32")
    index.reconstruct_n(0, index.ntotal, vecs)
    return chunks, vecs

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PDF RAG 파이프라인 (FAISS)")
    parser.add_argument("--build", action="store_true", help="PDF 인덱싱")
    parser.add_argument("--query", type=str, default="", help="검색 테스트")
    parser.add_argument("--n", type=int, default=5, help="반환 청크 수")
    parser.add_argument("--pdf-dir", type=str, default=str(PDF_DIR))
    args = parser.parse_args()

    if not args.build and not args.query:
        parser.print_help()
        sys.exit(0)

    if args.build:
        build_index(pdf_dir=Path(args.pdf_dir))

    if args.query:
        print(f"\n[RAG] 질문: {args.query}")
        ctx = query_context(args.query, n_results=args.n)
        print("\n=== 검색 결과 ===\n" + ctx if ctx else "인덱스 없음. --build 먼저 실행하세요.")

if __name__ == "__main__":
    main()
