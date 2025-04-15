from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

"""_summary_
    1. PyPDFLoader(file)
    각 PDF를 페이지 단위로 Document 객체 리스트로 변환.
    Document에는 텍스트뿐만 아니라 metadata={"source": "파일명", "page": 페이지번호}가 자동 포함됨.
    이 메타정보는 나중에 출처 표시 + PDF Viewer 연동에 핵심 역할!

    2. RecursiveCharacterTextSplitter
    각 페이지는 길어서 바로 임베딩하면 검색 효율이 떨어짐.
    context 손상 없이 적절히 잘게 나누기 위해 사용.
    chunk_size=500, chunk_overlap=100: 500자씩 자르고, 100자는 겹쳐서 context 유지.

    3. SentenceTransformerEmbeddings
    Hugging Face의 all-MiniLM-L6-v2: 경량이면서도 성능 괜찮은 기본 모델.
    각 텍스트 조각을 768차원의 벡터로 임베딩함.

    4. Chroma.from_documents(...)
    문서 청크 + 임베딩 → Chroma 벡터 DB에 저장.
    내부적으로는 각 벡터를 faiss 혹은 duckdb 기반 구조로 저장함.
    "persist_directory"는 실제 DB 저장 위치 → 나중에 RAG 시스템에서 불러와서 검색 가능.
"""

# 1. PDF 파일 경로 불러오기
PDF_DIR = "../docs"
pdf_files = [
    os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")
]

# 2. 문서 로딩 (PDF 1개씩)
documents = []
for file in pdf_files:
    loader = PyPDFLoader(file)
    docs = loader.load()
    documents.extend(docs)  # 각 페이지 단위로 리스트에 추가됨

# 3. 문서 청크 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 4. 임베딩 모델 로딩
embedding_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",  # 또는 BM-K/KoSimCSE-roberta
    model_kwargs={"device": "cuda"},
)

# 5. Chroma 벡터 DB에 저장
vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embedding_model, persist_directory="../chroma_db"
)

vectorstore.persist()
print("[INFO] 모든 문서가 Chroma DB에 벡터로 저장되었습니다.")
