from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path
from rag_handler import RAGHandler
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAGHandler 초기화
rag_handler = RAGHandler()

# PDF 파일을 저장할 경로 (FastAPI 서버가 이 파일을 서빙할 수 있도록 해야함)
pdf_storage_directory = "../docs/"


class QueryRequest(BaseModel):
    query: str


# API 엔드포인트: 질문을 받으면 답변과 관련 PDF를 반환
@app.post("/ask/")
async def ask_question(query: QueryRequest, request: Request):
    # RAGHandler를 통해 질문에 대한 답변과 관련 문서 메타데이터를 받는다.
    result = rag_handler.ask(query.query)

    # 답변과 관련된 PDF 파일 이름 가져오기 (출처 문서에서 추출된 PDF 파일)
    pdf_files = [metadata.get("source") for metadata in result["sources"]]
    # PDF 파일이 없다면 오류 처리
    if not pdf_files:
        raise HTTPException(status_code=404, detail="관련 PDF 파일을 찾을 수 없습니다.")

    # PDF 파일 경로 설정 (여기서는 첫 번째 파일을 선택)
    pdf_file_path = Path(pdf_storage_directory) / pdf_files[0]

    # 파일이 존재하지 않으면 오류 처리
    if not pdf_file_path.exists():
        raise HTTPException(status_code=404, detail="PDF 파일을 찾을 수 없습니다.")

    # 질문에 대한 답변과 함께 PDF 파일 경로 반환
    return {
        "answer": result["answer"],
        "pdf_url": f"{pdf_files[0]}",
        "pdf_page": result["sources_page"][0],
    }


# API 엔드포인트: PDF 파일을 반환
@app.get("/files/{filename}")
async def get_pdf(filename: str):
    file_path = Path("../docs") / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="PDF 파일을 찾을 수 없습니다.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8201)
