from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch

torch.cuda.empty_cache()  # 캐시 메모리 해제
torch.backends.cuda.matmul.allow_tf32 = True  # 계산 최적화


class RAGHandler:
    def __init__(
        self,
        embedding_model_name="jhgan/ko-sroberta-multitask",
        llm_model_path="../local_models/Phi-4",
        persist_directory="../chroma_db",
    ):
        # 1. 임베딩 모델 로드
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name, model_kwargs={"device": "cpu"}
        )

        # 2. Chroma 벡터DB 연결
        """
        Chroma는 벡터 데이터베이스로, 문서 임베딩을 저장하고 효율적으로 검색할 수 있는 시스템입니다. 문서가 벡터로 변환되어 저장되고, 나중에 유사한 문서를 검색하는 데 사용됩니다.
        persist_directory는 벡터 DB가 저장될 디렉터리 경로입니다. 문서가 임베딩된 벡터들은 이 디렉터리에 저장됩니다.
        embedding_function=self.embeddings는 위에서 로드한 임베딩 모델을 사용하여 문서를 벡터화합니다.
        search_kwargs={"k": 2}는 검색 시 상위 3개의 문서를 반환하도록 지정하는 옵션입니다. k 값은 반환할 문서의 수를 결정합니다.
        """
        self.db = Chroma(
            persist_directory=persist_directory, embedding_function=self.embeddings
        )
        self.retriever = self.db.as_retriever(search_kwargs={"k": 2})  # top-3 문서 검색

        # 3. LLM 로드
        """
        Hugging Face에서 제공하는 AutoTokenizer와 AutoModelForCausalLM 클래스를 사용하여 **LLM (Large Language Model)**을 로드합니다.
        AutoTokenizer는 텍스트를 모델이 처리할 수 있는 형식으로 변환하는 데 사용됩니다.
        AutoModelForCausalLM는 텍스트 생성 모델을 로드합니다. 여기서는 "Phi-4" 모델을 사용하고 있습니다.
        """
        tokenizer = AutoTokenizer.from_pretrained(llm_model_path, local_files_only=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                local_files_only=True,
                torch_dtype=torch.float16,
            )
            .to("cuda")
            .eval()
        )

        # 4. LLM 파이프라인 설정
        """
        pipeline은 Hugging Face의 텍스트 생성 파이프라인을 설정하는 데 사용됩니다.
        device=0은 첫 번째 GPU를 사용하여 모델을 실행하도록 설정합니다.
        max_new_tokens=512는 최대 생성되는 토큰 수를 512개로 제한합니다.
        do_sample=True는 샘플링 기법을 사용하여 생성된 텍스트의 다양성을 높입니다.
        top_p=0.95는 확률적 샘플링 방법인 nucleus sampling을 사용하여, 생성된 토큰들 중 상위 95% 확률에 해당하는 토큰만 고려하여 텍스트를 생성합니다.
        temperature=0.7은 생성된 텍스트의 창의성을 제어하는 매개변수로, 1.0보다 낮으면 더 결정적인(덜 창의적인) 텍스트를 생성하고, 높은 값은 더 창의적인 텍스트를 생성합니다.
        """
        pipe = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0,
            max_new_tokens=256,
            do_sample=False,
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

        prompt_template = PromptTemplate.from_template(
            """
            당신은 회사 내부 문서를 기반으로 질문에 답하는 사내 전용 AI 비서입니다.
            항상 제공된 문서 정보에 기반해서만 답변해야 하며, 문서에 없는 정보에 대해서는 절대 추측하지 마세요.
            반드시 문서에 있는 정보만으로 답하고, 문서에 없으면 '해당 정보는 문서에 없습니다.'라고 정중하게 답하세요.
            모든 답변은 신뢰성 있고 간결하게 표현되어야 합니다.

            Question: {question} 
            Context: {context} 

            Answer:
        """
        )
        # 5. QA 체인 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,  # 문서 출처 포함
            chain_type_kwargs={"prompt": prompt_template},  # 프롬프트 템플릿 설정
        )

    def ask(self, query: str):
        """
        Args:
            query (str): _description_
        Returns:
            _type_: _description_
        """
        result = self.qa_chain.invoke(query)
        # "Answer:" 이후의 텍스트만 추출
        answer_text = result["result"]

        if "Answer:" in answer_text:
            answer_text = answer_text.split("Answer:", 1)[-1].strip()
        return {
            "answer": answer_text,
            "sources": [
                {"source": Path(doc.metadata["source"]).name}  # 파일 이름만 추출
                for doc in result["source_documents"]
            ],
            "sources_page": [
                {"page": doc.metadata["page_label"]}  # 페이지 번호 추출
                for doc in result["source_documents"]
            ],
        }
