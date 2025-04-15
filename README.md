## 🔧 설치 및 실행

### 1. 환경 구성

```bash
git clone https://git.gravity.co.kr/parks602/gravity_chatbot.git
cd gravity_chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 1.1 pdf 파일 생성
- docs 폴더에 RAG에 사용할 pdf 파일을 복사

### 2. Streamlit 실행

```bash
cd src
python main.py
streamlit run ui.py
```

> 기본 포트: `http://localhost:8501`

---

## 📂 프로젝트 구조

```bash
gravity_chatbot/
├── app.py                 # Streamlit 메인 앱
├── rag/                   # RAG 관련 모듈
├── llm/                   # LLM 호출 모듈
├── data/                  # 사규 문서 등 원본
├── vectorstore/           # FAISS/Chroma 등 벡터 DB
└── utils/                 # 공통 유틸
```

---

## 💼 사용 예시

- `"회사에서 사용 가능한 메신저가 있나요?"`
- `"휴가관련된 내용을 알려주세요"`

---

## 🔐 온프레미스 / 보안

- 인터넷 연결 없이 완전한 로컬 환경에서 동작  
- 외부 API 호출 없음  
- 내부 정책 문서 유출 방지  

---

## 🧑‍💻 개발자 정보

- **만든이**: Kunwoo Park  
- **Email**: parks602@gravity.co.kr  
- **회사**: Gravity Co., Ltd.

---

## 📜 라이선스

이 프로젝트는 내부 전용이며, Gravity 사내에서만 사용 가능합니다.
