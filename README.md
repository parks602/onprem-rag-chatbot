# CHATBOT & ONPREM & RAG 
Gravity Chatbot은 사내 규정, 복지, 휴가 등 다양한 사내 문서를 기반으로 질문에 답변해주는 RAG 기반 챗봇입니다.

이 시스템은 로컬 환경에서 동작하는 LLM과 FAISS 기반의 벡터 검색을 사용해, 인터넷 없이도 안전하고 정확한 답변을 제공합니다.

Streamlit으로 구현된 직관적인 UI를 통해 누구나 쉽게 사용할 수 있으며,
검색 결과와 함께 해당 정보가 출처 문서의 몇 페이지에서 추출되었는지도 함께 제공합니다.

온프레미스 환경에 최적화되어 있어 보안 이슈가 있는 내부망에서도 문제없이 동작합니다.


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

### 1.2 모델 다운로드
```bash
cd src
python model_downloader.py   # huggingface 기준이며, 기본은 Phi-4지만, 변경을 원한다면 model_id 등의 수정 필요
```
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
├── venv/                  # 가상환경
├── src/                   # Streamlit 메인 앱
├── local_models/          # LLM 모델델
├── docs/                  # 사규 문서 등 원본
└── chroma_db/             # Chroma 벡터 DB
```

---

## 💼 사용 예시

- `"회사에서 사용 가능한 메신저가 있나요?"`
- `"휴가관련된 내용을 알려주세요"`

---

## 🖼️ 사용 화면 예시

아래는 실제 사용 화면 예시입니다:

### 메인 화면
![Chatbot Example](asset/main.png)

### 질문 응답 예시
![Chatbot Example](asset/example.png)
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
