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
![Chatbot Example](asset/main.png)
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
