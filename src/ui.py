import streamlit as st
import time
import requests
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from streamlit_pdf_viewer import pdf_viewer
from urllib.parse import quote

st.set_page_config(layout="wide")
st.markdown(
    """
        <style>
                .block-container {
                    padding-top: 2rem;
                    padding-bottom: 1rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)
API_URL = "http://localhost:8201/ask/"

# PDF 파일을 열 수 있도록 Streamlit에서 제공하는 링크를 설정
FILE_URL = "http://localhost:8201/files/"


def ask_question(query: str):
    # FastAPI 서버로 질문 보내기
    response = requests.post(API_URL, data={"query": query})

    if response.status_code == 200:
        # 서버에서 받은 JSON 응답
        data = response.json()
        return data["answer"], data["pdf_url"]
    else:
        st.error(f"Error: {response.status_code} - {response.json()['detail']}")
        return None, None


def stream_data(sentence):
    for word in sentence.split(" "):
        yield word + " "
        time.sleep(0.02)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "안녕하세요! 사규에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!",
        }
    ]

if "currunt_message" not in st.session_state:
    st.session_state.currunt_message = None

if "currunt_file_info" not in st.session_state:
    st.session_state.currunt_file_info = None

if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = ""

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

empyt1, con1, con2, empty2 = st.columns([0.01, 0.5, 0.5, 0.01])
empty1, con3, empty2 = st.columns([0.01, 1.0, 0.01])

if query := st.chat_input("질문을 입력해주세요."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("문서에서 내용을 찾는 중 입니다."):
        try:
            # FastAPI 서버에 POST 요청 전송 및 응답 로깅
            response = requests.post(
                "http://172.31.202.44:8201/ask/", json={"query": query}
            )

            if response.status_code == 200:
                result = response.json()["answer"]
                file_name = response.json()["pdf_url"]
                page_num = response.json()["pdf_page"]["page"]
                info_message = f"""해당 내용은 '{file_name})'의 {str(page_num)} 페이지에 근거했습니다."""
                st.session_state.currunt_message = result
                st.session_state.currunt_file_info = info_message
                # PDF 파일 다운로드 요청
                encoded_filename = quote(file_name)
                data_response = requests.get(
                    f"http://172.31.202.44:8201/files/{encoded_filename}"
                )

                if data_response.status_code == 200:
                    st.session_state.pdf_data = data_response.content
                else:
                    st.error("Failed to download file")

                st.session_state.pdf_name = file_name
            else:
                st.error("Failed to get response from server")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

with con1:
    st.subheader("대화")
    with st.container(height=550):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        if st.session_state.currunt_message is not None:
            with st.chat_message("assistant"):
                st.write_stream(stream_data(st.session_state.currunt_message))
                st.info(st.session_state.currunt_file_info)
                st.session_state.messages.append(
                    {"role": "assistant", "content": st.session_state.currunt_message}
                )
with con2:
    st.subheader("원본 데이터")
    with st.container(height=550):
        if st.session_state.pdf_data != "":
            with st.container():
                pdf_viewer(input=st.session_state.pdf_data)

with st.sidebar:
    st.markdown("# 🏢 사내 사규 챗봇")
    st.markdown("---")
    st.markdown("## 📎 사규 원문 다운로드")
    if st.session_state.pdf_data != "":
        st.write(f"문서 이름 - {st.session_state.pdf_name}")
        st.download_button(
            label=f"{st.session_state.pdf_name} Download",
            data=st.session_state.pdf_data,
            file_name=st.session_state.pdf_name,
            mime="application/pdf",
            use_container_width=True,
        )
    st.markdown("---")

    if st.button(
        "세션 초기화",
        type="primary",
        use_container_width=True,
    ):
        # 세션 상태 초기화 (세션 상태에 저장된 모든 값 삭제)
        st.session_state.clear()

        # 세션 초기화 후 앱을 재시작 (rerun)
        st.rerun()
    st.markdown("---")
    st.markdown(
        """<small>문의 <br> 시스템디비전 -  플랫폼개발 Unit 박건우</small>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<small>© 2025 Kunwoo Park. All Rights Reserved. <br>
            Follow us : <br>
            H.P 010-3302-6840<br>
            Email parks602@gravity.co.kr
        </small>""",
        unsafe_allow_html=True,
    )
