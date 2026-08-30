import os
import streamlit as st
import anthropic
from dotenv import load_dotenv
from rag import build_index, query_context, is_index_ready

# 로컬: .env 로드 / Streamlit Cloud: st.secrets 사용
load_dotenv()

def get_secret(key: str, default: str = "") -> str:
    """st.secrets(Streamlit Cloud) → os.environ(.env) → default 순으로 조회."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)

# ────────────────────────────────────────────────────────────────────────────
# 컴퓨터학습도우미 — 고정 지식 (시스템 프롬프트에 항상 포함)
# ────────────────────────────────────────────────────────────────────────────
BASE_PROFILE = """
당신은 초등학생을 위한 <컴퓨터학습도우미> 챗봇입니다.
컴퓨터 기초, 워드·엑셀·파워포인트 같은 OA 프로그램, AI와 인공지능, 프로그래밍 기초,
인터넷 안전 등을 초등학생이 이해하기 쉽게 설명하는 것이 역할입니다.

[답변 방침]
- 학생이 선택한 학습 주제와 난이도에 맞춰 설명합니다.
- 어려운 용어는 쉬운 예시나 비유를 들어 설명합니다.
- PDF 학습자료가 제공된 경우: 해당 자료를 우선 참고하여 답변하고, 출처를 명시합니다.
- 자료에 없는 내용은 "제공된 학습자료에서 확인할 수 없는 내용이에요"라고 솔직히 답합니다.
- 친절하고 격려하는 말투로 답변합니다.
- 모든 답변은 한국어로 합니다.
""".strip()

# 학습 주제별 세부 안내 (시스템 프롬프트에 추가)
TOPIC_GUIDES = {
    "💻 컴퓨터 기초": "이번 주제는 컴퓨터 기초입니다. 컴퓨터의 작동 원리, 하드웨어(본체, 모니터, 키보드 등)와 소프트웨어의 차이를 중심으로 설명하세요.",
    "📝 워드(Word) 사용법": "이번 주제는 워드(Word)입니다. 문서 작성, 글자 서식 지정, 그림 삽입, 인쇄 방법을 중심으로 설명하세요.",
    "📊 엑셀(Excel) 기초": "이번 주제는 엑셀(Excel)입니다. 표 만들기, 셀과 수식, 간단한 함수, 차트 만들기를 중심으로 설명하세요.",
    "🎨 파워포인트(PowerPoint)": "이번 주제는 파워포인트(PowerPoint)입니다. 슬라이드 만들기, 디자인 꾸미기, 애니메이션, 발표 준비 방법을 중심으로 설명하세요.",
    "🤖 AI와 인공지능": "이번 주제는 AI와 인공지능입니다. AI의 개념, 머신러닝·딥러닝의 기본 아이디어, 챗봇의 원리를 쉬운 비유로 설명하세요.",
    "🧠 프로그래밍 기초": "이번 주제는 프로그래밍 기초입니다. 알고리즘이 무엇인지, 순서도, 반복·조건 같은 코딩의 기본 개념을 쉬운 예시로 설명하세요.",
    "🌐 인터넷 안전": "이번 주제는 인터넷 안전입니다. 개인정보 보호, 안전한 비밀번호, 낯선 사람과의 온라인 대화 주의 등을 중심으로 설명하세요.",
    "📱 자유 질문": "이번 주제는 자유 질문입니다. 컴퓨터, OA, AI와 관련된 것이라면 어떤 질문이든 편하게 답해주세요.",
}

# 난이도별 설명 톤 안내
DIFFICULTY_GUIDES = {
    "⭐ 쉬움 (초등 1-2학년)": "초등학교 1~2학년 눈높이로, 아주 쉬운 단어와 짧은 문장, 실생활 비유를 사용해 설명하세요.",
    "⭐⭐ 보통 (초등 3-4학년)": "초등학교 3~4학년 눈높이로, 표준적인 낱말을 사용하되 어려운 개념은 예시를 들어 설명하세요.",
    "⭐⭐⭐ 어려움 (초등 5-6학년)": "초등학교 5~6학년 눈높이로, 조금 더 자세히 원리를 포함해 설명하세요.",
}

# 주제별 빠른 질문
QUICK_QUESTIONS = {
    "💻 컴퓨터 기초": ["컴퓨터는 어떻게 작동해?", "CPU가 뭐야?", "메모리가 뭐야?"],
    "📝 워드(Word) 사용법": ["글씨 크기는 어떻게 바꿔?", "그림은 어떻게 넣어?", "문서는 어떻게 인쇄해?"],
    "📊 엑셀(Excel) 기초": ["엑셀에서 표는 어떻게 만들어?", "함수가 뭐야?", "차트는 어떻게 만들어?"],
    "🎨 파워포인트(PowerPoint)": ["슬라이드는 어떻게 추가해?", "애니메이션은 어떻게 넣어?", "발표는 어떻게 준비해?"],
    "🤖 AI와 인공지능": ["인공지능이 뭐야?", "AI는 어떻게 학습해?", "챗봇은 뭐야?"],
    "🧠 프로그래밍 기초": ["알고리즘이 뭐야?", "코딩은 뭐야?", "순서도가 뭐야?"],
    "🌐 인터넷 안전": ["개인정보는 왜 지켜야 해?", "안전한 비밀번호는 어떻게 만들어?", "모르는 사람이 말 걸면 어떻게 해?"],
    "📱 자유 질문": ["컴퓨터에 대해 궁금한 게 있어!", "AI에 대해 더 알고 싶어!", "오늘 뭐 배울까?"],
}

# ────────────────────────────────────────────────────────────────────────────
# Streamlit 설정
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="컴퓨터 학습 도우미",
    page_icon="💻",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');

:root {
    --c-purple: #EBD4FF;
    --c-blue: #B8D5DD;
    --c-green: #DDF4D7;
    --c-cream: #F4ECD7;
    --c-pink: #FFD7D4;
}

html, body, [class*="css"] { font-family: 'Pretendard Variable', Pretendard, sans-serif; }

/* 전체 배경 */
.stApp { background-color: var(--c-cream); }

/* 사이드바 */
section[data-testid="stSidebar"] { background-color: var(--c-purple); }

/* 제목 강조선 */
h1 { border-bottom: 4px solid var(--c-blue); padding-bottom: 0.4rem; }

/* 버튼 (빠른 질문·로그인·인덱스 빌드·대화 초기화 공통) */
.stButton > button {
    background-color: var(--c-blue);
    color: #1e293b;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: background-color 0.15s ease;
}
.stButton > button:hover {
    background-color: var(--c-pink);
    color: #1e293b;
}

/* 참고 자료 박스 */
.source-box { background: var(--c-green); border-left:3px solid #0891b2; padding:0.6rem 0.8rem;
              font-size:0.82rem; color:#334155; border-radius:0 6px 6px 0; margin-top:0.5rem; }

/* 채팅 입력창 */
[data-testid="stChatInput"] { background-color: var(--c-cream); }
</style>
""", unsafe_allow_html=True)

st.title("💻 컴퓨터 학습 도우미")
st.caption("초등학생을 위한 컴퓨터·OA·인공지능 학습 도우미예요! 무엇이든 물어보세요.")

# ────────────────────────────────────────────────────────────────────────────
# 사이드바 — 학습 설정 & API 키 & RAG 인덱스 관리
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📚 학습 설정")

    learning_topic = st.selectbox(
        "배우고 싶은 주제를 선택하세요:",
        list(TOPIC_GUIDES.keys()),
    )

    difficulty = st.select_slider(
        "난이도를 선택하세요:",
        options=list(DIFFICULTY_GUIDES.keys()),
    )

    st.divider()
    st.header("⚙️ 설정")

    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = st.text_input(
            "Anthropic API 키",
            type="password",
            placeholder="sk-ant-...",
        )

    st.divider()
    st.subheader("📚 PDF 인덱스")

    ready = is_index_ready()
    if ready:
        st.success("인덱스 준비 완료")
    else:
        st.warning("인덱스 없음 — 관리자 로그인 후 빌드하세요.")

    # 관리자 잠금 영역
    if "admin_unlocked" not in st.session_state:
        st.session_state.admin_unlocked = False

    if not st.session_state.admin_unlocked:
        admin_pw = st.text_input("관리자 비밀번호", type="password", placeholder="비밀번호 입력…")
        if st.button("로그인", use_container_width=True):
            correct = get_secret("ADMIN_PASSWORD", "admin1234")
            if admin_pw == correct:
                st.session_state.admin_unlocked = True
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다.")
    else:
        st.success("관리자 모드")
        if st.button("🔄 인덱스 빌드 / 재빌드", use_container_width=True):
            with st.spinner("PDF 파싱 & 임베딩 중… (첫 실행 시 수 분 소요)"):
                try:
                    count = build_index()
                    st.success(f"완료: {count}개 청크 저장")
                    st.rerun()
                except FileNotFoundError as e:
                    st.error(str(e))
        if st.button("잠금", use_container_width=True, type="secondary"):
            st.session_state.admin_unlocked = False
            st.rerun()

    use_rag = st.toggle("RAG 사용", value=ready, disabled=not ready)

    st.divider()
    n_results = st.slider("검색할 청크 수", 1, 10, 5)

if not api_key:
    st.info("사이드바에서 Anthropic API 키를 입력하세요.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

# ────────────────────────────────────────────────────────────────────────────
# 빠른 질문
# ────────────────────────────────────────────────────────────────────────────
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

st.subheader("💡 빠른 질문")
qcols = st.columns(len(QUICK_QUESTIONS[learning_topic]))
for col, q in zip(qcols, QUICK_QUESTIONS[learning_topic]):
    with col:
        if st.button(q, use_container_width=True, key=f"quick_{q}"):
            st.session_state.pending_question = q
            st.rerun()

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# 대화
# ────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("질문을 입력하세요…")
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG 검색
    rag_context = ""
    if use_rag and ready:
        rag_context = query_context(prompt, n_results=n_results)

    # 시스템 프롬프트 구성: 고정 지식 + 주제 + 난이도 + (있다면) RAG 컨텍스트
    system_prompt = (
        BASE_PROFILE
        + "\n\n[선택된 학습 주제]\n" + TOPIC_GUIDES[learning_topic]
        + "\n\n[선택된 난이도]\n" + DIFFICULTY_GUIDES[difficulty]
    )
    if rag_context:
        system_prompt += f"\n\n[PDF 자료에서 검색된 관련 내용]\n{rag_context}"

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중…"):
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                system=system_prompt,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            )
            answer = response.content[0].text

        st.markdown(answer)

        # 참고 자료 표시
        if rag_context:
            sources = set()
            for line in rag_context.splitlines():
                if line.startswith("[출처:"):
                    src = line.split("|")[0].replace("[출처:", "").strip()
                    sources.add(src)
            if sources:
                st.markdown(
                    "<div class='source-box'>📄 참고 자료: "
                    + ", ".join(sorted(sources))
                    + "</div>",
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append({"role": "assistant", "content": answer})

# 초기화 버튼
if st.session_state.messages:
    if st.button("🔄 새로운 대화 시작", type="secondary"):
        st.session_state.messages = []
        st.rerun()
