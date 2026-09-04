from dotenv import load_dotenv
import os

load_dotenv()
import io
import datetime
import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pypdf import PdfReader
from docx import Document


# ==========================================
# Helpers
# ==========================================
def extract_text_from_file(uploaded_file):
    """Extract plain text from an uploaded PDF, DOCX, or TXT resume."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    else:  # .txt or fallback
        return data.decode("utf-8", errors="ignore")


def count_reports(messages):
    return sum(1 for m in messages if isinstance(m, AIMessage))


def validate_api_key(api_key: str):
    """
    Actually verify the key against OpenAI instead of just checking it's non-empty.
    Returns (is_valid: bool, message: str).
    """
    if not api_key or not api_key.strip():
        return False, "Please enter your API key."

    try:
        client = openai.OpenAI(api_key=api_key.strip())
        # Cheapest possible call that requires a valid, authenticated key.
        client.models.list()
        return True, "Key verified successfully."

    except openai.AuthenticationError:
        return False, "Invalid API key. Please check your key and try again."
    except openai.PermissionDeniedError:
        return False, "This key doesn't have permission to access the API."
    except openai.RateLimitError:
        return False, "Key looks valid, but you've hit a rate limit or quota. Try again shortly."
    except openai.APIConnectionError:
        return False, "Couldn't reach OpenAI's servers. Check your internet connection."
    except Exception as e:
        return False, f"Verification failed: {e}"


# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="AgentScope AI | Resume Intelligence",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. Global Styling
# ==========================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        :root {
            --ts-primary: #4F46E5;
            --ts-primary-dark: #3730A3;
            --ts-accent: #06B6D4;
            --ts-bg: #F8FAFC;
            --ts-card: #FFFFFF;
            --ts-border: #E2E8F0;
            --ts-text: #0F172A;
            --ts-muted: #64748B;
        }

        .stApp {
            background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
        }

        /* Hide default Streamlit chrome */
        #MainMenu, footer, header {visibility: hidden;}

        /* ---- Hero header ---- */
        .ts-hero {
            background: linear-gradient(120deg, var(--ts-primary) 0%, var(--ts-primary-dark) 55%, var(--ts-accent) 130%);
            border-radius: 18px;
            padding: 2.2rem 2.4rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
        }
        .ts-hero h1 {
            color: #ffffff;
            font-weight: 800;
            font-size: 2.1rem;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }
        .ts-hero p {
            color: #E0E7FF;
            font-size: 1.02rem;
            margin: 0;
            max-width: 640px;
        }
        .ts-badge {
            display: inline-block;
            background: rgba(255,255,255,0.16);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.35);
            border-radius: 999px;
            padding: 0.25rem 0.85rem;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-bottom: 0.9rem;
        }

        /* ---- Cards ---- */
        .ts-card {
            background: var(--ts-card);
            border: 1px solid var(--ts-border);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .ts-metric-label {
            font-size: 0.78rem;
            color: var(--ts-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .ts-metric-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--ts-text);
        }

        /* ---- Chat bubbles ---- */
        [data-testid="stChatMessage"] {
            background: var(--ts-card);
            border: 1px solid var(--ts-border);
            border-radius: 14px;
            padding: 0.4rem 0.2rem;
            margin-bottom: 0.7rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: #0F172A;
        }
        section[data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }
        section[data-testid="stSidebar"] .stButton button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #334155;
            background: #1E293B;
            font-weight: 600;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            border-color: var(--ts-accent);
            color: #ffffff !important;
        }

        /* ---- Primary button ---- */
        .stButton > button[kind="primary"] {
            background: linear-gradient(120deg, var(--ts-primary), var(--ts-primary-dark));
            border: none;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.55rem 1.2rem;
        }

        /* ---- Chat input ---- */
        [data-testid="stChatInput"] {
            border-radius: 14px;
        }

        .ts-footer {
            text-align: center;
            color: var(--ts-muted);
            font-size: 0.8rem;
            padding: 1.4rem 0 0.6rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 3. Session State
# ==========================================
defaults = {
    "api_authenticated": False,
    "api_key": "",
    "messages": [],
    "model_name": "gpt-4o-mini",
    "temperature": 0.3,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

SYSTEM_PROMPT = (
    "You are an expert HR recruiter and talent acquisition specialist. "
    "When a user provides a resume, CV, or candidate profile, analyze it thoroughly and output a structured report containing: "
    "1. Overall Candidate Fit Score (out of 10). "
    "2. Top 3 strengths (skills, experience, achievements) relevant to a professional role. "
    "3. Top 3 gaps or red flags (missing skills, experience gaps, inconsistencies). "
    "4. A brief actionable recommendation for the hiring manager (e.g., advance to interview, request more info, reject). "
    "Keep the layout clean using markdown headers and bullet points."
)

# ==========================================
# 4. Lock Screen (with real key verification)
# ==========================================
if not st.session_state["api_authenticated"]:
    col_l, col_mid, col_r = st.columns([1, 1.3, 1])
    with col_mid:
        st.markdown(
            """
            <div class="ts-hero" style="text-align:center;">
                <div class="ts-badge">🔒 SECURE ACCESS</div>
                <h1>AgentScope AI</h1>
                <p style="margin:0 auto;">
                    AI-powered resume screening for modern hiring teams.
                    Enter your OpenAI API key to unlock the dashboard.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Your key is stored only in this session and never logged.",
            )
            unlock = st.button("Unlock Dashboard →", type="primary", use_container_width=True)

            if unlock:
                with st.spinner("Verifying API key..."):
                    is_valid, message = validate_api_key(api_key_input)

                if is_valid:
                    st.session_state["api_key"] = api_key_input.strip()
                    st.session_state["api_authenticated"] = True
                    st.success(message)
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        st.caption("🔐 Your key never leaves this session. Nothing is stored server-side.")
    st.stop()

# ==========================================
# 5. Sidebar — Dashboard Controls
# ==========================================
with st.sidebar:
    st.markdown("### 🧑‍💼 AgentScope AI")
    st.caption("Resume Intelligence Dashboard")
    st.divider()

    st.markdown("**Session**")
    st.write(f"📅 {datetime.date.today().strftime('%B %d, %Y')}")
    st.write(f"📄 Reports generated: **{count_reports(st.session_state['messages'])}**")
    st.write(f"🔑 API status: **Connected**")

    st.divider()
    st.markdown("**Model Settings**")
    st.session_state["model_name"] = st.selectbox(
        "Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0
    )
    st.session_state["temperature"] = st.slider(
        "Response creativity", 0.0, 1.0, st.session_state["temperature"], 0.1
    )

    st.divider()
    if st.button("🗑️  Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()

    if st.button("🔓  Log out"):
        st.session_state["api_authenticated"] = False
        st.session_state["api_key"] = ""
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.caption("Built with LangChain + OpenAI + Streamlit")

# ==========================================
# 6. Main Header
# ==========================================
st.markdown(
    """
    <div class="ts-hero">
        <div class="ts-badge">✨ AI-POWERED SCREENING</div>
        <h1>Candidate Analysis Dashboard</h1>
        <p>Turn raw resumes and candidate profiles into structured hiring insights,
        fit scores, and interview recommendations — instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick stats row
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""<div class="ts-card"><div class="ts-metric-label">Reports Generated</div>
        <div class="ts-metric-value">{count_reports(st.session_state['messages'])}</div></div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="ts-card"><div class="ts-metric-label">Active Model</div>
        <div class="ts-metric-value" style="font-size:1.15rem;">{st.session_state['model_name']}</div></div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """<div class="ts-card"><div class="ts-metric-label">Status</div>
        <div class="ts-metric-value" style="font-size:1.15rem; color:#16A34A;">🟢 Ready</div></div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ==========================================
# 7. Ensure system message is tracked
# ==========================================
if not any(isinstance(x, SystemMessage) for x in st.session_state["messages"]):
    st.session_state["messages"].append(SystemMessage(content=SYSTEM_PROMPT))

# ==========================================
# 8. Chat History / Empty State
# ==========================================
has_history = len(st.session_state["messages"][1:]) > 0

if not has_history:
    st.markdown(
        """
        <div class="ts-card" style="text-align:center; padding: 2.5rem 1.5rem; margin-bottom: 1rem;">
            <div style="font-size: 2.2rem;">📋</div>
            <h4 style="margin: 0.5rem 0 0.3rem 0;">No candidates analyzed yet</h4>
            <p style="color:#64748B; margin:0;">Paste a resume below or attach a PDF, DOCX, or TXT file to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state["messages"][1:]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

# ==========================================
# 9. Chat Input
# ==========================================
chat_data = st.chat_input(
    "Paste a resume, or attach a PDF/DOCX/TXT file...",
    accept_file=True,
    file_type=["pdf", "docx", "txt"],
)

if chat_data:
    user_prompt = chat_data.text or ""
    attached_files = chat_data.files if hasattr(chat_data, "files") else []

    file_note = ""
    for f in attached_files:
        extracted = extract_text_from_file(f)
        file_note += f"\n\n--- Attached file: {f.name} ---\n{extracted}"

    full_prompt = (user_prompt + file_note).strip()

    st.session_state["messages"].append(HumanMessage(content=full_prompt))
    with st.chat_message("user"):
        if attached_files:
            for f in attached_files:
                st.markdown(f"📎 **{f.name}**")
        if user_prompt:
            st.markdown(user_prompt)

    try:
        chat_model = ChatOpenAI(
            model_name=st.session_state["model_name"],
            temperature=st.session_state["temperature"],
            openai_api_key=st.session_state["api_key"],
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing candidate profile..."):
                response = chat_model.invoke(st.session_state["messages"])
                st.markdown(response.content)
                st.session_state["messages"].append(AIMessage(content=response.content))

        st.rerun()

    except Exception as e:
        st.error(f"An error occurred while communicating with the OpenAI API: {e}")

# ==========================================
# 10. Footer
# ==========================================
st.markdown(
    """
    <div class="ts-footer">
        AgentScope AI · Confidential candidate data is processed in-session only
    </div>
    """,
    unsafe_allow_html=True,
)