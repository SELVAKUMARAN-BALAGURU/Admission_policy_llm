import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama
import time
import base64, os
import markdown as md

st.set_page_config(
    page_title="SASTRA Admissions 2026",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ─────────────────────────────
APP BACKGROUND
───────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #ffffff !important;
    color: #111827 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(31,78,121,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(0,120,100,0.05) 0%, transparent 60%),
        #ffffff !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }

.block-container {
    padding: 0 2rem 180px 2rem !important;
    max-width: 100% !important;
}


/* ─────────────────────────────
SIDEBAR
───────────────────────────── */

[data-testid="stSidebar"] {
    background: #f5f7fa !important;
    border-right: 1px solid #e5e7eb !important;
}

[data-testid="stSidebar"] * {
    color: #111827 !important;
}

.sidebar-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6b7280 !important;
    margin: 18px 0 10px;
}

/* Sidebar items */

.category-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    border-radius: 10px;
    margin-bottom: 4px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    transition: all 0.2s ease;
}

.category-item:hover {
    background: #eef2ff !important;
    border-color: #c7d2fe !important;
}

.category-icon {
    font-size: 14px;
    width: 20px;
}

.category-text {
    font-size: 13px;
    color: #374151 !important;
}


/* ─────────────────────────────
MODEL BADGE
───────────────────────────── */

.model-active-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 12px 14px;
}

.model-active-dot {
    width: 8px;
    height: 8px;
    background: #22c55e;
    border-radius: 50%;
}

.model-active-name {
    font-size: 14px;
    font-weight: 600;
    color: #1f4e79 !important;
}

.model-active-sub {
    font-size: 11px;
    color: #6b7280 !important;
}


/* ─────────────────────────────
LINK BUTTONS
───────────────────────────── */

.stLinkButton a {
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #1f4e79 !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    width: 100% !important;
}

.stLinkButton a:hover {
    background: #dbeafe !important;
}


/* ─────────────────────────────
HERO HEADER
───────────────────────────── */

.hero-logo {
    display: flex;
    justify-content: center;
    margin-bottom: 18px;
    padding-top: 2rem;
}

.hero-logo img {
    height: 110px;
}

.hero {
    padding: 0.5rem 2rem 2rem;
    text-align: center;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 900;
    background: linear-gradient(135deg,#1f4e79,#2563eb,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size: 15px;
    color: #6b7280;
}

.hero-line {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    margin: 18px auto 0;
}


/* ─────────────────────────────
STATUS BAR
───────────────────────────── */

.status-bar {
    display: flex;
    justify-content: center;
    gap: 20px;
    padding: 10px 20px;
    border-bottom: 1px solid #e5e7eb;
}

.status-dot {
    width: 7px;
    height: 7px;
    background: #22c55e;
    border-radius: 50%;
}

.status-item {
    font-size: 12px;
    color: #6b7280;
}


/* ─────────────────────────────
CHAT MESSAGES
───────────────────────────── */

.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 14px 0;
}

.msg-user-bubble {
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
    color: #fff;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
}

.msg-assistant-wrap { margin: 16px 0; }

.msg-meta {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
}

.badge-model {
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    color: #2563eb;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
}

.badge-time {
    background: #ecfeff;
    border: 1px solid #a5f3fc;
    color: #0891b2;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
}

.msg-assistant-bubble {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    color: #111827;
    padding: 16px 20px;
    border-radius: 4px 18px 18px 18px;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}


/* ─────────────────────────────
CHAT INPUT
───────────────────────────── */

[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 58px !important;
    left: 244px !important;
    right: 0 !important;
    background: #ffffff !important;
    border-top: 1px solid #e5e7eb !important;
    padding: 12px 2rem !important;
    z-index: 999 !important;
}

[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 14px !important;
    color: #111827 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 14px 18px !important; 
    transition: border-color 0.2s !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #2563eb !important;
}


/* ─────────────────────────────
FOOTER
───────────────────────────── */

.footer {
    position: fixed; 
    bottom: 0;
    left: 0;
    right: 0;
    height: 58px;
    background: #ffffff;
    border-top: 1px solid #e5e7eb;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 998; 
    backdrop-filter: blur(20px);
}

.footer-text {
    font-size: 12px;
    color: #6b7280;
    font-family: 'DM Sans', sans-serif; 
    letter-spacing: 0.3px;
}

.footer-text b {
    color: #1f2937;
    font-weight: 500;
}


/* ─────────────────────────────
FOLLOW-UP QUESTION
───────────────────────────── */

.followup-wrap {
    margin: 10px 0 20px 0;
    padding: 14px 18px;
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-left: 3px solid #2563eb;
    border-radius: 0 12px 12px 0;
    max-width: 85%;
}

.followup-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2563eb;
    display: block;
    margin-bottom: 6px;
}

.followup-bubble {
    font-size: 14px;
    color: #1e3a5f;
    font-style: italic;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:

    # Active model display
    st.markdown("<div class='sidebar-label'>Active Model</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='model-active-badge'>
        <div class='model-active-dot'></div>
        <div>
            <div class='model-active-name'>llama3</div>
            <div class='model-active-sub'>Meta · 8B · Running locally</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:18px 0 4px'>", unsafe_allow_html=True)

    # Query categories
    st.markdown("<div class='sidebar-label'>Query Categories</div>", unsafe_allow_html=True)

    categories = [
        ("🎓", "Eligibility Criteria"),
        ("📘", "UG Programs"),
        ("📗", "PG Programs"),
        ("🔬", "PhD Programs"),
        ("💰", "Academic Fees"),
        ("🏠", "Hostel Fees"),
        ("❓", "FAQ"),
        ("🌍", "NRI Guidelines"),
    ]
    items_html = ""
    for icon, label in categories:
        items_html += f"""
        <div class='category-item'>
            <span class='category-icon'>{icon}</span>
            <span class='category-text'>{label}</span>
        </div>"""
    st.markdown(items_html, unsafe_allow_html=True)

    st.markdown("<hr style='margin:18px 0 4px'>", unsafe_allow_html=True)

    # Quick links
    st.markdown("<div class='sidebar-label'>Quick Links</div>", unsafe_allow_html=True)
    st.link_button("↗  Admissions FAQ", "https://www.sastra.edu/admissions/admissions-faq.html")
    st.link_button("↗  SASTRA Website", "https://www.sastra.edu")


# ─────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────
logo_html = ""
logo_path = "D:\\Admission_policy_llm\\sas40_WO_TPJ.jpg.jpeg"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<div class="hero-logo"><img src="data:image/jpeg;base64,{logo_b64}" alt="SASTRA Logo"></div>'

st.markdown(f"""
<div class="hero">
    {logo_html}
    <div class="hero-title">Admissions 2026</div>
    <div class="hero-sub">Ask anything about programs, eligibility, fees & more</div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="status-bar">
    <span class="status-item"><span class="status-dot"></span>Knowledge base active</span>
    <span class="status-item">· UG &nbsp;·&nbsp; PG &nbsp;·&nbsp; PhD &nbsp;·&nbsp; Fees &nbsp;·&nbsp; Hostel &nbsp;·&nbsp; NRI ·</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD VECTOR DB
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vector_db():
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_db = FAISS.load_local(
        "admission_vector_db",
        embedding,
        allow_dangerous_deserialization=True
    )
    return vector_db

with st.spinner("Initialising knowledge base..."):
    vector_db = load_vector_db()


# ─────────────────────────────────────────
# CHAT MEMORY
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────
# DISPLAY HISTORY
# ─────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-user-bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        rendered = md.markdown(msg["content"], extensions=["extra", "nl2br"])
        st.markdown(f"""
        <div class="msg-assistant-wrap">
            <div class="msg-meta">
                <span class="badge-model">🤖 llama3</span>
                <span class="badge-time">⏱ {msg.get("elapsed","")}</span>
            </div>
            <div class="msg-assistant-bubble">{rendered}</div>
        </div>
        """, unsafe_allow_html=True)
        # ── Replay follow-up question from history ──
        if msg.get("followup"):
            rendered_fq = md.markdown(msg["followup"], extensions=["extra", "nl2br"])
            st.markdown(f"""
            <div class="followup-wrap">
                <span class="followup-label">💬 Follow-up</span>
                <div class="followup-bubble">{rendered_fq}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────
query = st.chat_input("  Enter your query about admission policies...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f"""
    <div class="msg-user">
        <div class="msg-user-bubble">{query}</div>
    </div>
    """, unsafe_allow_html=True)

    ph = st.empty()
    ph.info("🔎 Analyzing query...")
    time.sleep(0.6)

    ph.info("📚 Retrieving from admission knowledge base...")
    docs = vector_db.similarity_search(query, k=5)
    context = "\n\n".join([d.page_content for d in docs])
    time.sleep(0.4)

    ph.info("🧠 Generating answer with llama3...")
    print(context)

    prompt = f"""You are an AI assistant for SASTRA University admissions.
                    Use ONLY the context below. 
                    Whatever the query give detailed answer based on the context.Dont make hallucinations.Strict to the context.
                    If there any mail id or contact number in the context and if it is related to the query asked then give that more precisely without any mistake.
                    you need to provide the nri admission link and contact number if and only if the query is related to nri.
                    If the answer is not there, say: "I cannot find this in the admission policy."

                Context:
                {context}

                Question: 
                {query}

                Answer clearly and concisely.
            """

    t0 = time.time()
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed = round(time.time() - t0, 2)
    answer = response["message"]["content"]

    # ── Second pass: generate follow-up question ──────────────────────────
    ph.info("💬 Generating follow-up question...")
    followup_prompt = f"""You are an AI assistant for SASTRA University admissions.
Based on the user's question and the answer provided below, generate exactly ONE short,
relevant follow-up question that would naturally continue the conversation.
The follow-up question should be directly related to the topic and help the user
explore the admission process further.
Return ONLY the question — no explanation, no prefix like "Follow-up:", just the question itself.

User's Question: {query}

Answer Given:
{answer}

Follow-up Question:"""

    followup_response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": followup_prompt}]
    )
    followup_question = followup_response["message"]["content"].strip()

    ph.empty()

    rendered_answer = md.markdown(answer, extensions=["extra", "nl2br"])

    st.markdown(f"""
    <div class="msg-assistant-wrap">
        <div class="msg-meta">
            <span class="badge-model">🤖 llama3</span>
            <span class="badge-time">⏱ {elapsed}s</span>
        </div>
        <div class="msg-assistant-bubble">{rendered_answer}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Render follow-up card ─────────────────────────────────────────────
    rendered_followup = md.markdown(followup_question, extensions=["extra", "nl2br"])
    st.markdown(f"""
    <div class="followup-wrap">
        <span class="followup-label">💬 Follow-up</span>
        <div class="followup-bubble">{rendered_followup}</div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "elapsed": f"{elapsed}s",
        "followup": followup_question
    })


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-text">
        <span style="color:#1d4ed8; font-weight:700; font-size:14.5px;">Mentor: Dr. G.R. Brindha</span>
        <span style="color:#3b82f6; font-size:13px;">&nbsp;·&nbsp; SoC</span>
        <span style="color:#93c5fd;">&nbsp;&nbsp;|&nbsp;&nbsp;</span>
        Developed by <b>Selvakumaran Balaguru</b>, <b>Rishi Grace M</b>, <b>Harrish AR</b>
        &nbsp;·&nbsp; AI&DS, School of Computing &nbsp;·&nbsp; © 2026 SASTRA University
    </div>
</div>
""", unsafe_allow_html=True)  

