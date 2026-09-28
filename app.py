"""
ClauseGuard dashboard.
Run from the project root:  streamlit run app.py
"""

import sys
import html
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st
from PyPDF2 import PdfReader
from pipeline import build_graph

st.set_page_config(page_title="ClauseGuard", page_icon="⚖️", layout="wide")

# ---------- design tokens ----------
INK, PAPER, MUTED, LINE = "#14213D", "#F7F8FA", "#5B6475", "#DDE1E8"
RISK = {  # label: (colour, tint, plain-English meaning)
    "High-Risk": ("#C62828", "#FDECEC", "Needs attention before signing"),
    "Needs-Human-Review": ("#6A1B9A", "#F3E8F9", "AI was unsure, have a person check"),
    "Unusual": ("#B26A00", "#FFF3DD", "Differs from typical wording"),
    "Missing": ("#B26A00", "#FFF3DD", "Expected clause not found"),
    "Standard": ("#2E7D5B", "#E7F4EE", "Looks like normal wording"),
}
ORDER = ["High-Risk", "Needs-Human-Review", "Unusual", "Missing", "Standard"]
SOURCE_NAMES = {
    "rule_engine": "Keyword rule",
    "ml_classifier": "ML model",
    "llm": "AI reviewer",
    "llm_rejected": "AI answer rejected",
    "llm_failure": "AI unavailable",
}

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Public Sans', sans-serif; color: {INK}; }}
.stApp {{ background: {PAPER}; }}
.block-container {{ max-width: 1100px; padding-top: 2.2rem; }}
h1, h2, h3 {{ font-family: 'Newsreader', serif !important; color: {INK}; letter-spacing: -0.01em; }}
.hero h1 {{ font-size: 2.6rem; margin: 0; }}
.hero p {{ color: {MUTED}; font-size: 1.05rem; max-width: 60ch; margin: .4rem 0 1.4rem; }}
.stat-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1rem 0 .8rem; }}
.stat {{ background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: 14px 16px; }}
.stat b {{ font-family: 'Newsreader', serif; font-size: 2rem; display: block; line-height: 1.1; }}
.stat span {{ color: {MUTED}; font-size: .85rem; }}
.bar {{ display: flex; height: 10px; border-radius: 6px; overflow: hidden; background: {LINE}; margin-bottom: 1.4rem; }}
.card {{ background: #fff; border: 1px solid {LINE}; border-left-width: 5px; border-radius: 8px;
         padding: 14px 18px; margin-bottom: 10px; }}
.card-top {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
.card-title {{ font-family: 'Newsreader', serif; font-size: 1.25rem; font-weight: 600; }}
.pill {{ font-size: .8rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; }}
.meta {{ color: {MUTED}; font-size: .85rem; margin-top: 4px; }}
.card details {{ margin-top: 8px; }}
.card summary {{ cursor: pointer; color: {MUTED}; font-size: .85rem; }}
.card blockquote {{ margin: 8px 0 0; padding-left: 12px; border-left: 2px solid {LINE};
                    color: #333; font-size: .92rem; line-height: 1.55; max-width: 80ch; }}
@media (max-width: 700px) {{ .stat-row {{ grid-template-columns: repeat(2, 1fr); }} }}
#MainMenu, footer {{ visibility: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)

SAMPLES = {
    "Employment contract": """1. Compensation
The Employee shall be paid a monthly salary of $5,000, payable on the last business day of each month.

2. Termination
The Company may terminate this agreement at its sole discretion, with or without cause, without any notice period.

3. Liability
The Employee shall bear unlimited liability for any damages caused to Company property, whether intentional or accidental.

4. Non-Solicitation
The Employee agrees not to solicit any employees or clients of the Company for a perpetual period following termination.

5. Dispute Resolution
Any disputes arising from this agreement shall be resolved through binding arbitration in New York.""",
    "Rental agreement": """1. Rent Payment
The Tenant shall pay monthly rent of $1,200 on the first day of each month.

2. Security Deposit
A security deposit equal to one month's rent shall be paid prior to occupancy and refunded within 30 days of lease termination, less any deductions for damages.

3. Termination For Convenience
Either party may terminate this lease with 60 days written notice.

4. Governing Law
This lease shall be governed by the laws of the State of Texas.""",
    "NDA": """1. Confidentiality Obligations
Each party agrees to keep all shared information confidential and shall not disclose it to any third party without prior written consent.

2. Non-Compete
The Receiving Party agrees not to engage in any competing business anywhere in the world for a period of 10 years following termination of this agreement.

3. Governing Law
This agreement shall be governed by and construed in accordance with the laws of the State of California.""",
}


@st.cache_resource(show_spinner="Loading models (first run only)...")
def get_app():
    return build_graph()


def risk_of(item):
    return item["risk"]["label"]


def render_stats(items):
    counts = {k: 0 for k in ORDER}
    for it in items:
        counts[risk_of(it)] = counts.get(risk_of(it), 0) + 1
    needs_look = counts["High-Risk"] + counts["Needs-Human-Review"] + counts["Unusual"] + counts["Missing"]
    total = max(len(items), 1)
    st.markdown(
        f"""<div class="stat-row">
<div class="stat"><b>{len(items)}</b><span>clauses reviewed</span></div>
<div class="stat"><b style="color:{RISK['High-Risk'][0]}">{counts['High-Risk']}</b><span>high risk</span></div>
<div class="stat"><b style="color:{RISK['Unusual'][0]}">{needs_look - counts['High-Risk']}</b><span>need a closer look</span></div>
<div class="stat"><b style="color:{RISK['Standard'][0]}">{counts['Standard']}</b><span>standard</span></div>
</div>""",
        unsafe_allow_html=True,
    )
    segs = "".join(
        f'<div style="width:{counts[k] / total * 100}%;background:{RISK[k][0]}"></div>'
        for k in ORDER if counts.get(k)
    )
    st.markdown(f'<div class="bar">{segs}</div>', unsafe_allow_html=True)


def render_cards(items):
    for it in sorted(items, key=lambda x: ORDER.index(risk_of(x)) if risk_of(x) in ORDER else 9):
        label = risk_of(it)
        colour, tint, meaning = RISK.get(label, (MUTED, "#eee", ""))
        risk = it["risk"]
        by = SOURCE_NAMES.get(risk.get("source"), risk.get("source", ""))
        conf = risk.get("confidence")
        conf_txt = f" · confidence {conf:.0%}" if isinstance(conf, float) and conf < 1 else ""
        match = it.get("match")
        match_txt = (
            f"Most similar to a <i>{html.escape(match['matched_clause_type'])}</i> clause "
            f"({match['similarity']:.0%} match)"
            if match else ""
        )
        reasoning = risk.get("reasoning")
        reason_txt = f"<div class='meta'>{html.escape(reasoning)}</div>" if reasoning else ""
        st.markdown(
            f"""<div class="card" style="border-left-color:{colour}">
<div class="card-top">
  <span class="card-title">{html.escape(it['heading'])}</span>
  <span class="pill" style="background:{tint};color:{colour}">{label}</span>
</div>
<div class="meta">{meaning}. Decided by: {by}{conf_txt}</div>
<div class="meta">{match_txt}</div>
{reason_txt}
<details><summary>Show clause text</summary><blockquote>{html.escape(it['clause_text'])}</blockquote></details>
</div>""",
            unsafe_allow_html=True,
        )


# ---------- page ----------
st.markdown(
    """<div class="hero"><h1>ClauseGuard</h1>
<p>Upload a contract and get a first-pass risk review, clause by clause. It flags what looks unusual so you know where to read closely or ask a lawyer.</p></div>""",
    unsafe_allow_html=True,
)

if "text" not in st.session_state:
    st.session_state.text = ""

left, right = st.columns([1, 1], gap="large")
with left:
    uploaded = st.file_uploader("Upload a contract (PDF)", type=["pdf"])
with right:
    choice = st.selectbox("Or try a sample contract", ["Choose one..."] + list(SAMPLES))
    if choice in SAMPLES:
        st.session_state.text = SAMPLES[choice]

pasted = st.text_area("Contract text", key="text", height=200, placeholder="Paste contract text here...")

contract_text = None
if uploaded:
    reader = PdfReader(uploaded)
    contract_text = "\n".join(page.extract_text() or "" for page in reader.pages)
elif pasted.strip():
    contract_text = pasted

if st.button("Review contract", type="primary", disabled=not contract_text):
    with st.spinner("Reading clauses and checking risks..."):
        result = get_app().invoke({"contract_text": contract_text})
    st.session_state.result = result

result = st.session_state.get("result")
if result:
    items = result["flagged"]
    st.markdown("### Results")
    if not items:
        st.warning("No clauses were found. Check that the text has numbered headings like '1. Termination'.")
    else:
        render_stats(items)
        render_cards(items)
        st.download_button("Download report (.md)", result["report"], file_name="clauseguard_report.md")
        with st.expander("Raw agent output (for debugging)"):
            st.json(items)