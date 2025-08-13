# app.py — Free Streamlit Resume Parser + JD Matcher + Google Sheets DB
# ---------------------------------------------------------------------------------
# Features
# - Upload multiple resumes (PDF/DOCX)
# - Paste or upload Job Description (JD)
# - Extract: Name, Email, Phone, Total Experience, Relevant Experience (approx), Position
# - Compute JD Match % (cosine similarity + skills overlap)
# - Download Excel of matching candidates
# - Automatically send NON-matching candidates to a Google Sheet (your team DB)
# - Beautiful, responsive Streamlit UI
# ---------------------------------------------------------------------------------

import io
import re
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import fitz  # PyMuPDF for PDFs
from docx import Document  # for DOCX

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gspread
from google.oauth2.service_account import Credentials

# Optional: spaCy for better name extraction
try:
    import spacy
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False

APP_TITLE = "Resume ↔ JD Matcher (Free)"
ACCENT = "#4f46e5"  # indigo-600

# ------------------------------- UI THEME / CSS ---------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")

st.markdown(
    f"""
    <style>
      :root {{ --accent: {ACCENT}; }}
      .big-title {{
        font-size: 2.0rem; font-weight: 800; margin-bottom: .25rem;
        background: linear-gradient(90deg, var(--accent), #06b6d4); -webkit-background-clip: text; color: transparent;
      }}
      .subtle {{ color: #6b7280; }}
      .card {{
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 18px;
        box-shadow: 0 8px 24px rgba(79,70,229,0.08);
      }}
      .pill {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; background: #EEF2FF; color: var(--accent); font-weight: 600; font-size: .80rem; }}
      .metric {{ font-size: 1.5rem; font-weight: 700; color: #111827; }}
      .muted {{ color: #6b7280; font-size: .85rem; }}
      .footer-note {{ color: #6b7280; text-align:center; margin-top: 14px; }}
      .good {{ color: #059669; font-weight: 700; }}
      .bad {{ color: #dc2626; font-weight: 700; }}
      .btn-primary {{
        background: var(--accent); color: white; padding: 10px 16px; border-radius: 10px; text-decoration: none;
        font-weight: 700; border: none;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='big-title'>📄 {APP_TITLE}</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Upload multiple resumes and a Job Description. Download Excel for matches. Non-matches are auto-saved to your team database (Google Sheets).</div>", unsafe_allow_html=True)

# ---------------------------- GOOGLE SHEETS CLIENT -------------------------------

def get_sheets_client():
    try:
        sa_info = dict(st.secrets["gcp_service_account"])  # section style
        creds = Credentials.from_service_account_info(sa_info, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ])
        client = gspread.authorize(creds)
        sh_name = st.secrets["sheets"]["spreadsheet_name"]
        ws_name = st.secrets["sheets"]["worksheet_name"]
        try:
            sh = client.open(sh_name)
        except gspread.SpreadsheetNotFound:
            sh = client.create(sh_name)
        try:
            ws = sh.worksheet(ws_name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=ws_name, rows=1000, cols=20)
        return ws
    except Exception as e:
        st.warning("Google Sheets not configured yet. Add Streamlit secrets to enable DB storage.")
        return None

# ------------------------------- HELPERS -----------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10,12})")

MONTHS = {
    'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
    'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'sept':9,'september':9,
    'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12
}

YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\+\s*)?(?:years?|yrs)\s*(?:of)?\s*(?:experience|exp)?", re.IGNORECASE)

def safe_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def read_pdf(file) -> str:
    text = []
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)
    except Exception:
        return ""

def read_docx(file) -> str:
    try:
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def extract_email(text: str) -> str:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else ""

def extract_phone(text: str) -> str:
    m = PHONE_RE.search(text)
    if not m:
        return ""
    num = re.sub(r"[^\d+]", "", m.group(0))
    return num

# Optional spaCy-based name extraction with fallbacks
def extract_name(text: str, filename: str) -> str:
    if _SPACY_OK:
        try:
            nlp = _get_spacy()
            doc = nlp(text[:2000])
            people = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON" and 2 <= len(ent.text.strip()) <= 60]
            if people:
                return people[0]
        except Exception:
            pass
    base = re.sub(r"\.[Pp][Dd][FfXx]+$|\.[Dd][Oo][Cc][Xx]$|\.[Dd][Oo][Cc]$", "", filename)
    base = base.replace('_',' ').replace('-',' ')
    words = [w for w in base.split() if w.isalpha() and len(w)>1]
    if 1 <= len(words) <= 4:
        return " ".join(w.capitalize() for w in words)
    first_lines = "\n".join(text.splitlines()[:8])
    cand = re.findall(r"^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}$", first_lines, flags=re.MULTILINE)
    return cand[0] if cand else ""

_spacy_model = None
def _get_spacy():
    global _spacy_model
    if _spacy_model is None:
        try:
            _spacy_model = spacy.load("en_core_web_sm")
        except Exception:
            from spacy.cli import download
            download("en_core_web_sm")
            import spacy as _sp
            _spacy_model = _sp.load("en_core_web_sm")
    return _spacy_model

def estimate_total_experience(text: str) -> float:
    m = YEARS_RE.search(text)
    if m:
        years = float(m.group(1))
        return round(years, 2)
    RANGE = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{2,4})\s*[–\-to]+\s*(present|current|now|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{2,4}))", re.IGNORECASE)
    now = (datetime.now().year, datetime.now().month)
    months = 0
    for m in RANGE.finditer(text):
        sm, sy = m.group(1), m.group(2)
        if m.group(3).lower() in ("present","current","now"):
            ey, em = now
        else:
            em_raw = m.group(4)
            ey_raw = m.group(5)
            MONTHS = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
            em = MONTHS.get(em_raw.lower(), 1)
            ey = int(ey_raw)
            if ey < 100:
                ey += 2000 if ey < 50 else 1900
        sy = int(sy)
        if sy < 100:
            sy += 2000 if sy < 50 else 1900
        MONTHS2 = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        sm = MONTHS2.get(sm.lower(), 1)
        months += max(0, (ey - sy) * 12 + (em - sm))
    return round(months / 12.0, 2) if months else 0.0

def detect_position(text: str, filename: str) -> str:
    for line in text.splitlines()[:25]:
        if re.search(r"(?i)(position|role|applied for|objective|profile)\s*[:|-]", line):
            return safe_text(re.sub(r"(?i).{0,20}(position|role|applied for|objective|profile)\s*[:|-]", "", line))
    base = re.sub(r"\.[Pp][Dd][FfXx]+$|\.[Dd][Oo][Cc][Xx]$|\.[Dd][Oo][Cc]$", "", filename)
    words = base.replace('_',' ').replace('-',' ').split()
    words = [w for w in words if w.lower() not in ("resume","cv")]
    if len(words) >= 2:
        return " ".join(words[1:4])
    return ""

def jd_skills(jd_text: str) -> List[str]:
    raw = [s.strip().lower() for s in re.split(r"[,/|\n]", jd_text) if s.strip()]
    skills = [re.sub(r"[^a-z0-9+#. ]", "", s) for s in raw]
    uniq = []
    for s in skills:
        s2 = s.strip()
        if s2 and s2 not in uniq:
            uniq.append(s2)
    return uniq[:150]

def relevant_experience_approx(total_exp_years: float, resume_text: str, skills: List[str]) -> float:
    if not skills or not total_exp_years:
        return 0.0
    text_lower = resume_text.lower()
    hits = 0
    for sk in skills:
        if len(sk) < 2:
            continue
        if re.search(rf"\b{re.escape(sk)}\b", text_lower):
            hits += 1
    ratio = hits / max(1, len(skills))
    return round(total_exp_years * min(1.0, ratio * 1.5), 2)

def match_score(resume_text: str, jd_text: str, skills: List[str]) -> float:
    docs = [resume_text, jd_text]
    tfv = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=15000)
    tfm = tfv.fit_transform(docs)
    cos = cosine_similarity(tfm[0:1], tfm[1:2])[0][0]
    text_lower = resume_text.lower()
    if skills:
        sk_hits = sum(1 for s in skills if re.search(rf"\b{re.escape(s)}\b", text_lower))
        sk_ratio = sk_hits / len(skills)
    else:
        sk_ratio = 0.0
    score = (cos * 0.7) + (sk_ratio * 0.3)
    return round(float(score) * 100, 2)

def to_excel_download(df: pd.DataFrame) -> bytes:
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Matches')
        return buffer.getvalue()

def append_non_matches_to_sheet(ws, rows: List[Dict]):
    if not ws or not rows:
        return False
    headers = ["Timestamp", "Name", "Email", "Phone", "Total Experience (yrs)", "Relevant Experience (yrs)", "Position", "Match %", "Source File"]
    try:
        existing = ws.get_all_values()
        if not existing:
            ws.append_row(headers)
        elif existing and existing[0] != headers:
            ws.insert_row(headers, index=1)
        for r in rows:
            ws.append_row([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                r.get('Name',''), r.get('Email',''), r.get('Phone',''),
                r.get('Total Experience (yrs)', ''), r.get('Relevant Experience (yrs)', ''),
                r.get('Position',''), r.get('Match %',''), r.get('Source File','')
            ])
        return True
    except Exception as e:
        st.warning(f"Could not write to Google Sheet: {e}")
        return False

# ------------------------------- SIDEBAR -----------------------------------------
with st.sidebar:
    st.markdown("<div class='pill'>Settings</div>", unsafe_allow_html=True)
    threshold = st.slider("Match Threshold (%) — candidates above go to Excel; below go to DB", 0, 100, 60, 1)
    st.caption("Tip: Start around 60–70%. Adjust as per role.")
    st.divider()
    st.markdown("<div class='pill'>Files</div>", unsafe_allow_html=True)
    files = st.file_uploader("Upload multiple resumes (PDF/DOCX)", type=["pdf","docx"], accept_multiple_files=True)
    jd_mode = st.radio("Job Description Input", ["Paste JD Text","Upload JD (txt)"], horizontal=True)
    jd_text = ""
    if jd_mode == "Paste JD Text":
        jd_text = st.text_area("Paste Job Description", height=200, placeholder="Paste the role responsibilities, must-have skills, years of experience, location, etc.")
    else:
        jd_file = st.file_uploader("Upload JD as .txt", type=["txt"], accept_multiple_files=False)
        if jd_file is not None:
            jd_text = jd_file.read().decode(errors='ignore')

# ------------------------------- MAIN ACTION -------------------------------------
col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.markdown("<div class='card'>" 
                "<div class='metric'>Ready to Process</div>"
                "<div class='muted'>Upload resumes and enter a JD, then click RUN.</div>"
                "</div>", unsafe_allow_html=True)
with col2:
    st.metric("Threshold", f"{threshold}%")
with col3:
    st.markdown("<span class='muted'>Non-matches will be saved to Google Sheets automatically.</span>", unsafe_allow_html=True)

run = st.button("🚀 Run Matching", type="primary")

if run:
    if not files or not jd_text.strip():
        st.error("Please upload at least one resume and provide a Job Description.")
        st.stop()

    ws = get_sheets_client()
    skills = jd_skills(jd_text)

    rows = []
    non_matches = []

    progress = st.progress(0)
    status_area = st.empty()

    for idx, f in enumerate(files, start=1):
        status_area.info(f"Parsing {f.name} ({idx}/{len(files)}) ...")
        text = ""
        if f.type in ("application/pdf",) or f.name.lower().endswith(".pdf"):
            text = read_pdf(f)
        elif f.name.lower().endswith(".docx"):
            text = read_docx(f)
        else:
            text = ""
        if not text:
            status_area.warning(f"Could not read {f.name}. Skipping.")
            progress.progress(min(100, int(idx/len(files)*100)))
            continue

        name = extract_name(text, f.name)
        email = extract_email(text)
        phone = extract_phone(text)
        total_exp = estimate_total_experience(text)
        position = detect_position(text, f.name)
        rel_exp = relevant_experience_approx(total_exp, text, skills)
        score = match_score(text, jd_text, skills)

        row = {
            "Name": name,
            "Email": email,
            "Phone": phone,
            "Total Experience (yrs)": total_exp,
            "Relevant Experience (yrs)": rel_exp,
            "Position": position,
            "Match %": score,
            "Source File": f.name,
        }
        rows.append(row)
        if score < threshold:
            non_matches.append(row)

        progress.progress(min(100, int(idx/len(files)*100)))
        time.sleep(0.05)

    status_area.success("Processing complete.")

    if rows:
        df = pd.DataFrame(rows)
        st.markdown("### 📊 Results")
        st.dataframe(
            df.style.hide(axis='index').bar(subset=["Match %"], vmin=0, vmax=100),
            use_container_width=True,
        )

        matches_df = df[df["Match %"] >= threshold].sort_values(by="Match %", ascending=False)
        st.markdown("### ⬇️ Download Matching Candidates")
        if not matches_df.empty:
            excel_bytes = to_excel_download(matches_df)
            st.download_button(
                label="Download Excel (Matches)",
                data=excel_bytes,
                file_name="matching_candidates.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("No candidates met the threshold. Adjust the threshold or review non-matches in the DB.")

        if non_matches:
            ok = append_non_matches_to_sheet(ws, non_matches)
            if ok:
                st.success(f"Saved {len(non_matches)} non-matching candidates to Google Sheets database.")
            else:
                st.warning("Non-matching candidates were NOT saved. Configure Google Sheets secrets and share the sheet with the service account.")

    else:
        st.warning("No readable resumes processed.")

# ------------------------------- FOOTER ------------------------------------------
st.markdown("<div class='footer-note'>Built for Talent Acquisition teams • Free, private, and shareable • Streamlit Cloud ready</div>", unsafe_allow_html=True)
