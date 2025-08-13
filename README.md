# Resume ↔ JD Matcher (Free, Team‑Ready)

A free Streamlit web app for Talent Acquisition teams:
- Upload multiple resumes (PDF/DOCX)
- Paste/upload a JD
- Extract candidate details (Name, Email, Phone, Total & Relevant Experience, Position)
- Compute JD Match % (TF‑IDF + skill overlap)
- Download Excel for matching candidates
- Auto‑store non‑matches in Google Sheets as your shared database

## Quick Start (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Free on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io/ and deploy the repo.
3. In **App Settings → Secrets**, paste the contents of `.streamlit/secrets.toml` (see below).
4. Create a Google Sheet named as in `secrets.toml` and **share** it with the service account email (Editor).

## Google Sheets Setup
- Create/Use a GCP project → enable **Google Sheets API** and **Google Drive API**.
- Create a **Service Account**, then generate a **JSON key**.
- Share your Google Sheet with the **service account email**.

## How Matching Works
- The app builds TF‑IDF vectors of resume vs JD and computes cosine similarity.
- It also checks JD skills presence in the resume text.
- Final score = 70% cosine + 30% skill‑overlap ratio.

## Notes
- Non‑matches auto‑append to Google Sheets with timestamp and source filename.
- Tweak the threshold from the sidebar (default 60%).
- This is heuristic and lightweight by design (fast & free). You can plug in advanced models later if needed.
