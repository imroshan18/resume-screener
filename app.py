import streamlit as st
import pickle
import re
import numpy as np
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Resume Screener Pro", layout="wide")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    mlp = pickle.load(open("mlp_model.pkl", "rb"))
    return tfidf, le, scaler, mlp

tfidf, le, scaler, mlp = load_models()

# -----------------------------
# ROLE DEFINITIONS
# -----------------------------
ROLE_SKILLS = {
    "Machine Learning Engineer": ["python","machine learning","numpy","pandas","scikit-learn"],
    "AI Engineer": ["deep learning","tensorflow","nlp"],
    "Data Analyst": ["sql","excel","power bi","pandas","visualization"]
}

# -----------------------------
# HELPERS
# -----------------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', text).lower()

def extract_name(text):
    return text.split("\n")[0].strip()

def extract_experience(text):
    matches = re.findall(r'(\d+)\s+years?', text.lower())
    return max([int(m) for m in matches], default=0)

def extract_skills(text):
    skills = [
        "python","java","sql","machine learning","deep learning",
        "tensorflow","pandas","numpy","excel","power bi",
        "scikit-learn","matplotlib","seaborn","plotly","mongodb","azure","nlp"
    ]
    return list(set([s for s in skills if s in text.lower()]))

# -----------------------------
# ROLE MATCHING
# -----------------------------
def check_role_fit(skills):
    results = {}
    for role, req in ROLE_SKILLS.items():
        matched = [s for s in req if s in skills]
        score = len(matched) / len(req)
        results[role] = {
            "score": score,
            "matched": matched,
            "missing": list(set(req) - set(matched))
        }
    return results

# -----------------------------
# ATS SCORE
# -----------------------------
def calculate_ats_score(skills, exp):
    score = len(skills) * 4 + min(exp * 5, 30)
    return min(score, 100)

# -----------------------------
# JOB MATCH
# -----------------------------
def match_job(resume_text, job_desc):
    v1 = tfidf.transform([resume_text])
    v2 = tfidf.transform([job_desc])
    return cosine_similarity(v1, v2)[0][0]

# -----------------------------
# AI FEEDBACK
# -----------------------------
def generate_ai_feedback(skills, exp, ats):
    feedback = []

    if ats < 50:
        feedback.append("Your resume is weak. Add more relevant technical skills.")
    elif ats < 75:
        feedback.append("Your resume is decent but can be improved.")
    else:
        feedback.append("Strong resume! You are job-ready.")

    if exp == 0:
        feedback.append("Add internships or real-world projects.")

    if "python" not in skills:
        feedback.append("Learn Python (very important).")

    if "machine learning" not in skills:
        feedback.append("Add Machine Learning skills.")

    return feedback

# -----------------------------
# PDF GENERATION
# -----------------------------
def generate_pdf(name, skills, ats):
    file = "report.pdf"
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph(f"Name: {name}", styles["Normal"]))
    content.append(Paragraph(f"ATS Score: {ats}", styles["Normal"]))
    content.append(Paragraph(f"Skills: {', '.join(skills)}", styles["Normal"]))

    doc.build(content)
    return file

# -----------------------------
# UI
# -----------------------------
st.title("🚀 AI Resume Screener Pro")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description (Optional)")

if uploaded_file:
    with st.spinner("Analyzing Resume..."):
        text = extract_text(uploaded_file)
        cleaned = clean_text(text)

    skills = extract_skills(text)
    exp = extract_experience(text)
    name = extract_name(text)

    # -----------------------------
    # ML Prediction
    # -----------------------------
    tfidf_vec = tfidf.transform([cleaned]).toarray()
    features = np.hstack((tfidf_vec, [[len(cleaned.split()), exp, len(skills)]]))
    features = scaler.transform(features)

    probs = mlp.predict_proba(features)[0]
    top3_idx = probs.argsort()[-3:][::-1]

    # -----------------------------
    # ROLE ANALYSIS
    # -----------------------------
    role_results = check_role_fit(skills)
    ats_score = calculate_ats_score(skills, exp)

    if job_desc:
        job_score = match_job(cleaned, clean_text(job_desc))
    else:
        job_score = None

    # -----------------------------
    # DISPLAY
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Resume Preview")
        st.text_area("", text[:1000], height=400)

    with col2:
        st.subheader("🎯 Top Predictions")
        for i in top3_idx:
            role = le.inverse_transform([i])[0]
            st.write(f"{role} ({probs[i]:.2%})")
            st.progress(float(probs[i]))

        st.subheader("📊 ATS Score")
        st.progress(ats_score/100)
        st.write(f"{ats_score}%")

        if job_score:
            st.subheader("💼 Job Match")
            st.progress(job_score)
            st.write(f"{job_score*100:.2f}% match")

    # -----------------------------
    # ROLE FIT
    # -----------------------------
    st.subheader("🧠 Role Eligibility")
    for role, data in role_results.items():
        st.write(f"### {role}")
        st.progress(data["score"])
        st.write("✅ Matched:", ", ".join(data["matched"]))
        st.write("❌ Missing:", ", ".join(data["missing"]))

    # -----------------------------
    # AI FEEDBACK
    # -----------------------------
    st.subheader("🤖 AI Feedback")
    feedback = generate_ai_feedback(skills, exp, ats_score)
    for f in feedback:
        st.write("👉", f)

    # -----------------------------
    # SKILL CHART
    # -----------------------------
    st.subheader("📊 Skills Chart")
    if skills:
        fig, ax = plt.subplots()
        ax.barh(skills, [1]*len(skills))
        st.pyplot(fig)

    # -----------------------------
    # DOWNLOAD REPORT
    # -----------------------------
    st.subheader("📄 Download Report")
    if st.button("Generate PDF"):
        file = generate_pdf(name, skills, ats_score)
        with open(file, "rb") as f:
            st.download_button("Download", f, file_name="report.pdf")

    # -----------------------------
    # JSON OUTPUT
    # -----------------------------
    st.subheader("📦 JSON Output")
    st.json({
        "name": name,
        "skills": skills,
        "experience": exp,
        "ats_score": ats_score
    })
