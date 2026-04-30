import streamlit as st
import pickle
import re
import numpy as np
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity

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
# ROLE SKILLS (Decision Engine)
# -----------------------------
ROLE_SKILLS = {
    "Machine Learning Engineer": ["python","machine learning","numpy","pandas","scikit-learn"],
    "AI Engineer": ["deep learning","tensorflow","nlp"],
    "Data Analyst": ["sql","excel","power bi","pandas"]
}

# -----------------------------
# HELPERS
# -----------------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', text).lower()

def extract_name(text):
    return text.split("\n")[0].strip()

def extract_skills(text):
    skills = [
        "python","java","sql","machine learning","deep learning",
        "tensorflow","pandas","numpy","excel","power bi",
        "scikit-learn","matplotlib","plotly","mongodb","azure","nlp"
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
# ATS (MODEL BASED)
# -----------------------------
def calculate_ats(probs):
    return int(max(probs) * 100)

# -----------------------------
# FEEDBACK (DOMAIN-AWARE)
# -----------------------------
def generate_feedback(best_role, missing, ats):
    feedback = []

    if ats < 50:
        feedback.append("Your resume is not aligned with the target domain.")
    elif ats < 75:
        feedback.append("Your resume is moderately aligned but can be improved.")
    else:
        feedback.append("Strong resume for the selected domain.")

    if missing:
        feedback.append(f"For {best_role}, improve: {', '.join(missing)}")
    else:
        feedback.append(f"You are well-prepared for {best_role} 🚀")

    return feedback

# -----------------------------
# UI
# -----------------------------
st.title("🚀 AI Resume Screener Pro")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description (Optional)")

if uploaded_file:

    with st.spinner("Analyzing resume..."):
        text = extract_text(uploaded_file)
        cleaned = clean_text(text)

    name = extract_name(text)
    skills = extract_skills(text)

    # -----------------------------
    # MODEL PREDICTION
    # -----------------------------
    tfidf_vec = tfidf.transform([cleaned]).toarray()
    features = np.hstack((tfidf_vec, [[len(cleaned.split()), len(skills), 0]]))
    features = scaler.transform(features)

    probs = mlp.predict_proba(features)[0]
    top_idx = probs.argsort()[-3:][::-1]

    ats_score = calculate_ats(probs)

    # -----------------------------
    # ROLE ENGINE
    # -----------------------------
    role_results = check_role_fit(skills)
    best_role = max(role_results, key=lambda x: role_results[x]["score"])
    best_data = role_results[best_role]

    # -----------------------------
    # JOB MATCH
    # -----------------------------
    if job_desc:
        job_score = cosine_similarity(
            tfidf.transform([cleaned]),
            tfidf.transform([clean_text(job_desc)])
        )[0][0]
    else:
        job_score = None

    # -----------------------------
    # UI DISPLAY
    # -----------------------------
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("📄 Resume Preview")
        st.text_area("", text[:1000], height=400)

    with col2:
        st.subheader("🎯 Best Role Recommendation")

        st.success(f"{best_role}")

        st.write("**Confidence (ATS):**")
        st.progress(ats_score/100)
        st.write(f"{ats_score}%")

        st.write("**Top ML Predictions:**")
        for i in top_idx:
            role = le.inverse_transform([i])[0]
            st.write(f"- {role} ({probs[i]:.2%})")

        if job_score:
            st.subheader("💼 Job Match Score")
            st.progress(job_score)
            st.write(f"{job_score*100:.2f}% match")

    # -----------------------------
    # ROLE DETAILS
    # -----------------------------
    st.subheader("🧠 Role Analysis")

    st.write("✅ Matched Skills:", ", ".join(best_data["matched"]))
    st.write("❌ Missing Skills:", ", ".join(best_data["missing"]))

    # -----------------------------
    # FEEDBACK
    # -----------------------------
    st.subheader("🤖 Smart Feedback")

    feedback = generate_feedback(best_role, best_data["missing"], ats_score)

    for f in feedback:
        st.write("👉", f)

    # -----------------------------
    # JSON OUTPUT
    # -----------------------------
    st.subheader("📦 Structured Output")

    st.json({
        "name": name,
        "best_role": best_role,
        "ats_score": ats_score,
        "skills": skills,
        "missing_skills": best_data["missing"]
    })
