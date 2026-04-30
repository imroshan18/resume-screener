import streamlit as st
import pickle
import re
import numpy as np
from pdfminer.high_level import extract_text

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model = pickle.load(open("mlp_model.pkl", "rb"))
    return tfidf, le, scaler, model

tfidf, le, scaler, model = load_models()

# -----------------------------
# CATEGORY → ROLE MAPPING
# -----------------------------
CATEGORY_ROLES = {
    "INFORMATION-TECHNOLOGY": ["Software Engineer", "Data Analyst", "ML Engineer"],
    "ENGINEERING": ["Engineer", "Project Engineer"],
    "HEALTHCARE": ["Doctor", "Nurse", "Healthcare Assistant"],
    "FINANCE": ["Financial Analyst", "Accountant"],
    "ACCOUNTANT": ["Accountant", "Auditor"],
    "BANKING": ["Bank Officer", "Relationship Manager"],
    "SALES": ["Sales Executive", "Sales Manager"],
    "BUSINESS-DEVELOPMENT": ["Business Development Executive"],
    "HR": ["HR Manager", "Recruiter"],
    "TEACHER": ["Teacher", "Trainer"],
    "PUBLIC-RELATIONS": ["PR Executive"],
    "DIGITAL-MEDIA": ["Digital Marketer", "Content Creator"],
    "DESIGNER": ["Graphic Designer", "UI/UX Designer"],
    "CONSTRUCTION": ["Civil Engineer"],
    "CONSULTANT": ["Business Consultant"],
    "CHEF": ["Chef"],
    "BPO": ["Customer Support"],
    "AUTOMOBILE": ["Automobile Engineer"],
    "AVIATION": ["Pilot", "Ground Staff"],
    "APPAREL": ["Fashion Designer"],
    "ARTS": ["Artist"],
    "ADVOCATE": ["Lawyer"],
    "AGRICULTURE": ["Agriculture Officer"]
}

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', text).lower()

# -----------------------------
# ATS SCORE (MODEL BASED)
# -----------------------------
def calculate_ats(probs):
    return int(max(probs) * 100)

# -----------------------------
# FEEDBACK SYSTEM
# -----------------------------
def generate_feedback(category, ats):
    feedback = []

    if ats < 50:
        feedback.append("Your resume is not strongly aligned with this domain.")
    elif ats < 75:
        feedback.append("Your resume is moderately aligned but can improve.")
    else:
        feedback.append("Your resume is well aligned with this domain.")

    # domain-specific suggestions
    if category == "INFORMATION-TECHNOLOGY":
        feedback.append("Add projects, GitHub links, and technical skills.")

    elif category == "SALES":
        feedback.append("Highlight achievements, targets, and communication skills.")

    elif category == "HEALTHCARE":
        feedback.append("Add certifications, patient care experience.")

    elif category == "FINANCE":
        feedback.append("Include financial tools, analysis experience.")

    return feedback

# -----------------------------
# UI
# -----------------------------
st.title("🚀 AI Resume Screener")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:

    with st.spinner("Analyzing Resume..."):
        text = extract_text(uploaded_file)
        cleaned = clean_text(text)

    # -----------------------------
    # MODEL PREDICTION
    # -----------------------------
    tfidf_vec = tfidf.transform([cleaned]).toarray()

    features = np.hstack((tfidf_vec, [[len(cleaned.split()), 0, 0]]))
    features = scaler.transform(features)

    probs = model.predict_proba(features)[0]

    top_idx = probs.argsort()[-3:][::-1]

    predicted_category = le.inverse_transform([probs.argmax()])[0]
    ats_score = calculate_ats(probs)

    roles = CATEGORY_ROLES.get(predicted_category, ["General Role"])

    # -----------------------------
    # DISPLAY
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Resume Preview")
        st.text_area("", text[:1000], height=400)

    with col2:
        st.subheader("🎯 Predicted Domain")
        st.success(predicted_category)

        st.subheader("💼 Recommended Roles")
        for r in roles:
            st.write("👉", r)

        st.subheader("📊 ATS Score")
        st.progress(ats_score / 100)
        st.write(f"{ats_score}%")

        st.subheader("🔝 Top Predictions")
        for i in top_idx:
            cat = le.inverse_transform([i])[0]
            st.write(f"{cat} ({probs[i]:.2%})")

    # -----------------------------
    # FEEDBACK
    # -----------------------------
    st.subheader("🤖 Feedback")

    feedback = generate_feedback(predicted_category, ats_score)

    for f in feedback:
        st.write("👉", f)

    # -----------------------------
    # JSON OUTPUT
    # -----------------------------
    st.subheader("📦 Structured Output")

    st.json({
        "predicted_category": predicted_category,
        "recommended_roles": roles,
        "ats_score": ats_score
    })
