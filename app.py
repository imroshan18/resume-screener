import streamlit as st
import pickle
import re
import numpy as np
from pdfminer.high_level import extract_text

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

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
# HELPERS
# -----------------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', text).lower()

def extract_experience(text):
    matches = re.findall(r'(\d+)\s+years?', text.lower())
    valid = [int(m) for m in matches if 0 < int(m) <= 40]
    return max(valid) if valid else 0

def extract_name(text):
    return text.split("\n")[0].strip()

def extract_skills(text):
    skills = [
        "python","java","sql","machine learning","deep learning",
        "tensorflow","pandas","numpy","excel","power bi",
        "scikit-learn","matplotlib","seaborn","plotly","mongodb","azure"
    ]
    return sorted(list(set([s for s in skills if s in text.lower()])))

def extract_roles(text):
    roles = []
    for word in text.lower().split():
        if any(r in word for r in ["intern","analyst","engineer","developer"]):
            roles.append(word.replace(",", ""))
    return list(set(roles))

def extract_organizations(text):
    org_keywords = ["pvt", "ltd", "technologies", "company", "solutions"]

    orgs = []
    lines = text.split("\n")

    for line in lines:
        line_lower = line.lower()
        if any(k in line_lower for k in org_keywords):
            if len(line.split()) < 10:  # avoid long noisy lines
                orgs.append(line.strip())

    return list(set(orgs))


# -----------------------------
# UI HEADER
# -----------------------------
st.title("📄 AI Resume Screener")
st.markdown("Upload your resume to classify job role and extract structured insights.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_file:
    with st.spinner("🔍 Analyzing resume..."):
        text = extract_text(uploaded_file)

    col1, col2 = st.columns([1, 1])

    # -----------------------------
    # LEFT SIDE
    # -----------------------------
    with col1:
        st.subheader("📄 Resume Preview")
        st.text_area("", text[:1000], height=400)

    # -----------------------------
    # RIGHT SIDE
    # -----------------------------
    with col2:

        # ---- Feature Engineering ----
        cleaned = clean_text(text)
        word_count = len(cleaned.split())
        experience = extract_experience(text)
        skill_list = extract_skills(text)
        skill_count = len(skill_list)

        tfidf_vec = tfidf.transform([cleaned]).toarray()
        features = np.hstack((tfidf_vec, [[word_count, experience, skill_count]]))
        features = scaler.transform(features)

        # ---- Prediction ----
        probs = mlp.predict_proba(features)[0]
        top3_idx = probs.argsort()[-3:][::-1]

        st.subheader("🎯 Top Matching Roles")
        for i in top3_idx:
            role = le.inverse_transform([i])[0]
            score = probs[i]
            st.progress(float(score))
            st.write(f"👉 **{role}** ({score:.2%})")

        # ---- Extraction ----
        name = extract_name(text)
        roles = extract_roles(text)
        orgs = extract_organizations(text)

        # Remove skill contamination
        orgs = [o for o in orgs if not any(s in o.lower() for s in skill_list)]

        st.subheader("📊 Extracted Information")

        st.markdown(f"**👤 Name:** {name}")

        st.markdown("**🏢 Organizations:**")
        if orgs:
            for org in orgs:
                st.write(f"• {org}")
        else:
            st.write("Not detected")

        st.markdown("**🧠 Skills:**")
        if skill_list:
            st.write(", ".join(skill_list))
        else:
            st.write("Not detected")

        st.markdown("**💼 Roles:**")
        if roles:
            st.write(", ".join(roles))
        else:
            st.write("Not detected")

    # -----------------------------
    # JSON OUTPUT
    # -----------------------------
    st.subheader("📦 Structured JSON Output")

    result = {
        "name": name,
        "organizations": orgs,
        "skills": skill_list,
        "roles": roles
    }

    st.json(result)
