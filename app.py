import streamlit as st
import pickle
import re
import numpy as np
from pdfminer.high_level import extract_text
from transformers import pipeline

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    mlp = pickle.load(open("mlp_model.pkl", "rb"))

    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

    return tfidf, le, scaler, mlp, ner


tfidf, le, scaler, mlp, ner = load_models()

# -----------------------------
# Helper Functions
# -----------------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', text).lower()

def extract_experience(text):
    matches = re.findall(r'(\d+)\s+years?', text.lower())
    valid = [int(m) for m in matches if 0 < int(m) <= 40]
    return max(valid) if valid else 0

def extract_name(text):
    return text.split("\n")[0]

def extract_skills(text):
    skills = [
        "python","java","sql","machine learning","deep learning",
        "tensorflow","pandas","numpy","excel","power bi",
        "scikit-learn","matplotlib","seaborn","plotly","mongodb","azure"
    ]
    return [s for s in skills if s in text.lower()]

def extract_roles(text):
    roles = []
    for word in text.lower().split():
        if any(r in word for r in ["intern","analyst","engineer","developer"]):
            roles.append(word.replace(",", ""))
    return list(set(roles))

# -----------------------------
# UI Header
# -----------------------------
st.title("📄 AI-Powered Resume Screener")
st.markdown("Upload a resume to classify job role and extract structured information.")

st.divider()

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:

    with st.spinner("Processing resume..."):
        text = extract_text(uploaded_file)

    col1, col2 = st.columns([1, 1])

    # -----------------------------
    # LEFT SIDE → Resume Preview
    # -----------------------------
    with col1:
        st.subheader("📄 Resume Preview")
        st.text_area("", text[:1000], height=400)

    # -----------------------------
    # RIGHT SIDE → Results
    # -----------------------------
    with col2:

        # -------- Classification --------
        cleaned = clean_text(text)
        word_count = len(cleaned.split())
        experience = extract_experience(text)
        skill_count = len(extract_skills(text))

        tfidf_vec = tfidf.transform([cleaned]).toarray()
        features = np.hstack((tfidf_vec, [[word_count, experience, skill_count]]))
        features = scaler.transform(features)

        pred = mlp.predict(features)[0]
        probs = mlp.predict_proba(features)

        category = le.inverse_transform([pred])[0]
        confidence = max(probs[0])

        st.subheader("🎯 Predicted Role")
        st.success(f"{category} ({confidence:.2%} confidence)")

        # -------- NER + Rules --------
        name = extract_name(text)
        skills = extract_skills(text)
        roles = extract_roles(text)

        entities = ner(text)
        orgs = list(set([e['word'] for e in entities if e['entity_group'] == 'ORG']))

        # Clean orgs
        orgs = [o for o in orgs if len(o) > 3 and o != "t. Ltd"]

        st.subheader("📊 Extracted Information")

        st.markdown(f"**👤 Name:** {name}")
        st.markdown(f"**🏢 Organizations:** {', '.join(orgs)}")
        st.markdown(f"**🧠 Skills:** {', '.join(skills)}")
        st.markdown(f"**💼 Roles:** {', '.join(roles)}")

    st.divider()

    # -----------------------------
    # JSON Output
    # -----------------------------
    st.subheader("📦 Structured JSON Output")

    result = {
        "name": name,
        "organizations": orgs,
        "skills": skills,
        "roles": roles
    }

    st.json(result)
