import streamlit as st
import torch
import pickle
import re
from pdfminer.high_level import extract_text
from transformers import pipeline
import numpy as np

# -----------------------------
# Load models
# -----------------------------
class MLP(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Load files
tfidf = pickle.load(open("tfidf.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

model = MLP(tfidf.transform(["test"]).shape[1] + 3, len(le.classes_))
model.load_state_dict(torch.load("mlp_model.pth", map_location="cpu"))
model.eval()

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# -----------------------------
# Helper functions
# -----------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

def extract_experience(text):
    matches = re.findall(r'(\d+)\s+years?', text.lower())
    valid = [int(m) for m in matches if 0 < int(m) <= 40]
    return max(valid) if valid else 0

def count_skills(text):
    skills_list = [
        "python","java","sql","machine learning","deep learning",
        "tensorflow","pandas","numpy","excel","power bi",
        "scikit-learn","matplotlib","seaborn","plotly","mongodb","azure"
    ]
    return sum([1 for s in skills_list if s in text.lower()])

def extract_name(text):
    return text.split("\n")[0]

def extract_skills(text):
    skills_list = [
        "python","java","sql","machine learning","deep learning",
        "tensorflow","pandas","numpy","excel","power bi",
        "scikit-learn","matplotlib","seaborn","plotly","mongodb","azure"
    ]
    return [s for s in skills_list if s in text.lower()]

def extract_roles(text):
    roles = []
    for word in text.lower().split():
        if any(r in word for r in ["intern","analyst","engineer","developer"]):
            roles.append(word.replace(",", ""))
    return list(set(roles))


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Intelligent Resume Screener")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    
    # Extract text
    text = extract_text(uploaded_file)

    st.subheader("📄 Resume Preview")
    st.write(text[:500])

    # -----------------------------
    # Classification
    # -----------------------------
    cleaned = clean_text(text)
    word_count = len(cleaned.split())
    experience = extract_experience(text)
    skill_count = count_skills(text)

    tfidf_vec = tfidf.transform([cleaned]).toarray()
    features = np.hstack((tfidf_vec, [[word_count, experience, skill_count]]))
    features = scaler.transform(features)

    tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()

    category = le.inverse_transform([pred])[0]

    st.subheader("🎯 Predicted Job Category")
    st.success(category)

    # -----------------------------
    # NER + Rules
    # -----------------------------
    name = extract_name(text)
    skills = extract_skills(text)
    roles = extract_roles(text)

    entities = ner(text)
    orgs = list(set([e['word'] for e in entities if e['entity_group'] == 'ORG']))

    # clean orgs
    orgs = [o for o in orgs if len(o) > 3]

    result = {
        "name": name,
        "organizations": orgs,
        "skills": skills,
        "roles": roles
    }

    st.subheader("📊 Extracted Information")
    st.json(result)
