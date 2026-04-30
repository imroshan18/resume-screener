import streamlit as st
import pickle
import re
import numpy as np
import os
import io

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Load MLP Model (PyTorch .pth)
# -----------------------------
@st.cache_resource
def load_models():
    """
    Load all model artifacts safely.
    - tfidf.pkl, scaler.pkl, label_encoder.pkl  → sklearn pickle files
    - mlp_model.pth                              → PyTorch state dict
    """
    errors = []

    # --- Load sklearn pickles ---
    def load_pkl(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    try:
        tfidf = load_pkl("tfidf.pkl")
    except Exception as e:
        errors.append(f"tfidf.pkl: {e}")
        tfidf = None

    try:
        le = load_pkl("label_encoder.pkl")
    except Exception as e:
        errors.append(f"label_encoder.pkl: {e}")
        le = None

    try:
        scaler = load_pkl("scaler.pkl")
    except Exception as e:
        errors.append(f"scaler.pkl: {e}")
        scaler = None

    # --- Load MLP: supports BOTH .pth (PyTorch) and .pkl (sklearn) ---
    mlp = None

    # Try PyTorch .pth first
    if os.path.exists("mlp_model.pth"):
        try:
            import torch
            import torch.nn as nn

            # We need to know input size from tfidf + extra features (3)
            if tfidf is not None:
                input_size = len(tfidf.get_feature_names_out()) + 3
            else:
                input_size = 1003  # fallback: 1000 tfidf + 3 manual features

            # Dynamically detect architecture from state dict
            checkpoint = torch.load(
                "mlp_model.pth",
                map_location=torch.device("cpu"),
                weights_only=False   # needed for full model objects
            )

            # Case 1: full model object was saved
            if isinstance(checkpoint, nn.Module):
                mlp = checkpoint
                mlp.eval()

            # Case 2: state dict was saved
            elif isinstance(checkpoint, dict) and any(
                k.startswith(("layers", "fc", "linear", "0.", "weight"))
                for k in checkpoint.keys()
            ):
                # Infer layer sizes from state dict
                layer_sizes = _infer_layer_sizes(checkpoint, input_size)
                mlp = _build_mlp(layer_sizes)
                mlp.load_state_dict(checkpoint)
                mlp.eval()

            # Case 3: checkpoint dict with 'model_state_dict' key
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
                layer_sizes = _infer_layer_sizes(state, input_size)
                mlp = _build_mlp(layer_sizes)
                mlp.load_state_dict(state)
                mlp.eval()

            else:
                errors.append(
                    "mlp_model.pth: Unrecognised checkpoint format. "
                    f"Keys found: {list(checkpoint.keys())[:5]}"
                )

        except ImportError:
            errors.append(
                "mlp_model.pth: PyTorch not installed. "
                "Add 'torch' to requirements.txt"
            )
        except Exception as e:
            errors.append(f"mlp_model.pth (PyTorch): {e}")

    # Fallback: try sklearn pickle named mlp_model.pkl
    if mlp is None and os.path.exists("mlp_model.pkl"):
        try:
            mlp = load_pkl("mlp_model.pkl")
        except Exception as e:
            errors.append(f"mlp_model.pkl (sklearn): {e}")

    if errors:
        st.error("⚠️ Model loading issues:\n\n" + "\n\n".join(errors))

    return tfidf, le, scaler, mlp


def _infer_layer_sizes(state_dict, input_size):
    """
    Walk the state dict to reconstruct [input, hidden..., output] sizes.
    Works for models saved with sequential integer keys (0, 2, 4…)
    or named keys (fc1, fc2…).
    """
    sizes = [input_size]
    # Collect all weight tensors in order
    weights = [
        (k, v) for k, v in state_dict.items()
        if "weight" in k and v.ndim == 2
    ]
    for _, w in weights:
        sizes.append(w.shape[0])   # out_features of that layer
    return sizes


def _build_mlp(layer_sizes):
    """Build a ReLU MLP matching the inferred layer sizes."""
    import torch.nn as nn
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ---- Prediction helpers for PyTorch vs sklearn ----
def _is_torch_model(mlp):
    try:
        import torch.nn as nn
        return isinstance(mlp, nn.Module)
    except ImportError:
        return False


def _predict(mlp, features_np):
    """Returns (predicted_label_index, probabilities_array)."""
    if _is_torch_model(mlp):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            x = torch.tensor(features_np, dtype=torch.float32)
            logits = mlp(x)
            probs = F.softmax(logits, dim=1).numpy()
        pred = int(np.argmax(probs[0]))
        return pred, probs
    else:
        pred = mlp.predict(features_np)[0]
        probs = mlp.predict_proba(features_np)
        return pred, probs


# ---- Load at startup ----
tfidf, le, scaler, mlp = load_models()

# Warn clearly if anything is missing
if any(x is None for x in [tfidf, le, scaler, mlp]):
    st.warning(
        "⚠️ One or more model files could not be loaded. "
        "Make sure **tfidf.pkl**, **label_encoder.pkl**, **scaler.pkl**, "
        "and **mlp_model.pth** are in the root of your repository."
    )

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
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[0] if lines else "Unknown"

def extract_skills(text):
    skills = [
        "python", "java", "sql", "machine learning", "deep learning",
        "tensorflow", "pandas", "numpy", "excel", "power bi",
        "scikit-learn", "matplotlib", "seaborn", "plotly",
        "mongodb", "azure", "aws", "docker", "kubernetes",
        "flask", "django", "react", "javascript", "html", "css",
        "git", "linux", "tableau", "spark", "hadoop",
    ]
    return [s for s in skills if s in text.lower()]

def extract_roles(text):
    roles = []
    for word in text.lower().split():
        if any(r in word for r in ["intern", "analyst", "engineer", "developer",
                                    "scientist", "manager", "architect"]):
            roles.append(word.replace(",", "").replace(".", ""))
    return list(set(roles))

def extract_organizations(text):
    org_keywords = [
        "technologies", "pvt", "ltd", "company",
        "github", "google", "microsoft", "solutions",
        "systems", "services", "inc", "corp", "llc",
    ]
    orgs = []
    words = text.split()
    for i in range(len(words)):
        for k in org_keywords:
            if k in words[i].lower():
                phrase = " ".join(words[max(0, i - 2):i + 2])
                orgs.append(phrase.strip())
    return list(set(orgs))


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
        try:
            from pdfminer.high_level import extract_text as pdf_extract
            text = pdf_extract(io.BytesIO(uploaded_file.read()))
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
            st.stop()

    if not text or not text.strip():
        st.error("PDF appears to be empty or scanned (no extractable text).")
        st.stop()

    col1, col2 = st.columns([1, 1])

    # ------ LEFT: Resume Preview ------
    with col1:
        st.subheader("📄 Resume Preview")
        st.text_area("", text[:1500], height=420)

    # ------ RIGHT: Results ------
    with col2:
        if all(x is not None for x in [tfidf, le, scaler, mlp]):
            try:
                cleaned      = clean_text(text)
                word_count   = len(cleaned.split())
                experience   = extract_experience(text)
                skill_count  = len(extract_skills(text))

                tfidf_vec = tfidf.transform([cleaned]).toarray()
                extra     = np.array([[word_count, experience, skill_count]])
                features  = np.hstack((tfidf_vec, extra))
                features  = scaler.transform(features)

                pred, probs  = _predict(mlp, features)
                category     = le.inverse_transform([pred])[0]
                confidence   = float(np.max(probs[0]))

                st.subheader("🎯 Predicted Role")
                st.success(f"**{category}** — {confidence:.2%} confidence")

                # Top-3 predictions
                top3_idx  = np.argsort(probs[0])[::-1][:3]
                top3      = [(le.inverse_transform([i])[0], float(probs[0][i]))
                             for i in top3_idx]
                with st.expander("See top 3 predictions"):
                    for rank, (role, prob) in enumerate(top3, 1):
                        st.markdown(f"{rank}. **{role}** — {prob:.2%}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)
        else:
            st.warning("Models not loaded — cannot classify.")

        # ------ NER ------
        name  = extract_name(text)
        skills = extract_skills(text)
        roles  = extract_roles(text)
        orgs   = extract_organizations(text)

        st.subheader("📊 Extracted Information")
        st.markdown(f"**👤 Name:** {name}")
        st.markdown(f"**🏢 Organizations:** {', '.join(orgs) if orgs else 'None found'}")
        st.markdown(f"**🧠 Skills:** {', '.join(skills) if skills else 'None found'}")
        st.markdown(f"**💼 Roles:** {', '.join(roles) if roles else 'None found'}")

    st.divider()

    # ------ JSON Output ------
    st.subheader("📦 Structured JSON Output")
    result = {
        "name": name,
        "organizations": orgs,
        "skills": skills,
        "roles": roles,
    }
    st.json(result)
# -----------------------------
# Load Models
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

def extract_organizations(text):
    org_keywords = [
        "technologies", "pvt", "ltd", "company",
        "github", "google", "microsoft", "solutions"
    ]

    orgs = []
    words = text.split()

    for i in range(len(words)):
        for k in org_keywords:
            if k in words[i].lower():
                phrase = " ".join(words[max(0, i-2):i+2])
                orgs.append(phrase)

    return list(set(orgs))


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

        # -------- NER (Rule-Based) --------
        name = extract_name(text)
        skills = extract_skills(text)
        roles = extract_roles(text)
        orgs = extract_organizations(text)

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
