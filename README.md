# 🚀 AI Resume Screener

🔗 **Live App:** https://resume-screener-kf4rjngyk4hqkjsmuzsneb.streamlit.app/

---

## 📌 Overview

The **AI Resume Screener** is an intelligent web application that automates resume analysis using **Natural Language Processing (NLP)** and a **Deep Learning model**.

It reads resumes in PDF format, classifies them into job categories, calculates an ATS-style score, and suggests suitable roles along with feedback — making it a practical decision-support tool for recruitment.

---

## 🎯 Key Features

* 📄 Upload and analyze PDF resumes
* 🧠 Resume classification using Deep Learning (MLP)
* 🔍 NLP-based feature extraction (TF-IDF)
* 🎯 Domain prediction (e.g., IT, Healthcare, Finance, etc.)
* 💼 Role recommendations based on predicted category
* 📊 ATS score based on model confidence
* 🤖 Domain-specific feedback for improvement
* 📦 Structured JSON output

---

## 🧠 Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Machine Learning:** Scikit-learn (MLP Classifier)
* **NLP:** TF-IDF Vectorization
* **Data Processing:** NumPy, Regex
* **PDF Parsing:** pdfminer

---

## ⚙️ How It Works

1. The user uploads a resume (PDF)
2. Text is extracted and cleaned
3. NLP (TF-IDF) converts text into numerical features
4. Deep Learning model (MLP) predicts job category
5. ATS score is calculated based on prediction confidence
6. Recommended roles are generated from category mapping
7. Feedback is provided based on domain alignment

---

## 📊 Model Details

* **Model Type:** Multi-Layer Perceptron (MLP)
* **Architecture:** Input → 256 → 128 → Output
* **Activation:** ReLU
* **Regularization:** Dropout + L2
* **Optimizer:** Adam
* **Evaluation:** Accuracy, Precision, Recall, F1-score

---

## 🧠 Objectives Achieved

### 🔹 NLP Objective

Convert unstructured resume text into meaningful numerical features for machine understanding.

### 🔹 Deep Learning Objective

Learn patterns from resume data and accurately classify resumes into job categories.

---

## 📂 Project Structure

```bash
.
├── app.py
├── tfidf.pkl
├── scaler.pkl
├── mlp_model.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/resume-screener.git
cd resume-screener
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 Deployment

The application is deployed using **Streamlit Cloud**.

🔗 **Live Demo:**
https://resume-screener-kf4rjngyk4hqkjsmuzsneb.streamlit.app/

---

## 💡 Future Improvements

* 🔍 Job description matching
* 📈 Resume scoring based on real ATS systems
* 🤖 LLM-based resume feedback (GPT integration)
* 📄 Resume improvement suggestions
* 🌐 Multi-language resume support

---

## 🙌 Author

**Roshan Paul Goddu**
📧 [roshanpaul604@gmail.com](mailto:roshanpaul604@gmail.com)
🔗 https://github.com/imroshan18

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
