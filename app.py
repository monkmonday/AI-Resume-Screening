import streamlit as st
import re
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# UI
st.title("🤖 AI Resume Screening Tool")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if uploaded_file and job_desc:
        resume = extract_text_from_pdf(uploaded_file)

        resume_clean = preprocess(resume)
        job_clean = preprocess(job_desc)

        # TF-IDF
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        vectors = vectorizer.fit_transform([resume_clean, job_clean])

        score = cosine_similarity(vectors[0], vectors[1])[0][0]

        # Skills
        skills = {
            "python": ["python"],
            "machine learning": ["machine learning", "ml"],
            "data analysis": ["data analysis"],
            "deep learning": ["deep learning"],
            "nlp": ["nlp", "natural language processing"],
            "tensorflow": ["tensorflow"],
            "sql": ["sql"],
            "pandas": ["pandas"]
        }

        found_skills = []
        missing_skills = []

        for skill, keywords in skills.items():
            if any(keyword in resume_clean for keyword in keywords):
                found_skills.append(skill)
            else:
                missing_skills.append(skill)

        # Output
        st.subheader("📊 Match Score")
        st.write(round(score * 100, 2), "%")

        if score > 0.75:
            st.success("Excellent Match")
        elif score > 0.5:
            st.info("Good Match")
        elif score > 0.3:
            st.warning("Average Match")
        else:
            st.error("Low Match")

        st.subheader("✅ Skills Found")
        st.write(found_skills)

        st.subheader("❌ Missing Skills")
        st.write(missing_skills)

        st.subheader("📌 Recommendations")
        for skill in missing_skills:
            st.write("-", skill)

    else:
        st.warning("Please upload resume and enter job description")