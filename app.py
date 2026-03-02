import streamlit as st
import PyPDF2
import re
import tempfile
from sentence_transformers import SentenceTransformer, util
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="AI Resume Analyzer Pro", page_icon="🚀", layout="wide")

st.title("🚀 AI Resume Analyzer PRO")
st.write("---")

# ----------------------------
# Load Model (Cached)
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ----------------------------
# Skill Database
# ----------------------------
skills_list = [
    "python", "java", "c++", "machine learning",
    "deep learning", "sql", "data analysis",
    "html", "css", "javascript", "react",
    "django", "flask", "tensorflow",
    "pandas", "numpy", "power bi", "excel"
]

# ----------------------------
# Extract PDF Text
# ----------------------------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# ----------------------------
# Clean Text
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# ----------------------------
# File Upload Section
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_file = st.file_uploader("📑 Upload Job Description (PDF)", type=["pdf"])

jd_text_manual = st.text_area("OR Paste Job Description Here")

# ----------------------------
# Analyze Resume
# ----------------------------
if st.button("🔍 Analyze Resume"):

    if resume_file and (jd_file or jd_text_manual):

        resume_text = extract_text_from_pdf(resume_file)

        if jd_file:
            job_text = extract_text_from_pdf(jd_file)
        else:
            job_text = jd_text_manual

        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_text)

        # ----------------------------
        # Semantic Similarity
        # ----------------------------
        embeddings = model.encode([resume_text, job_text])
        similarity_score = util.cos_sim(embeddings[0], embeddings[1])
        match_percentage = round(float(similarity_score[0][0]) * 100, 2)

        # ----------------------------
        # Skill Matching
        # ----------------------------
        resume_skills = [skill for skill in skills_list if skill in resume_clean]
        job_skills = [skill for skill in skills_list if skill in job_clean]

        matched_skills = list(set(resume_skills) & set(job_skills))
        missing_skills = list(set(job_skills) - set(resume_skills))

        # ----------------------------
        # Display Results
        # ----------------------------
        st.subheader("📊 Match Percentage")
        st.progress(int(match_percentage))
        st.write(f"### {match_percentage}% Match")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("✅ Matched Skills")
            st.write(matched_skills if matched_skills else "No matches found")

        with col4:
            st.subheader("❌ Missing Skills")
            st.write(missing_skills if missing_skills else "No major skills missing")

        # ----------------------------
        # Recommendation
        # ----------------------------
        st.subheader("📌 Recommendation")

        if match_percentage > 75:
            st.success("Strong match! High selection probability.")
        elif match_percentage > 50:
            st.warning("Moderate match. Improve missing skills.")
        else:
            st.error("Low match. Consider updating resume.")

        # Store results for PDF
        st.session_state["report"] = {
            "match": match_percentage,
            "matched": matched_skills,
            "missing": missing_skills
        }

    else:
        st.error("Please upload resume and job description.")

# ----------------------------
# Download PDF Report
# ----------------------------
if "report" in st.session_state:

    if st.button("📥 Download Report as PDF"):

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

        doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
        elements = []

        style = ParagraphStyle(
            name='Normal',
            fontSize=12,
            textColor=colors.black
        )

        elements.append(Paragraph("AI Resume Analysis Report", style))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Match Percentage: {st.session_state['report']['match']}%", style))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"Matched Skills: {', '.join(st.session_state['report']['matched'])}", style))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"Missing Skills: {', '.join(st.session_state['report']['missing'])}", style))

        doc.build(elements)

        with open(temp_file.name, "rb") as f:
            st.download_button(
                label="Click to Download",
                data=f,
                file_name="Resume_Report.pdf",
                mime="application/pdf"
            )