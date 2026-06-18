import streamlit as st
from google import genai
import pypdf as pdf
import os
from dotenv import load_dotenv

# Directly set your Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

def extract_text_from_pdf(uploaded_file):
    """Extracts text content from an uploaded PDF file."""
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def get_llm_analysis(resume_text, job_description):
    """Sends prompt to Gemini LLM to analyze resume against job description."""
    prompt = f"""
    You are an expert HR Recruiter and ATS (Applicant Tracking System) optimization specialist.
    Analyze the following Resume against the provided Job Description (JD).
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Provide a structured response exactly with the following sections:
    1. **Match Percentage**: A realistic percentage score based on criteria matching.
    2. **Key Skill Gaps**: Missing technical/soft skills or experiences required by the JD.
    3. **Strengths**: Where the candidate aligns perfectly.
    4. **Actionable Recommendations**: Clear advice on how the candidate can improve their resume for this role.
    """
    
    # FIXED: Uses the updated client generation syntax natively supporting gemini-2.5-flash
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    return response.text

# --- Streamlit UI ---
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI Resume Analyzer")
st.subheader("Smart ATS Evaluation & Skill Gap Analysis")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Inputs")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area("Paste Job Description Here", height=300)
    
    submit_button = st.button("Analyze Resume")

with col2:
    st.header("Analysis Results")
    if submit_button:
        if uploaded_file is not None and job_description.strip() != "":
            with st.spinner("Extracting text and analyzing with GenAI..."):
                try:
                    # Parse PDF
                    resume_text = extract_text_from_pdf(uploaded_file)
                    
                    # Get AI response
                    analysis_result = get_llm_analysis(resume_text, job_description)
                    
                    st.success("Analysis Complete!")
                    st.markdown(analysis_result)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a PDF resume and provide a job description.")
