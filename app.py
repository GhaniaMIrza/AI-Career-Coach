import io
import os
import json

import streamlit as st
from pypdf import PdfReader
from groq import Groq
from crewai import Agent, Task, Crew, Process


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Career Coach",
    page_icon="🎯",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .result-card {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #ddd;
            margin-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🎯 AI Career Coach</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload your resume and enter your target job to get a personalized career analysis.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# GET GROQ API KEY FROM STREAMLIT SECRETS
# ---------------------------------------------------------

def get_api_key():
    """Read the Groq API key securely from Streamlit Secrets."""

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        if not api_key:
            raise ValueError("GROQ_API_KEY is empty.")

        return api_key

    except Exception:
        return None


# ---------------------------------------------------------
# RESUME TEXT EXTRACTION
# ---------------------------------------------------------

def extract_resume_text(uploaded_file):
    """Extract text from PDF or TXT files."""

    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".pdf"):
            pdf_bytes = uploaded_file.read()
            pdf_file = io.BytesIO(pdf_bytes)

            reader = PdfReader(pdf_file)

            pages = []

            for page in reader.pages:
                text = page.extract_text()

                if text:
                    pages.append(text)

            return "\n".join(pages)

        elif file_name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")

        else:
            raise ValueError("Unsupported file type.")

    except Exception as error:
        raise RuntimeError(f"Could not read the resume: {error}")


# ---------------------------------------------------------
# CREATE CREWAI AGENT
# ---------------------------------------------------------

def create_career_agent(api_key):
    """
    Create one CrewAI agent using Groq.
    CrewAI uses the OpenAI-compatible Groq endpoint.
    """

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

    agent = Agent(
        role="AI Career Coach",
        goal=(
            "Analyze a user's resume and target job description, "
            "identify relevant skills and gaps, recommend realistic "
            "career directions, and create a personalized learning roadmap."
        ),
        backstory=(
            "You are an experienced career coach and hiring advisor. "
            "You carefully compare a candidate's existing skills with "
            "the requirements of their target role. You give practical, "
            "beginner-friendly and realistic career advice."
        ),
        llm="openai/gpt-oss-120b",
        verbose=False,
        allow_delegation=False
    )

    return agent


# ---------------------------------------------------------
# RUN CAREER ANALYSIS
# ---------------------------------------------------------

def analyze_career(resume_text, target_job, job_description, api_key):
    """Run the single CrewAI agent."""

    agent = create_career_agent(api_key)

    task_description = f"""
You are analyzing a candidate for a career coaching report.

CANDIDATE RESUME:
{resume_text}

TARGET JOB:
{target_job}

JOB DESCRIPTION:
{job_description if job_description.strip() else "No detailed job description was provided."}

Analyze the information above and produce a clear career coaching report.

Your analysis MUST include:

1. Candidate Profile
- Briefly summarize the candidate's current background.

2. Relevant Skills
- Identify skills from the resume that are relevant to the target job.
- Separate technical skills and soft skills where possible.

3. Skill Gaps
- Identify important skills required by the target role that appear to be missing
  or insufficiently demonstrated in the resume.
- Do not invent experience that is not present.

4. Career Directions
- Recommend 2 to 3 realistic career directions based on the candidate's
  current skills and target role.
- Explain why each direction fits.

5. Personalized Learning Roadmap
Create a practical roadmap divided into:
- 0–1 month
- 1–3 months
- 3–6 months

For every period, recommend skills, learning activities, and practical projects.

6. Resume Improvement Suggestions
- Give 3 to 5 specific suggestions for improving the resume for the target role.

7. Final Recommendation
- Give a short overall assessment.
- Clearly mention the most important skill the candidate should learn next.

IMPORTANT:
- Base the analysis only on the provided resume and job information.
- Do not claim the candidate has skills or experience that are not shown.
- Keep the advice practical and beginner-friendly.
- Use headings and bullet points.
"""

    task = Task(
        description=task_description,
        expected_output=(
            "A detailed but concise career coaching report with clear headings, "
            "skill analysis, skill gaps, career directions, learning roadmap, "
            "resume suggestions, and final recommendation."
        ),
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()

    return str(result)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("📋 Career Coach")

    st.write(
        "Upload your resume and provide information about the job "
        "you want to pursue."
    )

    st.divider()

    st.info(
        "🔒 Your Groq API key is read from Streamlit Secrets "
        "and is never written in the application code."
    )


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

uploaded_resume = st.file_uploader(
    "📄 Upload your Resume",
    type=["pdf", "txt"],
    help="Supported formats: PDF and TXT"
)

target_job = st.text_input(
    "🎯 Target Job",
    placeholder="Example: Junior Data Analyst"
)

job_description = st.text_area(
    "📝 Job Description",
    height=180,
    placeholder="Paste the job description here..."
)


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

analyze_button = st.button(
    "🚀 Analyze My Career",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------

if analyze_button:

    # Validate API key
    api_key = get_api_key()

    if not api_key:
        st.error(
            "Groq API key not found. Please add GROQ_API_KEY "
            "to your Streamlit Secrets."
        )
        st.stop()

    # Validate resume
    if uploaded_resume is None:
        st.warning("Please upload your resume first.")
        st.stop()

    # Validate target job
    if not target_job.strip():
        st.warning("Please enter your target job.")
        st.stop()

    # Extract resume
    try:
        with st.spinner("📄 Reading your resume..."):
            resume_text = extract_resume_text(uploaded_resume)

        if not resume_text.strip():
            st.error(
                "The uploaded resume appears to contain no readable text."
            )
            st.stop()

    except Exception as error:
        st.error(str(error))
        st.stop()

    # Run AI analysis
    try:

        with st.spinner(
            "🤖 AI Career Coach is analyzing your profile..."
        ):

            report = analyze_career(
                resume_text=resume_text,
                target_job=target_job,
                job_description=job_description,
                api_key=api_key
            )

        st.success("Career analysis completed!")

        st.divider()

        st.header("📊 Your Career Analysis")

        st.markdown(report)

    except Exception as error:

        st.error(
            "Something went wrong while generating your career analysis."
        )

        st.caption(
            f"Technical details: {str(error)}"
        )