import io
import os

import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Task, Crew, Process


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Career Coach",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM UI / THEME
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background-color: #F7F9FC;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- TYPOGRAPHY ---------- */

    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }

    h1, h2, h3 {
        color: #172033;
        letter-spacing: -0.4px;
    }

    p, label {
        color: #4B5565;
    }

    /* ---------- HEADER ---------- */

    .brand-container {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }

    .brand-icon {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        background: linear-gradient(135deg, #3157D5, #6846D8);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 27px;
        box-shadow: 0 8px 20px rgba(49, 87, 213, 0.18);
    }

    .brand-title {
        font-size: 36px;
        font-weight: 750;
        color: #172033;
        line-height: 1.1;
    }

    .brand-subtitle {
        margin-left: 66px;
        margin-top: 4px;
        color: #667085;
        font-size: 16px;
    }

    /* ---------- SECTION TITLES ---------- */

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #172033;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    .section-description {
        font-size: 14px;
        color: #667085;
        margin-bottom: 16px;
    }

    /* ---------- INPUT CARDS ---------- */

    .input-card {
        background: #FFFFFF;
        border: 1px solid #E5E9F2;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 4px 18px rgba(31, 41, 55, 0.04);
        margin-bottom: 18px;
    }

    /* ---------- UPLOAD AREA ---------- */

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px solid #DDE3EE;
        border-radius: 14px;
        padding: 8px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #F8FAFD;
        border-radius: 10px;
    }

    /* ---------- TEXT INPUTS ---------- */

    div[data-baseweb="input"],
    div[data-baseweb="textarea"] {
        border-radius: 10px;
    }

    input, textarea {
        font-size: 15px !important;
    }

    /* ---------- PRIMARY BUTTON ---------- */

    div.stButton > button[kind="primary"] {
        width: 100%;
        min-height: 50px;
        border-radius: 11px;
        border: none;
        background: linear-gradient(135deg, #3157D5, #6846D8);
        color: white;
        font-size: 16px;
        font-weight: 650;
        box-shadow: 0 8px 18px rgba(49, 87, 213, 0.20);
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 10px 24px rgba(49, 87, 213, 0.28);
        transform: translateY(-1px);
    }

    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E7EAF0;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .sidebar-brand {
        font-size: 21px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 5px;
    }

    .sidebar-text {
        font-size: 14px;
        color: #667085;
        line-height: 1.6;
    }

    .connection-card {
        background: #F7F9FC;
        border: 1px solid #E4E8F0;
        border-radius: 12px;
        padding: 14px;
        margin-top: 15px;
    }

    .connection-title {
        font-weight: 650;
        color: #172033;
        font-size: 14px;
    }

    .connection-status {
        color: #17834A;
        font-size: 13px;
        margin-top: 5px;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #20A464;
        border-radius: 50%;
        margin-right: 6px;
    }

    /* ---------- RESULT CONTAINER ---------- */

    .result-header {
        background: linear-gradient(
            135deg,
            #EEF3FF,
            #F5F1FF
        );
        border: 1px solid #DDE5FA;
        border-radius: 16px;
        padding: 22px;
        margin-top: 8px;
        margin-bottom: 20px;
    }

    .result-header-title {
        font-size: 22px;
        font-weight: 700;
        color: #172033;
    }

    .result-header-text {
        color: #667085;
        font-size: 14px;
        margin-top: 4px;
    }

    /* ---------- EXPANDERS ---------- */

    [data-testid="stExpander"] {
        border: 1px solid #E2E6EF;
        border-radius: 12px;
        background: #FFFFFF;
    }

    /* ---------- DIVIDER ---------- */

    hr {
        border-color: #E5E9F2;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="brand-container">
        <div class="brand-icon">💼</div>
        <div class="brand-title">AI Career Coach</div>
    </div>

    <div class="brand-subtitle">
        Turn your resume into a personalized career roadmap.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# GET GROQ API KEY FROM STREAMLIT SECRETS
# =========================================================

def get_api_key():
    """Read the Groq API key securely from Streamlit Secrets."""

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        if not api_key:
            raise ValueError("GROQ_API_KEY is empty.")

        return api_key

    except Exception:
        return None


# =========================================================
# RESUME TEXT EXTRACTION
# =========================================================

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
            return uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        else:
            raise ValueError("Unsupported file type.")

    except Exception as error:
        raise RuntimeError(
            f"Could not read the resume: {error}"
        )


# =========================================================
# CREATE CREWAI AGENT
# =========================================================

def create_career_agent(api_key):
    """
    Create one CrewAI agent using Groq.
    """

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = (
        "https://api.groq.com/openai/v1"
    )

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


# =========================================================
# RUN CAREER ANALYSIS
# =========================================================

def analyze_career(
    resume_text,
    target_job,
    job_description,
    api_key
):
    """Run the single CrewAI agent."""

    agent = create_career_agent(api_key)

    task_description = f"""
You are analyzing a candidate for a career coaching report.

CANDIDATE RESUME:
{resume_text}

TARGET JOB:
{target_job}

JOB DESCRIPTION:
{
    job_description
    if job_description.strip()
    else "No detailed job description was provided."
}

Analyze the information above and produce a clear career coaching report.

Your analysis MUST include:

1. Candidate Profile
- Briefly summarize the candidate's current background.

2. Relevant Skills
- Identify skills from the resume that are relevant to the target job.
- Separate technical skills and soft skills where possible.

3. Skill Gaps
- Identify important skills required by the target role that appear to be
  missing or insufficiently demonstrated in the resume.
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


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">💼 AI Career Coach</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
        Build a clearer career direction from your resume,
        skills and target role.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
        <div class="connection-card">
            <div class="connection-title">
                🔑 AI Connection
            </div>
            <div class="connection-status">
                <span class="status-dot"></span>
                Groq AI connected
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption("Powered by CrewAI + Groq")


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">Build Your Career Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Upload your resume and provide the role you want to pursue.
        Add a job description for a more targeted analysis.
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# RESUME
# ---------------------------------------------------------

st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)

st.markdown(
    "**📄 Resume**"
)

st.caption(
    "Upload your current resume in PDF or TXT format."
)

uploaded_resume = st.file_uploader(
    "Upload your resume",
    type=["pdf", "txt"],
    label_visibility="collapsed",
    help="Supported formats: PDF and TXT"
)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# JOB INPUTS
# ---------------------------------------------------------

col1, col2 = st.columns(
    [1, 1],
    gap="large"
)

with col1:

    st.markdown(
        '<div class="section-title">🎯 Target Role</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "What position are you interested in?"
    )

    target_job = st.text_input(
        "Target Job",
        placeholder="e.g. Junior Data Analyst",
        label_visibility="collapsed"
    )


with col2:

    st.markdown(
        '<div class="section-title">📝 Job Description</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Paste the job description for a more precise analysis."
    )

    job_description = st.text_area(
        "Job Description",
        height=120,
        placeholder=(
            "Paste the job description here..."
        ),
        label_visibility="collapsed"
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_button = st.button(
    "🚀  Analyze My Career",
    type="primary",
    use_container_width=True
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    # -----------------------------------------------------
    # Validate API key
    # -----------------------------------------------------

    api_key = get_api_key()

    if not api_key:
        st.error(
            "AI connection is unavailable. "
            "Please check your Streamlit Secrets configuration."
        )
        st.stop()

    # -----------------------------------------------------
    # Validate resume
    # -----------------------------------------------------

    if uploaded_resume is None:
        st.warning(
            "Please upload your resume before starting the analysis."
        )
        st.stop()

    # -----------------------------------------------------
    # Validate target job
    # -----------------------------------------------------

    if not target_job.strip():
        st.warning(
            "Please enter your target job."
        )
        st.stop()

    # -----------------------------------------------------
    # Extract resume
    # -----------------------------------------------------

    try:

        with st.spinner(
            "Reading your resume..."
        ):

            resume_text = extract_resume_text(
                uploaded_resume
            )

        if not resume_text.strip():

            st.error(
                "The uploaded resume appears to contain "
                "no readable text."
            )

            st.stop()

    except Exception as error:

        st.error(str(error))
        st.stop()

    # -----------------------------------------------------
    # Run AI analysis
    # -----------------------------------------------------

    try:

        with st.spinner(
            "Your AI Career Coach is analyzing your profile..."
        ):

            report = analyze_career(
                resume_text=resume_text,
                target_job=target_job,
                job_description=job_description,
                api_key=api_key
            )

        st.success(
            "Your personalized career analysis is ready."
        )

        st.markdown(
            """
            <div class="result-header">
                <div class="result-header-title">
                    📊 Your Career Analysis
                </div>
                <div class="result-header-text">
                    Personalized insights based on your resume
                    and target career.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(report)

    # -----------------------------------------------------
    # Error handling
    # -----------------------------------------------------

    except Exception as error:

        st.error(
            "Something went wrong while generating your career analysis."
        )

        st.caption(
            f"Technical details: {str(error)}"
        )
