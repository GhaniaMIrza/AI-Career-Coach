import io
import os
import base64

import streamlit as st
from pypdf import PdfReader
from PIL import Image
from groq import Groq
from crewai import Agent, Task, Crew, Process, LLM


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

    .stApp {
        background-color: #F7F9FC;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

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

    .input-card {
        background: #FFFFFF;
        border: 1px solid #E5E9F2;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 4px 18px rgba(31, 41, 55, 0.04);
        margin-bottom: 18px;
    }

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

    div[data-baseweb="input"],
    div[data-baseweb="textarea"] {
        border-radius: 10px;
    }

    input, textarea {
        font-size: 15px !important;
    }

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

    [data-testid="stExpander"] {
        border: 1px solid #E2E6EF;
        border-radius: 12px;
        background: #FFFFFF;
    }

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
# GET API KEY
# =========================================================

def get_api_key():
    """Read Groq API key securely from Streamlit Secrets."""

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        if not api_key:
            return None

        return api_key

    except Exception:
        return None


# =========================================================
# NORMAL PDF/TXT EXTRACTION
# =========================================================

def extract_text_from_pdf(uploaded_file):
    """Try extracting normal text from a PDF."""

    pdf_bytes = uploaded_file.getvalue()

    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages).strip()


def extract_text_from_txt(uploaded_file):
    """Extract text from TXT."""

    return uploaded_file.getvalue().decode(
        "utf-8",
        errors="ignore"
    ).strip()


# =========================================================
# IMAGE PREPARATION
# =========================================================

def image_to_data_url(image_bytes, mime_type="image/jpeg"):
    """Convert image bytes to a base64 data URL."""

    encoded = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def prepare_image(image_bytes):
    """
    Resize an image so that OCR requests remain reasonably small.
    """

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    max_width = 1600

    if image.width > max_width:
        ratio = max_width / image.width

        new_height = int(image.height * ratio)

        image = image.resize(
            (max_width, new_height)
        )

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=85,
        optimize=True
    )

    return output.getvalue()


# =========================================================
# GROQ VISION OCR
# =========================================================

def extract_text_from_image(
    image_bytes,
    api_key,
    mime_type="image/jpeg"
):
    """
    Use Groq's vision model to read text from a resume image.
    This is only used for OCR.
    """

    client = Groq(api_key=api_key)

    prepared_image = prepare_image(image_bytes)

    image_data_url = image_to_data_url(
        prepared_image,
        "image/jpeg"
    )

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
Read this resume image carefully.

Extract all readable resume information as plain text.

Preserve important details such as:
- Name
- Professional summary
- Education
- Skills
- Work experience
- Projects
- Certifications
- Courses
- Achievements
- Contact information if visible

Do not analyze the candidate.
Do not give career advice.
Only transcribe the useful resume content accurately.
If something is unclear, do not invent it.
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=6000
    )

    text = response.choices[0].message.content

    if not text:
        raise RuntimeError(
            "The vision model could not extract text from the image."
        )

    return text.strip()


# =========================================================
# IMAGE PDF OCR
# =========================================================

def extract_text_from_scanned_pdf(
    uploaded_file,
    api_key
):
    """
    Convert scanned PDF pages into images and read them
    with Groq Vision.
    """

    try:
        import fitz

    except ImportError:
        raise RuntimeError(
            "PyMuPDF is required for scanned PDF support. "
            "Add pymupdf to requirements.txt."
        )

    pdf_bytes = uploaded_file.getvalue()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    page_texts = []

    total_pages = len(document)

    if total_pages == 0:
        raise RuntimeError(
            "The PDF contains no pages."
        )

    for start in range(0, total_pages, 3):

        end = min(start + 3, total_pages)

        batch_images = []

        for page_number in range(start, end):

            page = document.load_page(page_number)

            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(1.5, 1.5),
                alpha=False
            )

            image_bytes = pixmap.tobytes("jpg")

            prepared_image = prepare_image(
                image_bytes
            )

            image_data_url = image_to_data_url(
                prepared_image,
                "image/jpeg"
            )

            batch_images.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                }
            )

        client = Groq(api_key=api_key)

        content = [
            {
                "type": "text",
                "text": """
Read these resume pages carefully.

Extract all readable resume information as plain text.

Combine the information from all pages in their
logical order.

Preserve:
- Name
- Summary
- Education
- Skills
- Experience
- Projects
- Certifications
- Courses
- Achievements
- Contact details if visible

Do not analyze the candidate.
Do not give career advice.
Only extract the resume information.
Do not invent missing information.
"""
            }
        ]

        content.extend(batch_images)

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0,
            max_tokens=8000
        )

        text = response.choices[0].message.content

        if text:
            page_texts.append(text.strip())

    document.close()

    final_text = "\n\n".join(page_texts)

    if not final_text.strip():
        raise RuntimeError(
            "Could not read text from the scanned PDF."
        )

    return final_text


# =========================================================
# UNIVERSAL RESUME EXTRACTION
# =========================================================

def extract_resume_text(
    uploaded_file,
    api_key
):
    """
    Decide automatically whether the resume needs
    normal text extraction or vision OCR.
    """

    file_name = uploaded_file.name.lower()

    # ---------------- PDF ----------------

    if file_name.endswith(".pdf"):

        text = extract_text_from_pdf(
            uploaded_file
        )

        # Normal text PDF
        if len(text.strip()) >= 50:
            return text

        # Scanned/image PDF
        return extract_text_from_scanned_pdf(
            uploaded_file,
            api_key
        )

    # ---------------- TXT ----------------

    if file_name.endswith(".txt"):

        text = extract_text_from_txt(
            uploaded_file
        )

        if not text:
            raise RuntimeError(
                "The TXT resume is empty."
            )

        return text

    # ---------------- JPG / JPEG / PNG ----------------

    if file_name.endswith(
        (".jpg", ".jpeg", ".png")
    ):

        mime_type = "image/png"

        if file_name.endswith(
            (".jpg", ".jpeg")
        ):
            mime_type = "image/jpeg"

        return extract_text_from_image(
            uploaded_file.getvalue(),
            api_key,
            mime_type
        )

    raise ValueError(
        "Unsupported resume format."
    )


# =========================================================
# CREATE CREWAI AGENT
# =========================================================

def create_career_agent(api_key):
    """
    Create ONE CrewAI agent.

    Explicitly configure the Groq LLM so CrewAI receives
    the complete model ID:
    openai/gpt-oss-120b
    """

    career_llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2
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

        llm=career_llm,

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
    """Run the single CrewAI career coach agent."""

    agent = create_career_agent(
        api_key
    )

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

0–1 month
1–3 months
3–6 months

For every period, recommend:
- Skills to learn
- Learning activities
- Practical projects

6. Resume Improvement Suggestions
- Give 3 to 5 specific suggestions for improving the resume
  for the target role.

7. Final Recommendation
- Give a short overall assessment.
- Clearly mention the most important skill the candidate should learn next.

IMPORTANT:
- Base the analysis only on the provided resume and job information.
- Do not claim the candidate has skills or experience that are not shown.
- Keep the advice practical and beginner-friendly.
- Use clear headings and bullet points.
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

    st.caption(
        "Powered by CrewAI + Groq"
    )


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


# =========================================================
# RESUME UPLOAD
# =========================================================

st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)

st.markdown(
    "**📄 Resume**"
)

st.caption(
    "Supported formats: PDF, TXT, JPG and PNG. "
    "Scanned/image-based resumes are supported."
)

uploaded_resume = st.file_uploader(
    "Upload your resume",
    type=[
        "pdf",
        "txt",
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed",
    help=(
        "Upload a text PDF, scanned PDF, TXT, JPG or PNG resume."
    )
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# =========================================================
# JOB INPUTS
# =========================================================

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


st.markdown(
    "<br>",
    unsafe_allow_html=True
)


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
    # API KEY
    # -----------------------------------------------------

    api_key = get_api_key()

    if not api_key:

        st.error(
            "AI connection is unavailable. "
            "Please check your Streamlit Secrets configuration."
        )

        st.stop()

    # -----------------------------------------------------
    # RESUME VALIDATION
    # -----------------------------------------------------

    if uploaded_resume is None:

        st.warning(
            "Please upload your resume before starting the analysis."
        )

        st.stop()

    # -----------------------------------------------------
    # TARGET JOB VALIDATION
    # -----------------------------------------------------

    if not target_job.strip():

        st.warning(
            "Please enter your target job."
        )

        st.stop()

    # -----------------------------------------------------
    # EXTRACT RESUME
    # -----------------------------------------------------

    try:

        file_name = uploaded_resume.name.lower()

        if file_name.endswith(".pdf"):

            with st.spinner(
                "📄 Reading your PDF resume..."
            ):

                normal_text = extract_text_from_pdf(
                    uploaded_resume
                )

            if len(normal_text.strip()) >= 50:

                resume_text = normal_text

            else:

                with st.spinner(
                    "🔍 This looks like an image-based resume. "
                    "Reading the resume with AI vision..."
                ):

                    resume_text = extract_text_from_scanned_pdf(
                        uploaded_resume,
                        api_key
                    )

        elif file_name.endswith(".txt"):

            with st.spinner(
                "📄 Reading your resume..."
            ):

                resume_text = extract_text_from_txt(
                    uploaded_resume
                )

        else:

            with st.spinner(
                "🔍 Reading your image-based resume..."
            ):

                mime_type = "image/png"

                if file_name.endswith(
                    (".jpg", ".jpeg")
                ):
                    mime_type = "image/jpeg"

                resume_text = extract_text_from_image(
                    uploaded_resume.getvalue(),
                    api_key,
                    mime_type
                )

        if not resume_text.strip():

            st.error(
                "The uploaded resume could not be read."
            )

            st.stop()

    except Exception as error:

        st.error(
            "We could not read this resume."
        )

        st.caption(
            f"Technical details: {str(error)}"
        )

        st.stop()

    # -----------------------------------------------------
    # CAREER ANALYSIS
    # -----------------------------------------------------

    try:

        with st.spinner(
            "🤖 Your AI Career Coach is analyzing your profile..."
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

    except Exception as error:

        st.error(
            "Something went wrong while generating your career analysis."
        )

        st.caption(
            f"Technical details: {str(error)}"
        )
