import io
import base64

# =========================================================
# CREWAI + GROQ COMPATIBILITY FIX
# =========================================================
# Prevents the unsupported cache_breakpoint field from
# causing Groq requests to fail in some CrewAI/LiteLLM versions.

try:
    import crewai.llms.cache as crew_cache

    if hasattr(crew_cache, "mark_cache_breakpoint"):
        crew_cache.mark_cache_breakpoint = lambda msg: msg

except Exception:
    pass


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
# CUSTOM UI
# =========================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
    ========================= */

    .stApp {
        background-color: #F7F8FC;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    html, body, [class*="css"] {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    h1, h2, h3 {
        color: #171326;
        letter-spacing: -0.4px;
    }

    p, label {
        color: #555B6E;
    }


    /* =========================
       BRAND
       ========================= */

    .brand-container {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }

    .brand-icon {
        width: 54px;
        height: 54px;
        border-radius: 15px;

        background: linear-gradient(
            135deg,
            #6D3DF5,
            #8057F7
        );

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 28px;

        box-shadow:
            0 8px 22px rgba(109, 61, 245, 0.20);
    }

    .brand-title {
        font-size: 36px;
        font-weight: 750;
        color: #171326;
        line-height: 1.1;
    }

    .brand-subtitle {
        margin-left: 68px;
        margin-top: 5px;
        color: #6B7280;
        font-size: 16px;
    }


    /* =========================
       SECTION HEADINGS
       ========================= */

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #171326;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    .section-description {
        font-size: 14px;
        color: #6B7280;
        margin-bottom: 16px;
    }


    /* =========================
       INPUT CARD
       ========================= */

    .input-card {
        background: #FFFFFF;

        border: 1px solid #E6E8F0;

        border-radius: 16px;

        padding: 22px;

        box-shadow:
            0 4px 18px rgba(31, 41, 55, 0.04);

        margin-bottom: 18px;
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF;

        border: 1px solid #DDE1EB;

        border-radius: 14px;

        padding: 8px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #F9FAFD;
        border-radius: 10px;
    }


    /* =========================
       INPUTS
       ========================= */

    div[data-baseweb="input"],
    div[data-baseweb="textarea"] {
        border-radius: 10px;
    }

    input,
    textarea {
        font-size: 15px !important;
    }


    /* =========================
       MAIN CTA BUTTON
       ========================= */

    div.stButton > button[kind="primary"] {

        width: 100%;

        min-height: 52px;

        border-radius: 12px;

        border: none;

        background:
            linear-gradient(
                135deg,
                #6D3DF5,
                #7C4DFF
            );

        color: #FFFFFF;

        font-size: 16px;

        font-weight: 700;

        box-shadow:
            0 8px 22px
            rgba(109, 61, 245, 0.28);

        transition:
            all 0.2s ease;
    }


    div.stButton > button[kind="primary"]:hover {

        background:
            linear-gradient(
                135deg,
                #5B2DD8,
                #6D3DF5
            );

        box-shadow:
            0 10px 28px
            rgba(109, 61, 245, 0.38);

        transform:
            translateY(-1px);
    }


    /* =========================
       SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {

        background-color: #FFFFFF;

        border-right:
            1px solid #E7E9F0;
    }

    [data-testid="stSidebar"] .block-container {

        padding-top: 2rem;
    }

    .sidebar-brand {

        font-size: 21px;

        font-weight: 700;

        color: #171326;

        margin-bottom: 5px;
    }

    .sidebar-text {

        font-size: 14px;

        color: #6B7280;

        line-height: 1.6;
    }


    /* =========================
       RESULT HEADER
       ========================= */

    .result-header {

        background:
            linear-gradient(
                135deg,
                #F1EDFF,
                #F7F3FF
            );

        border:
            1px solid #E3DBFF;

        border-radius: 16px;

        padding: 22px;

        margin-top: 8px;

        margin-bottom: 20px;
    }

    .result-header-title {

        font-size: 22px;

        font-weight: 700;

        color: #171326;
    }

    .result-header-text {

        color: #6B7280;

        font-size: 14px;

        margin-top: 4px;
    }


    /* =========================
       EXPANDERS
       ========================= */

    [data-testid="stExpander"] {

        border:
            1px solid #E2E5ED;

        border-radius: 12px;

        background: #FFFFFF;
    }


    hr {

        border-color:
            #E5E7EF;
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

        <div class="brand-icon">
            💼
        </div>

        <div class="brand-title">
            AI Career Coach
        </div>

    </div>

    <div class="brand-subtitle">
        Turn your resume into a personalized career roadmap.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# SECURE API KEY
# =========================================================

def get_api_key():
    """
    Read the Groq API key from Streamlit Secrets.

    The API key is never hard-coded in the application.
    """

    try:

        api_key = st.secrets["GROQ_API_KEY"]

        if not api_key:
            return None

        return api_key

    except Exception:

        return None


# =========================================================
# TEXT PDF EXTRACTION
# =========================================================

def extract_text_from_pdf(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    reader = PdfReader(
        io.BytesIO(pdf_bytes)
    )

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages).strip()


# =========================================================
# TXT EXTRACTION
# =========================================================

def extract_text_from_txt(uploaded_file):

    return uploaded_file.getvalue().decode(
        "utf-8",
        errors="ignore"
    ).strip()


# =========================================================
# IMAGE PREPARATION
# =========================================================

def image_to_data_url(
    image_bytes,
    mime_type="image/jpeg"
):

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        f"data:{mime_type};base64,{encoded}"
    )


def prepare_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    max_width = 1600

    if image.width > max_width:

        ratio = (
            max_width / image.width
        )

        new_height = int(
            image.height * ratio
        )

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

    client = Groq(
        api_key=api_key
    )

    prepared_image = prepare_image(
        image_bytes
    )

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

Extract all readable resume information
as plain text.

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

Only transcribe the useful resume content.

If something is unclear,
do not invent it.
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

    text = (
        response
        .choices[0]
        .message
        .content
    )

    if not text:

        raise RuntimeError(
            "The vision model could not "
            "extract text from the image."
        )

    return text.strip()


# =========================================================
# SCANNED PDF OCR
# =========================================================

def extract_text_from_scanned_pdf(
    uploaded_file,
    api_key
):

    try:

        import fitz

    except ImportError:

        raise RuntimeError(
            "PyMuPDF is required for scanned "
            "PDF support."
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

    for start in range(
        0,
        total_pages,
        3
    ):

        end = min(
            start + 3,
            total_pages
        )

        batch_images = []

        for page_number in range(
            start,
            end
        ):

            page = document.load_page(
                page_number
            )

            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(
                    1.5,
                    1.5
                ),
                alpha=False
            )

            image_bytes = pixmap.tobytes(
                "jpg"
            )

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

        client = Groq(
            api_key=api_key
        )

        content = [

            {
                "type": "text",

                "text": """
Read these resume pages carefully.

Extract all readable resume information
as plain text.

Combine information from all pages
in logical order.

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

Only extract resume information.

Do not invent missing information.
"""
            }

        ]

        content.extend(
            batch_images
        )

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

        text = (
            response
            .choices[0]
            .message
            .content
        )

        if text:

            page_texts.append(
                text.strip()
            )

    document.close()

    final_text = "\n\n".join(
        page_texts
    )

    if not final_text.strip():

        raise RuntimeError(
            "Could not read text from "
            "the scanned PDF."
        )

    return final_text


# =========================================================
# UNIVERSAL RESUME EXTRACTION
# =========================================================

def extract_resume_text(
    uploaded_file,
    api_key
):

    file_name = (
        uploaded_file
        .name
        .lower()
    )


    # -------------------------
    # PDF
    # -------------------------

    if file_name.endswith(".pdf"):

        text = extract_text_from_pdf(
            uploaded_file
        )

        # Text-based PDF
        if len(text.strip()) >= 50:

            return text

        # Image-based PDF
        return extract_text_from_scanned_pdf(
            uploaded_file,
            api_key
        )


    # -------------------------
    # TXT
    # -------------------------

    if file_name.endswith(".txt"):

        text = extract_text_from_txt(
            uploaded_file
        )

        if not text:

            raise RuntimeError(
                "The TXT resume is empty."
            )

        return text


    # -------------------------
    # JPG / PNG
    # -------------------------

    if file_name.endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    ):

        mime_type = "image/png"

        if file_name.endswith(
            (
                ".jpg",
                ".jpeg"
            )
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
# CREATE SINGLE CREWAI AGENT
# =========================================================

def create_career_agent(
    api_key
):

    career_llm = LLM(

        model="groq/openai/gpt-oss-120b",

        api_key=api_key,

        base_url=(
            "https://api.groq.com/openai/v1"
        ),

        temperature=0.2
    )


    agent = Agent(

        role="AI Career Coach",

        goal=(
            "Analyze a user's resume and target "
            "job description, identify relevant "
            "skills and skill gaps, recommend "
            "realistic career directions, and "
            "create a personalized learning roadmap."
        ),

        backstory=(
            "You are an experienced career coach "
            "and hiring advisor. You carefully "
            "compare a candidate's skills with "
            "the requirements of their target "
            "role and provide practical, realistic "
            "career advice."
        ),

        llm=career_llm,

        verbose=False,

        allow_delegation=False
    )

    return agent


# =========================================================
# CAREER ANALYSIS
# =========================================================

def analyze_career(
    resume_text,
    target_job,
    job_description,
    api_key
):

    agent = create_career_agent(
        api_key
    )


    task_description = f"""

Analyze this candidate for career coaching.

========================
CANDIDATE RESUME
========================

{resume_text}


========================
TARGET JOB
========================

{target_job}


========================
JOB DESCRIPTION
========================

{
    job_description
    if job_description.strip()
    else
    "No detailed job description was provided."
}


========================
REQUIRED ANALYSIS
========================

1. Candidate Profile

Briefly summarize the candidate's
current background.


2. Relevant Skills

Identify skills from the resume
that are relevant to the target job.

Separate technical and soft skills
where appropriate.


3. Skill Gaps

Identify important skills required
by the target role that appear to be
missing or insufficiently demonstrated.

Do not invent experience.


4. Career Directions

Recommend 2 to 3 realistic career
directions based on the candidate's
current skills and target role.

Explain why each direction fits.


5. Personalized Learning Roadmap

Create a roadmap divided into:

0–1 month
1–3 months
3–6 months

For each period include:

- Skills to learn
- Learning activities
- Practical projects


6. Resume Improvement Suggestions

Give 3 to 5 specific suggestions
for improving the resume for the
target role.


7. Final Recommendation

Give a short overall assessment.

Clearly identify the most important
skill the candidate should learn next.


========================
IMPORTANT RULES
========================

- Use only the provided resume and job information.
- Do not invent experience.
- Do not invent qualifications.
- Keep advice practical.
- Keep advice beginner-friendly.
- Use clear headings and bullet points.
"""


    task = Task(

        description=task_description,

        expected_output=(
            "A clear career coaching report "
            "containing candidate profile, "
            "relevant skills, skill gaps, "
            "career directions, personalized "
            "learning roadmap, resume suggestions, "
            "and final recommendation."
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
        """
        <div class="sidebar-brand">
            💼 AI Career Coach
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
            Build a clearer career direction
            from your resume, skills and
            target role.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption(
        "Powered by CrewAI + Groq"
    )


# =========================================================
# MAIN INPUT SECTION
# =========================================================

st.markdown(
    """
    <div class="section-title">
        Build Your Career Profile
    </div>

    <div class="section-description">
        Upload your resume and provide the
        role you want to pursue. Add a job
        description for a more targeted analysis.
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
    "Scanned and image-based resumes are supported."
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
        "Upload a text PDF, scanned PDF, "
        "TXT, JPG or PNG resume."
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
        """
        <div class="section-title">
            🎯 Target Role
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "What position are you interested in?"
    )

    target_job = st.text_input(

        "Target Job",

        placeholder=(
            "e.g. Junior Data Analyst"
        ),

        label_visibility="collapsed"
    )


with col2:

    st.markdown(
        """
        <div class="section-title">
            📝 Job Description
        </div>
        """,
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
# MAIN ANALYSIS
# =========================================================

if analyze_button:

    # -------------------------
    # API KEY
    # -------------------------

    api_key = get_api_key()

    if not api_key:

        st.error(
            "The AI service is currently unavailable. "
            "Please check the application configuration."
        )

        st.stop()


    # -------------------------
    # RESUME CHECK
    # -------------------------

    if uploaded_resume is None:

        st.warning(
            "Please upload your resume first."
        )

        st.stop()


    # -------------------------
    # TARGET JOB CHECK
    # -------------------------

    if not target_job.strip():

        st.warning(
            "Please enter your target job."
        )

        st.stop()


    # -------------------------
    # RESUME EXTRACTION
    # -------------------------

    try:

        file_name = (
            uploaded_resume
            .name
            .lower()
        )


        # PDF

        if file_name.endswith(".pdf"):

            with st.spinner(
                "📄 Reading your resume..."
            ):

                normal_text = (
                    extract_text_from_pdf(
                        uploaded_resume
                    )
                )


            if len(normal_text.strip()) >= 50:

                resume_text = normal_text

            else:

                with st.spinner(
                    "🔍 Image-based resume detected. "
                    "Reading it with AI vision..."
                ):

                    resume_text = (
                        extract_text_from_scanned_pdf(
                            uploaded_resume,
                            api_key
                        )
                    )


        # TXT

        elif file_name.endswith(".txt"):

            with st.spinner(
                "📄 Reading your resume..."
            ):

                resume_text = (
                    extract_text_from_txt(
                        uploaded_resume
                    )
                )


        # IMAGE

        else:

            with st.spinner(
                "🔍 Reading your image-based resume..."
            ):

                mime_type = "image/png"

                if file_name.endswith(
                    (
                        ".jpg",
                        ".jpeg"
                    )
                ):

                    mime_type = "image/jpeg"


                resume_text = (
                    extract_text_from_image(
                        uploaded_resume.getvalue(),
                        api_key,
                        mime_type
                    )
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


    # -------------------------
    # CAREER ANALYSIS
    # -------------------------

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
                    Personalized insights based on
                    your resume and target career.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            report
        )


    except Exception as error:

        st.error(
            "Something went wrong while generating "
            "your career analysis."
        )

        st.caption(
            f"Technical details: {str(error)}"
        )
