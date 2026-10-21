import sys
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mri_predict import load_mri_model, predict_mri, validate_mri_volume
from session_reset import render_reset_button


@st.cache_resource
def load_torch_mri_model():
    return load_mri_model(
        PROJECT_ROOT / "models" / "better_3dcnn_asd.pth"
    )


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MRI Analysis | NeuroMind",
    page_icon="🧠",
    layout="wide"
)

if "mri_volume" not in st.session_state:
    st.session_state.mri_volume = None

if "mri_preview_image" not in st.session_state:
    st.session_state.mri_preview_image = None


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2.5rem;
    padding-left: 4rem;
    padding-right: 4rem;
    padding-bottom: 3rem;
}

/* Hide Streamlit automatic navigation */
[data-testid="stSidebarNav"] {
    display: none;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f3f5f9;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
    padding-left: 1.3rem;
    padding-right: 1.3rem;
}

/* Page title */
.page-title {
    font-size: 42px;
    font-weight: 750;
    color: #292d3d;
    margin-bottom: 5px;
}

.page-subtitle {
    font-size: 19px;
    color: #68748b;
    margin-bottom: 30px;
}

/* Upload heading */
.upload-heading {
    font-size: 26px;
    font-weight: 700;
    color: #292d3d;
}

.upload-description {
    font-size: 17px;
    color: #68748b;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div style="font-size:28px; font-weight:750; '
        'color:#292d3d; margin-bottom:35px;">'
        '🧠 NeuroMind'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    st.page_link(
        "main.py",
        label="Dashboard",
        icon="🏠"
    )

    st.page_link(
        "pages/behavioral_assessment.py",
        label="Behavioral Assessment",
        icon="📋"
    )

    st.page_link(
        "pages/mri_analysis.py",
        label="MRI Analysis",
        icon="🧠"
    )

    st.page_link(
        "pages/face_body_analysis.py",
        label="Face / Body Analysis",
        icon="👤"
    )

    st.page_link(
        "pages/results.py",
        label="Results",
        icon="📊"
    )

    render_reset_button()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="page-title">MRI Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Upload a preprocessed MRI volume for analysis'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD SECTION
# =========================================================

with st.container(border=True):

    st.markdown(
        '<div class="upload-heading">Upload Preprocessed MRI Volume</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="upload-description">'
        'Select the preprocessed .npy volume used by the Colab model '
        '(shape: 128 × 160 × 160).'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a preprocessed MRI volume",
        type=["npy"]
    )


# =========================================================
# PREVIEW
# =========================================================

if uploaded_file is not None:
    st.session_state.mri_volume = None
    st.session_state.mri_preview_image = None
    try:
        uploaded_volume = validate_mri_volume(
            np.load(uploaded_file, allow_pickle=False)
        )
    except (OSError, TypeError, ValueError) as error:
        st.error(f"Invalid MRI volume: {error}")
    else:
        st.session_state.mri_volume = uploaded_volume
        middle_slice = uploaded_volume[uploaded_volume.shape[0] // 2]
        lower, upper = np.percentile(middle_slice, (1, 99))
        if upper > lower:
            st.session_state.mri_preview_image = np.clip(
                (middle_slice - lower) / (upper - lower) * 255,
                0,
                255,
            ).astype(np.uint8)
        else:
            st.session_state.mri_preview_image = np.zeros(
                middle_slice.shape, dtype=np.uint8
            )

mri_volume = st.session_state.mri_volume
if mri_volume is not None:
        st.markdown("### MRI Scan Preview")
        st.image(
            st.session_state.mri_preview_image,
            caption="Middle slice of the preprocessed MRI volume",
            use_container_width=True,
            clamp=True,
        )

        if st.button("🧠 Analyze MRI →", use_container_width=True):
            try:
                prediction = predict_mri(
                    mri_volume,
                    load_torch_mri_model(),
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                st.error(f"MRI model inference failed: {error}")
            else:
                confidence = prediction["confidence"]
                st.session_state.mri = prediction["class_id"]
                st.session_state.ms = int(round(confidence * 100))
                st.session_state.mri_result = prediction["label"]
                st.success(
                    f"{prediction['label']} "
                    f"(confidence: {confidence:.2%})"
                )