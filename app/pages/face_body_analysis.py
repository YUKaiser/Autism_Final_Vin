import streamlit as st
import cv2
import av
import hashlib
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from facial_predict import load_facial_models, predict_facial
from session_reset import render_reset_button


@st.cache_resource
def load_torch_facial_models():
    return load_facial_models(PROJECT_ROOT / "models" / "facial")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Face / Body Analysis | NeuroMind",
    page_icon="🎥",
    layout="wide"
)


# =========================================================
# CAMERA PROCESSOR
# =========================================================

class FaceCameraProcessor(VideoProcessorBase):

    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        self.latest_frame = img.copy()

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


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

/* Camera heading */
.camera-title {
    font-size: 26px;
    font-weight: 700;
    color: #292d3d;
}

.camera-description {
    font-size: 17px;
    color: #68748b;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:28px;
            font-weight:750;
            color:#292d3d;
            margin-bottom:35px;
        ">
        🧠 NeuroMind
        </div>
        """,
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
        icon="🎥"
    )

    st.page_link(
        "pages/results.py",
        label="Results",
        icon="📊"
    )

    render_reset_button()


# =========================================================
# SESSION STATE
# =========================================================

if "captured_face" not in st.session_state:
    st.session_state.captured_face = None

if "face_camera_error" not in st.session_state:
    st.session_state.face_camera_error = None

if "face_result" not in st.session_state:
    st.session_state.face_result = None

if "face_asd_probability" not in st.session_state:
    st.session_state.face_asd_probability = None

if "face_prediction" not in st.session_state:
    st.session_state.face_prediction = None

if "face_image_data" not in st.session_state:
    st.session_state.face_image_data = None

if "face_image_signature" not in st.session_state:
    st.session_state.face_image_signature = None

if "face_input_signature" not in st.session_state:
    st.session_state.face_input_signature = None

if "face" not in st.session_state:
    st.session_state.face = None

if "fs" not in st.session_state:
    st.session_state.fs = 0


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="page-title">Face / Body Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Upload a photo or capture one using the live camera'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# IMAGE SOURCE AND AGE GATE
# =========================================================

image_source = st.radio(
    "Choose an image source",
    ["Upload a photo", "Use camera"],
    horizontal=True,
)

declared_age = st.number_input(
    "Child's age",
    min_value=1,
    max_value=120,
    value=15,
    step=1,
    help="Facial analysis is intended only for children under 15.",
)
age_is_supported = declared_age < 15
st.caption(
    "Age is self-reported; these facial checkpoints do not estimate or "
    "verify age from an image."
)
if not age_is_supported:
    st.warning("Facial analysis is available only for a declared age under 15.")
    st.session_state.face_result = None
    st.session_state.face_asd_probability = None
    st.session_state.face_prediction = None
    st.session_state.face_input_signature = None
    st.session_state.face = None
    st.session_state.fs = 0

image_for_analysis = None
image_signature = None

if image_source == "Upload a photo":
    uploaded_face = st.file_uploader(
        "Upload a child's face photo",
        type=["jpg", "jpeg", "png"],
        help="Choose a clear JPG or PNG image from this device.",
        key="face_photo_upload_page",
    )
    if uploaded_face is not None:
        image_bytes = uploaded_face.getvalue()
        image_signature = hashlib.sha256(image_bytes).hexdigest()
        try:
            image_for_analysis = np.asarray(
                Image.open(uploaded_face).convert("RGB")
            )
        except (OSError, ValueError) as error:
            st.error(f"Could not read the uploaded image: {error}")
        else:
            st.session_state.face_image_data = image_for_analysis
            st.session_state.face_image_signature = image_signature
            st.markdown("### Photo Preview")
            st.image(
                image_for_analysis,
                caption="Uploaded photo",
                use_container_width=True,
            )
    elif st.session_state.face_image_data is not None:
        image_for_analysis = st.session_state.face_image_data
        image_signature = st.session_state.face_image_signature
        st.markdown("### Photo Preview")
        st.image(
            image_for_analysis,
            caption="Uploaded photo",
            use_container_width=True,
        )
else:
    st.markdown(
        '<div class="camera-title">Live Camera</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="camera-description">'
        'Allow camera access and position the face clearly inside the camera view.'
        '</div>',
        unsafe_allow_html=True
    )

    ctx = webrtc_streamer(
        key="face-camera",
        video_processor_factory=FaceCameraProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    if st.button("📸 Capture Image", use_container_width=True):
        if (
            ctx.video_processor is not None
            and ctx.video_processor.latest_frame is not None
        ):
            frame_bgr = ctx.video_processor.latest_frame.copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.session_state.captured_face = frame_rgb
            st.session_state.face_image_data = frame_rgb
            st.session_state.face_image_signature = hashlib.sha256(
                frame_rgb.tobytes()
            ).hexdigest()
            st.session_state.face_camera_error = None
            st.session_state.face_result = None
            st.session_state.face_asd_probability = None
            st.session_state.face_prediction = None
            st.session_state.face_input_signature = None
            st.session_state.face = None
            st.session_state.fs = 0
            st.rerun()
        else:
            st.session_state.face_camera_error = (
                "No camera frame is available yet. "
                "Start the camera and try again."
            )

    if st.session_state.face_camera_error:
        st.warning(st.session_state.face_camera_error)

    image_for_analysis = st.session_state.captured_face
    if image_for_analysis is not None:
        image_signature = hashlib.sha256(
            np.asarray(image_for_analysis).tobytes()
        ).hexdigest()
        st.markdown("### Captured Image")
        st.image(
            image_for_analysis,
            caption="Captured Face Image",
            use_container_width=True
        )


if image_signature != st.session_state.get("face_input_signature"):
    st.session_state.face_result = None
    st.session_state.face_asd_probability = None
    st.session_state.face_prediction = None
    st.session_state.face = None
    st.session_state.fs = 0


if image_for_analysis is not None and age_is_supported:
    if st.button(
        "🧠 Analyze Face →",
        use_container_width=True
    ):
        try:
            prediction = predict_facial(
                image_for_analysis,
                load_torch_facial_models(),
            )
        except (FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            st.error(f"Facial model inference failed: {error}")
        else:
            st.session_state.face_result = prediction["label"]
            st.session_state.face = prediction["class_id"]
            st.session_state.face_asd_probability = prediction["asd_probability"]
            st.session_state.face_prediction = prediction
            st.session_state.face_input_signature = image_signature
            st.session_state.fs = int(round(prediction["confidence"] * 100))
            st.success(
                f"{prediction['label']} "
                f"(ASD likelihood: {prediction['asd_probability']:.2%}; "
                f"model confidence: {prediction['confidence']:.2%})"
            )
            st.markdown("#### Emotion model output")
            st.write({
                name: f"{probability:.1%}"
                for name, probability in prediction["emotions"].items()
            })
            if prediction["is_ood"]:
                st.warning(
                    "This image differs substantially from the facial model's "
                    "training data. Treat its prediction with extra caution."
                )
            st.caption(
                "Experimental screening output only; this is not a clinical diagnosis."
            )

if image_source == "Use camera" and image_for_analysis is not None:
    if st.button("🔄 Capture Again", use_container_width=True):
        st.session_state.captured_face = None
        st.session_state.face_camera_error = None
        st.session_state.face_result = None
        st.session_state.face_asd_probability = None
        st.session_state.face_prediction = None
        st.session_state.face_input_signature = None
        st.session_state.face_image_data = None
        st.session_state.face_image_signature = None
        st.session_state.face = None
        st.session_state.fs = 0

        st.rerun()