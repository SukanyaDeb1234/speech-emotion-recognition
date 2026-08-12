import streamlit as st
from pathlib import Path
import sys

# Make src importable when running from project root
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import SpeechEmotionRecognizer


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)


# ============================================================
# Title
# ============================================================

st.title("🎙️ Speech Emotion Recognition")

st.write(
    "Upload a WAV file and the system will predict the "
    "emotion, confidence score, and inference time."
)


# ============================================================
# Load model once
# ============================================================

@st.cache_resource
def load_model():

    return SpeechEmotionRecognizer()


with st.spinner("Loading Whisper Large-v3..."):
    recognizer = load_model()


# ============================================================
# File upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a WAV file",
    type=["wav"]
)


# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    # Temporary file
    temp_dir = Path("temp")

    temp_dir.mkdir(
        exist_ok=True
    )

    temp_path = (
        temp_dir /
        uploaded_file.name
    )

    with open(
        temp_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    st.divider()

    if st.button(
        "Analyze Emotion",
        type="primary"
    ):

        try:

            with st.spinner(
                "Analyzing speech..."
            ):

                result = recognizer.predict(
                    temp_path
                )

            # ================================================
            # Main prediction
            # ================================================

            emotion = result["emotion"].upper()

            confidence = result["confidence"]

            inference_time = result[
                "total_time"
            ]

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Emotion",
                    emotion
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with col3:

                st.metric(
                    "Inference Time",
                    f"{inference_time:.2f} s"
                )

            # ================================================
            # Probability distribution
            # ================================================

            st.subheader(
                "Emotion Probabilities"
            )

            probabilities = (
                result["probabilities"]
            )

            for emotion_name, probability in (
                probabilities.items()
            ):

                st.write(
                    f"**{emotion_name.capitalize()}** "
                    f"{probability:.2f}%"
                )

                st.progress(
                    min(
                        probability / 100.0,
                        1.0
                    )
                )

            # ================================================
            # Timing details
            # ================================================

            st.subheader(
                "Inference Breakdown"
            )

            st.write(
                f"Preprocessing: "
                f"{result['preprocessing_time']:.3f} s"
            )

            st.write(
                f"Whisper encoding: "
                f"{result['whisper_time']:.3f} s"
            )

            st.write(
                f"Classifier: "
                f"{result['classifier_time']:.3f} s"
            )

            st.write(
                f"Total: "
                f"{result['total_time']:.3f} s"
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

        finally:

            if temp_path.exists():

                temp_path.unlink()


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Model: Whisper Large-v3 encoder + MLP | "
    "Dataset: CREMA-D | "
    "Held-out test accuracy: 72.47%"
)