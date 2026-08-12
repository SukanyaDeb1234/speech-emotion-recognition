import time
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torch.nn as nn

from transformers import (
    AutoProcessor,
    WhisperModel,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/whisper-large-v3"

BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    BASE_DIR / "models" / "best_whisper_mlp.pt"
)

SCALER_PATH = (
    BASE_DIR / "models" / "whisper_scaler.pkl"
)

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
]


# ============================================================
# WHISPER MLP
# Must match the architecture used during training
# ============================================================

class WhisperMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# SPEECH EMOTION RECOGNIZER
# ============================================================

class SpeechEmotionRecognizer:

    def __init__(self):

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("Device:", self.device)

        # ----------------------------------------------------
        # Validate required files
        # ----------------------------------------------------

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n{CHECKPOINT_PATH}"
            )

        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler not found:\n{SCALER_PATH}"
            )

        # ----------------------------------------------------
        # Whisper processor
        # ----------------------------------------------------

        print("Loading Whisper processor...")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

        # ----------------------------------------------------
        # Whisper Large-v3
        # ----------------------------------------------------

        print("Loading Whisper Large-v3...")

        if self.device.type == "cuda":

            self.whisper = WhisperModel.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)

        else:

            self.whisper = WhisperModel.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)

        self.whisper.eval()

        # ----------------------------------------------------
        # Emotion classifier
        # ----------------------------------------------------

        print("Loading emotion classifier...")

        self.classifier = WhisperMLP().to(
            self.device
        )

        self.classifier.load_state_dict(
            torch.load(
                CHECKPOINT_PATH,
                map_location=self.device
            )
        )

        self.classifier.eval()

        # ----------------------------------------------------
        # StandardScaler
        # ----------------------------------------------------

        print("Loading scaler...")

        self.scaler = joblib.load(
            SCALER_PATH
        )

        print("Model ready.")

    # ========================================================
    # AUDIO PREPROCESSING
    # ========================================================

    def load_audio(self, audio_path):

        audio_path = Path(
            str(audio_path).strip().strip('"').strip("'")
        )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found:\n{audio_path}"
            )

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        waveform, sample_rate = sf.read(
            audio_path,
            dtype="float32"
        )

        # ----------------------------------------------------
        # Convert stereo → mono
        # ----------------------------------------------------

        if waveform.ndim == 2:

            waveform = waveform.mean(
                axis=1
            )

        # ----------------------------------------------------
        # Convert numpy → torch
        # ----------------------------------------------------

        waveform = torch.from_numpy(
            np.asarray(waveform)
        ).float()

        # ----------------------------------------------------
        # Resample to 16 kHz
        # Whisper requires 16 kHz
        # ----------------------------------------------------

        if sample_rate != 16000:

            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                orig_freq=sample_rate,
                new_freq=16000
            ).squeeze(0)

            sample_rate = 16000

        print(
            f"Audio sample rate: {sample_rate} Hz"
        )

        print(
            f"Audio samples: {waveform.numel()}"
        )

        return waveform, sample_rate

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, audio_path):

        total_start = time.perf_counter()

        # ----------------------------------------------------
        # Load + preprocess audio
        # ----------------------------------------------------

        preprocessing_start = time.perf_counter()

        waveform, sample_rate = self.load_audio(
            audio_path
        )

        preprocessing_time = (
            time.perf_counter()
            - preprocessing_start
        )

        # ----------------------------------------------------
        # Whisper feature extraction
        # ----------------------------------------------------

        whisper_start = time.perf_counter()

        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(
            self.device
        )

        # Correct dtype for Whisper
        if self.device.type == "cuda":
            input_features = input_features.half()
        else:
            input_features = input_features.float()

        # ----------------------------------------------------
        # Whisper encoder
        # ----------------------------------------------------

        with torch.no_grad():

            encoder_outputs = self.whisper.encoder(
                input_features=input_features
            )

            hidden = (
                encoder_outputs
                .last_hidden_state
            )

            # [1, T, 1280]
            # ↓
            # [1280]
            embedding = (
                hidden
                .mean(dim=1)
                .squeeze(0)
                .float()
                .cpu()
                .numpy()
            )

        whisper_time = (
            time.perf_counter()
            - whisper_start
        )

        # ----------------------------------------------------
        # StandardScaler
        # IMPORTANT:
        # scaler was fitted ONLY on training data
        # ----------------------------------------------------

        scaled_embedding = self.scaler.transform(
            embedding.reshape(1, -1)
        )

        feature_tensor = torch.tensor(
            scaled_embedding,
            dtype=torch.float32,
            device=self.device
        )

        # ----------------------------------------------------
        # MLP prediction
        # ----------------------------------------------------

        classifier_start = time.perf_counter()

        with torch.no_grad():

            logits = self.classifier(
                feature_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        classifier_time = (
            time.perf_counter()
            - classifier_start
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        total_time = (
            time.perf_counter()
            - total_start
        )

        predicted_emotion = EMOTIONS[
            prediction.item()
        ]

        confidence_percent = (
            confidence.item() * 100
        )

        # Probability of every emotion
        emotion_probabilities = {
            emotion: float(prob) * 100
            for emotion, prob in zip(
                EMOTIONS,
                probabilities[0].cpu().numpy()
            )
        }

        return {
            "emotion": predicted_emotion,
            "confidence": confidence_percent,
            "probabilities": emotion_probabilities,
            "preprocessing_time": preprocessing_time,
            "whisper_time": whisper_time,
            "classifier_time": classifier_time,
            "total_time": total_time,
        }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    print()
    print("=" * 55)
    print("      SPEECH EMOTION RECOGNITION")
    print("=" * 55)
    print()

    try:

        recognizer = SpeechEmotionRecognizer()

        print()

        audio_file = input(
            "Enter path to WAV file: "
        ).strip()

        result = recognizer.predict(
            audio_file
        )

        print()
        print("=" * 55)
        print("                 RESULT")
        print("=" * 55)

        print(
            f"Emotion           : "
            f"{result['emotion'].upper()}"
        )

        print(
            f"Confidence        : "
            f"{result['confidence']:.2f}%"
        )

        print(
            f"Preprocessing     : "
            f"{result['preprocessing_time']:.3f} s"
        )

        print(
            f"Whisper encoding  : "
            f"{result['whisper_time']:.3f} s"
        )

        print(
            f"Classifier         : "
            f"{result['classifier_time']:.3f} s"
        )

        print(
            f"Total inference   : "
            f"{result['total_time']:.3f} s"
        )

        print()
        print("Emotion probabilities:")

        for emotion, probability in (
            result["probabilities"].items()
        ):

            print(
                f"  {emotion:8s}: "
                f"{probability:6.2f}%"
            )

        print("=" * 55)

    except Exception as e:

        print()
        print("ERROR:")
        print(e)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()