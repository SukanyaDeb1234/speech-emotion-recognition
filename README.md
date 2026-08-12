---
title: Speech Emotion Recognition
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# 🎙️ Speech Emotion Recognition

An end-to-end speech emotion recognition system using
Whisper Large-v3 encoder representations and an MLP classifier.

## Model

Whisper Large-v3
↓
Mean Pooling
↓
StandardScaler
↓
MLP Classifier

## Dataset

CREMA-D

## Emotions

- Anger
- Disgust
- Fear
- Happy
- Neutral
- Sad

## Results

Validation Accuracy: **76.44%**

Speaker-independent held-out test accuracy: **72.47%**

## Features

- WAV audio upload
- Emotion prediction
- Model confidence
- Emotion probability distribution
- Inference-time measurement

## Deployment

Streamlit application deployed using Hugging Face Spaces.