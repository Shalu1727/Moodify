import streamlit as st
import torch
import torchaudio
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load model
try:
    from model import MultimodalFusion  # Your custom multimodal model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = MultimodalFusion()
    model.load_state_dict(torch.load(
        "/Users/prithvipandey/Documents/dataset/Emotion/multimodal_model.pth",
        map_location=device
    ))
    model.to(device)
    model.eval()
except Exception as e:
    st.error(f"🔴 Error loading model: {e}")

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Streamlit UI
st.title("🎭 Multimodal Emotion Detection")
st.markdown("Upload a **face image** and **voice sample** to detect the emotion.")

img_file = st.file_uploader("📷 Upload face image", type=["jpg", "jpeg", "png"])
audio_file = st.file_uploader("🎙️ Upload audio file (WAV only)", type=["wav"])

if st.button("🔍 Predict") and img_file and audio_file:
    try:
        # 🔹 Process image
        image = Image.open(img_file).convert("RGB")
        img_tensor = image_transform(image).unsqueeze(0).to(device)  # [1, 3, 48, 48]

        # 🔹 Process audio
        waveform, sample_rate = torchaudio.load(audio_file)

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Extract MFCC features
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=40
        )
        mfcc = mfcc_transform(waveform)  # [1, 40, time]
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # [time, 40]

        # Pad or trim to fixed length (e.g. 10 time steps)
        desired_len = 10
        if mfcc.shape[0] < desired_len:
            pad = desired_len - mfcc.shape[0]
            mfcc = torch.cat([mfcc, torch.zeros(pad, mfcc.shape[1])])
        else:
            mfcc = mfcc[:desired_len, :]

        mfcc = mfcc.unsqueeze(0).to(device)  # [1, 10, 40]

        # 🔹 Make prediction
        with torch.no_grad():
            output = model(img_tensor, mfcc)
            predicted = torch.argmax(output, dim=1).item()
            probs = torch.softmax(output, dim=1).cpu().numpy().flatten()

        # 🔹 Class labels
        classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # 🔹 Display results
        st.success(f"Predicted Emotion: **{classes[predicted].upper()}**")
        st.subheader("📊 Prediction Probabilities")
        st.bar_chart(probs)

        for i, cls in enumerate(classes):
            st.write(f"**{cls.capitalize()}**: {probs[i]:.4f}")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
