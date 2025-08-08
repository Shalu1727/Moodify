import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import soundfile as sf
import librosa
from transformers import pipeline
import tempfile
import os
from streamlit_mic_recorder import mic_recorder # For microphone input
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration # For webcam input
import av # Required by streamlit-webrtc

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Multimodal Emotion Recognition")

# --- Model Loading (Cached for performance) ---
# Use st.cache_resource for non-data objects like models
@st.cache_resource
def load_fer_model():
    """Loads the facial emotion recognition model (uses DeepFace)."""
    # DeepFace builds models automatically on first use.
    # You can optionally specify a model: models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
    # You can optionally specify a detector backend: detector_backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    # Pre-load a model to avoid delay on first use (optional)
    try:
        print("Loading DeepFace model...")
        # Analyze a dummy image to trigger model loading
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False)
        print("DeepFace model loaded successfully.")
        # Return True or some identifier if needed, but primary goal is pre-loading
        return True
    except Exception as e:
        st.error(f"Error loading DeepFace model: {e}")
        return False

@st.cache_resource
def load_ser_model():
    """Loads the speech emotion recognition model from Hugging Face."""
    try:
        print("Loading Speech Emotion Recognition model...")
        # Using a popular SER model from Hugging Face Hub
        # You can choose other models available on the Hub
        # Example: 'superb/hubert-large-superb-er' (requires 'transformers', 'torch', 'torchaudio', 'librosa')
        # Example: 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'
        ser_pipeline = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        print("SER model loaded successfully.")
        return ser_pipeline
    except Exception as e:
        st.error(f"Error loading SER model: {e}. Make sure you have torch/torchaudio installed.")
        st.warning("SER functionality will be limited.")
        return None

# --- Load Models ---
fer_model_loaded = load_fer_model()
ser_model_pipeline = load_ser_model()

# --- Helper Functions ---
def analyze_facial_emotion(image_np):
    """Analyzes facial emotion from a numpy image array."""
    if not fer_model_loaded:
        return "FER model not loaded.", None

    try:
        # DeepFace expects BGR format, OpenCV reads in BGR by default
        result = DeepFace.analyze(image_np, actions=['emotion'], enforce_detection=True) # Set enforce_detection=True

        # Check if result is a list (multiple faces) or dict (single face)
        if isinstance(result, list):
            # Handle multiple faces - analyze the first detected face
            if len(result) > 0:
                emotions = result[0]['emotion']
                dominant_emotion = result[0]['dominant_emotion']
                face_region = result[0]['region'] # {'x': int, 'y': int, 'w': int, 'h': int}
                return dominant_emotion, emotions, face_region
            else:
                return "No face detected.", None, None
        elif isinstance(result, dict):
             # Handle single face
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            face_region = result['region']
            return dominant_emotion, emotions, face_region
        else:
             return "Unexpected result format from DeepFace.", None, None

    except ValueError as ve:
        # Specific error when no face is detected and enforce_detection=True
        if "Face could not be detected" in str(ve):
            return "No face detected.", None, None
        else:
            # Other ValueErrors
            st.error(f"An unexpected error occurred during face analysis: {ve}")
            return "Analysis error.", None, None
    except Exception as e:
        st.error(f"An error occurred during facial emotion analysis: {e}")
        return "Analysis error.", None, None


def analyze_speech_emotion(audio_path):
    """Analyzes speech emotion from an audio file path."""
    if ser_model_pipeline is None:
        return "SER model not loaded.", None

    try:
        # Load audio file - librosa ensures consistent sampling rate if needed by model later
        # Or let the pipeline handle loading:
        # speech, sr = librosa.load(audio_path, sr=16000) # Ensure target SR matches model requirement if needed

        # Use the pipeline directly
        results = ser_model_pipeline(audio_path, top_k=5) # Get top 5 predictions

        if results:
            dominant_emotion = results[0]['label']
            # Extract scores for display
            emotion_scores = {r['label']: f"{r['score']:.2%}" for r in results}
            return dominant_emotion, emotion_scores
        else:
            return "Could not analyze audio.", None

    except Exception as e:
        st.error(f"An error occurred during speech emotion analysis: {e}")
        return "Analysis error.", None


def draw_face_box(image, region, emotion):
    """Draws bounding box and emotion label on the image."""
    if region:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        # Draw rectangle (BGR color format for OpenCV)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green box
        # Put text
        label = f"{emotion}"
        # Calculate text size for background rectangle
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # Put a background rectangle
        cv2.rectangle(image, (x, y - label_height - baseline), (x + label_width, y), (0, 255, 0), cv2.FILLED)
        # Put text on top of background
        cv2.putText(image, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Black text
    return image

# --- Streamlit UI ---
st.title("🎭🗣️ Multimodal Emotion Recognition")
st.write("Detect emotions from facial expressions or voice recordings.")
st.info("Models are loaded on startup. Please wait a moment for the first analysis.")

tab1, tab2 = st.tabs(["😊 Facial Emotion Recognition (FER)", "🎤 Speech Emotion Recognition (SER)"])

# --- Facial Emotion Recognition Tab ---
with tab1:
    st.header("Facial Emotion Recognition")
    fer_source = st.radio("Select Input Source for Face:", ("Upload Image", "Use Webcam"), key="fer_source", horizontal=True)

    uploaded_image = None
    webcam_frame = None

    if fer_source == "Upload Image":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            # Read the image bytes
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            # Decode the image using OpenCV
            image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(image_np, channels="BGR", caption="Uploaded Image")

            if st.button("Analyze Facial Emotion", key="analyze_face_upload"):
                if fer_model_loaded:
                    with st.spinner("Analyzing face..."):
                        dominant_emotion, emotions, face_region = analyze_facial_emotion(image_np)

                    st.subheader("Analysis Result:")
                    if dominant_emotion not in ["No face detected.", "Analysis error."]:
                        st.success(f"**Dominant Emotion:** {dominant_emotion}")
                        st.write("**Emotion Scores:**")
                        st.json(emotions) # Display all emotion scores nicely

                        # Draw bounding box and display image with box
                        if face_region:
                           image_with_box = draw_face_box(image_np.copy(), face_region, dominant_emotion)
                           st.image(image_with_box, channels="BGR", caption="Image with Detected Emotion")
                    else:
                        st.warning(dominant_emotion) # Show "No face detected" or "Analysis error"
                else:
                    st.error("FER model could not be loaded. Cannot analyze.")

    elif fer_source == "Use Webcam":
        st.info("Click 'START' to access the webcam. Analysis is performed on snapshots.")

        # Using streamlit-webrtc
        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.frame = None

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                self.frame = img # Store the latest frame
                # Can optionally perform analysis here continuously, but that's CPU intensive
                # It's better to analyze on demand via a button below
                return img # Display the stream

        webrtc_ctx = webrtc_streamer(
            key="webcam",
            video_transformer_factory=VideoTransformer,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False}, # Only video
            async_processing=True,
        )

        if webrtc_ctx.video_transformer:
             # Add a button to capture and analyze the current frame
            if st.button("Capture & Analyze Webcam Frame", key="analyze_webcam"):
                 # Access the last frame captured by the transformer
                captured_frame = webrtc_ctx.video_transformer.frame
                if captured_frame is not None:
                    st.image(captured_frame, channels="BGR", caption="Captured Frame")
                    if fer_model_loaded:
                        with st.spinner("Analyzing face..."):
                            dominant_emotion, emotions, face_region = analyze_facial_emotion(captured_frame)

                        st.subheader("Analysis Result:")
                        if dominant_emotion not in ["No face detected.", "Analysis error."]:
                            st.success(f"**Dominant Emotion:** {dominant_emotion}")
                            st.write("**Emotion Scores:**")
                            st.json(emotions)

                            # Draw bounding box and display image with box
                            if face_region:
                                image_with_box = draw_face_box(captured_frame.copy(), face_region, dominant_emotion)
                                st.image(image_with_box, channels="BGR", caption="Frame with Detected Emotion")
                        else:
                            st.warning(dominant_emotion)
                    else:
                        st.error("FER model could not be loaded. Cannot analyze.")
                else:
                    st.warning("Could not capture frame from webcam. Is it active?")
        else:
            st.warning("Webcam stream not available. Please click 'START'.")


# --- Speech Emotion Recognition Tab ---
with tab2:
    st.header("Speech Emotion Recognition")
    ser_source = st.radio("Select Input Source for Speech:", ("Upload Audio File", "Record Audio"), key="ser_source", horizontal=True)

    uploaded_audio = None
    recorded_audio_bytes = None

    if ser_source == "Upload Audio File":
        uploaded_audio = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "ogg", "flac"])
        if uploaded_audio:
            st.audio(uploaded_audio, format=uploaded_audio.type)

            if st.button("Analyze Speech Emotion", key="analyze_speech_upload"):
                if ser_model_pipeline:
                    # Save uploaded file temporarily to pass path to pipeline
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmpfile:
                        tmpfile.write(uploaded_audio.getvalue())
                        audio_path = tmpfile.name

                    with st.spinner("Analyzing speech..."):
                        dominant_emotion, emotion_scores = analyze_speech_emotion(audio_path)

                    st.subheader("Analysis Result:")
                    if dominant_emotion not in ["Could not analyze audio.", "Analysis error."]:
                         st.success(f"**Detected Emotion:** {dominant_emotion}")
                         if emotion_scores:
                            st.write("**Emotion Probabilities:**")
                            st.json(emotion_scores) # Display scores
                    else:
                         st.warning(dominant_emotion) # Show error/warning

                    # Clean up temporary file
                    os.remove(audio_path)
                else:
                    st.error("SER model could not be loaded. Cannot analyze.")

    elif ser_source == "Record Audio":
        st.info("Click the microphone icon below to record audio (allow microphone access when prompted). Max duration ~30s.")

        # Use streamlit-mic-recorder component
        audio_info = mic_recorder(
            start_prompt="Start Recording 🎤",
            stop_prompt="Stop Recording 🛑",
            just_once=False, # Allow multiple recordings
            use_container_width=False,
            format="wav", # Output format
            callback=None, # No callback needed here, we use the returned value
            key='mic_recorder'
        )

        if audio_info and audio_info['bytes']:
            st.success("Recording finished!")
            # Get the audio bytes
            recorded_audio_bytes = audio_info['bytes']
            # Display the recorded audio
            st.audio(recorded_audio_bytes, format="audio/wav")

            if st.button("Analyze Recorded Speech", key="analyze_speech_record"):
                if ser_model_pipeline:
                     # Save recorded bytes to a temporary WAV file
                     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                         tmpfile.write(recorded_audio_bytes)
                         audio_path = tmpfile.name

                     with st.spinner("Analyzing speech..."):
                         dominant_emotion, emotion_scores = analyze_speech_emotion(audio_path)

                     st.subheader("Analysis Result:")
                     if dominant_emotion not in ["Could not analyze audio.", "Analysis error."]:
                         st.success(f"**Detected Emotion:** {dominant_emotion}")
                         if emotion_scores:
                            st.write("**Emotion Probabilities:**")
                            st.json(emotion_scores) # Display scores
                     else:
                         st.warning(dominant_emotion) # Show error/warning

                     # Clean up temporary file
                     os.remove(audio_path)
                else:
                    st.error("SER model could not be loaded. Cannot analyze.")

# --- Footer / Info ---
st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io/), [DeepFace](https://github.com/serengil/deepface), and [Hugging Face Transformers](https://huggingface.co/transformers).")