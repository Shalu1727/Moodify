# EmotiFy: Multimodal Emotion Detection System

EmotiFy is a **Multimodal Emotion Detection System** that combines **facial expression analysis** and **speech emotion recognition** to accurately detect human emotions in real-time.  
It uses **deep learning models** for both image and audio inputs, delivering a robust, cross-modal emotion classification.

---

## 📌 Features
- 🎭 **Facial Emotion Recognition** using **ResNet50 CNN** on the FER2013 dataset.
- 🎤 **Speech Emotion Recognition** using **Bidirectional LSTM** on MFCC features from the TESS dataset.
- 🔄 **Multimodal Fusion** of predictions from both modalities for improved accuracy.
- 🌐 **Interactive Web UI** built with **Streamlit** for easy testing.
- ⚡ **Real-time prediction** from webcam and microphone inputs.
- 📂 **Pre-trained models** for quick setup.

---

## 🛠 Tech Stack
- **Programming Language:** Python
- **Deep Learning Frameworks:** TensorFlow / Keras
- **Data Processing:** NumPy, Pandas, Librosa, OpenCV
- **Web Framework:** Streamlit
- **Visualization:** Matplotlib, Seaborn

---

## 📂 Datasets Used
- **[FER2013](https://www.kaggle.com/datasets/msambare/fer2013)** – Facial expressions dataset with 7 emotion classes.
- **[TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)** – Emotional speech dataset with audio samples in different emotions.

---

## ⚙️ How It Works
1. **Face Detection:** Capture frames from the webcam and detect faces using OpenCV.
2. **Feature Extraction (Image):** Process face images and classify emotions with the trained CNN model.
3. **Feature Extraction (Audio):** Extract **MFCC features** from audio and classify emotions using the BiLSTM model.
4. **Fusion:** Combine predictions from both models to get the final emotion output.
5. **UI Display:** Show emotion results in real-time via Streamlit.

git clone https://github.com/your-username/emotify.git
cd emotify
