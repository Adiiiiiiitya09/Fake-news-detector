import streamlit as st
import pandas as pd
import joblib
import re
import string
import os
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_assets():
    if os.path.exists('model.joblib') and os.path.exists('vectorizer.joblib'):
        model = joblib.load('model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        return model, vectorizer
    return None, None

model, vectorizer = load_assets()

# --- Preprocessing Function ---
def wordopt(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)    
    return text

# --- Sidebar ---
st.sidebar.title("Dashboard & Analytics")
st.sidebar.info("This app uses Machine Learning to detect fake news based on the article content.")

if st.sidebar.checkbox("Show Training Analytics"):
    st.sidebar.subheader("Model Performance")
    if os.path.exists('model_accuracies.png'):
        st.sidebar.image('model_accuracies.png', caption="Accuracy Comparison")
    
    st.sidebar.subheader("Dataset Distribution")
    if os.path.exists('news_categories.png'):
        st.sidebar.image('news_categories.png', caption="News Categories")
    if os.path.exists('real_fake_count.png'):
        st.sidebar.image('real_fake_count.png', caption="True vs Fake Count")

# --- Main App Interface ---
st.title("📰 Fake News Detection System")
st.markdown("---")

if model is None or vectorizer is None:
    st.warning("⚠️ Model files not found. Please run `train.py` first to train and save the model.")
else:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Analyze Article")
        news_text = st.text_area("Paste the news article text here:", height=300, placeholder="Enter news content...")
        
        if st.button("Predict Authenticity"):
            if news_text.strip() == "":
                st.error("Please enter some text to analyze.")
            else:
                # Preprocess and Predict
                cleaned_text = wordopt(news_text)
                vectorized_text = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized_text)[0]
                
                # Calculate Confidence Score (if model supports it)
                confidence = None
                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(vectorized_text)
                        confidence = max(proba[0]) * 100
                except:
                    pass

                # Show Result
                st.markdown("### Prediction Result")
                if prediction == 1:
                    conf_text = f" (Confidence: {confidence:.2f}%)" if confidence else ""
                    st.success(f"✅ This news article appears to be **REAL**.{conf_text}")
                    st.balloons()
                else:
                    conf_text = f" (Confidence: {confidence:.2f}%)" if confidence else ""
                    st.error(f"🚨 This news article appears to be **FAKE**.{conf_text}")

    with col2:
        st.subheader("How it works")
        st.write("""
        1. **Preprocessing**: The text is cleaned by removing punctuation, links, and special characters.
        2. **Vectorization**: The cleaned text is converted into numbers using TF-IDF.
        3. **Classification**: A trained Machine Learning model analyzes the patterns in the words.
        4. **Output**: The model classifies the article as 'True' or 'Fake'.
        """)
        
        st.info("Note: This is a prototype based on the ISOT Fake News Dataset.")

# --- Footer ---
st.markdown("---")
st.caption("Developed based on the 'Fake News Detection Using Machine Learning' Project Report.")
