import streamlit as st
import joblib
import string
import re
import sys
import os
# NOTE: Ensure you have 'spam_detector_model.pkl' and 'tfidf_vectorizer.pkl' 
# in the same directory as this file.

# --- 0. Configuration and Initialization ---
# Set the Streamlit page configuration
st.set_page_config(
    page_title="Advanced Spam Detector Demo",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- 1. Load Saved Components ---
@st.cache_resource
def load_assets():
    """Loads the model and vectorizer only once when the app starts."""
    model_path = 'spam_detector_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error(
            "🚨 Required model files not found! "
            "Please ensure 'spam_detector_model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder."
        )
        st.stop()
        
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Failed to load assets: {e}")
        st.stop()

model, vectorizer = load_assets()

# --- 2. Preprocessing Function (MUST MATCH TRAINING) ---
punctuations = string.punctuation
remove_chars = punctuations + string.digits
stop_words = set() # Empty set to match the previous training script's execution

def preprocess_text(text):
    """Cleans and tokenizes the email text, matching the training pipeline."""
    text = text.lower()
    if text.startswith('subject:'):
        text = text[len('subject:'):]
    text = text.replace('\r\n', ' ')
    text = ''.join([char for char in text if char not in remove_chars])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and word]
    return ' '.join(tokens)


# --- 3. Prediction Function (Now returns confidence) ---
def predict_spam(email_text):
    """Processes text and returns a spam/ham prediction and confidence score."""
    processed_text = preprocess_text(email_text)
    
    # Vectorize and Predict
    vector = vectorizer.transform([processed_text])
    prediction = model.predict(vector)[0]
    
    # Get the prediction probability (confidence)
    confidence = model.predict_proba(vector)[0]
    
    return "SPAM" if prediction == 1 else "HAM", max(confidence)


# --- 4. Streamlit App Layout ---
st.title("🛡️ AI-Powered Spam Email Detector")
st.markdown("A **Multinomial Naive Bayes** model trained to classify emails as SPAM or HAM. Upload your code to GitHub and deploy it using Streamlit Cloud!")

# Input Area
email_input = st.text_area(
    "Paste the email content here:", 
    "Subject: Your Netflix payment failed. Update your billing details now or your account will be suspended. Click this link: http://malicious.link", 
    height=250
)

# Classification Button
if st.button("Classify Email", type="primary"):
    if email_input:
        with st.spinner('Analyzing email content...'):
            # Run prediction
            prediction, confidence = predict_spam(email_input)
            
            # Display Results in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Result")
                if prediction == "SPAM":
                    st.error(f"## 🚨 Predicted Class: {prediction}")
                    st.balloons()
                else:
                    st.success(f"## ✅ Predicted Class: {prediction}")
                
            with col2:
                st.subheader("Model Confidence")
                # Display confidence score with a progress bar
                confidence_percent = confidence * 100
                st.metric(
                    label="Confidence Score", 
                    value=f"{confidence_percent:.2f} %"
                )
                st.progress(confidence)

            # Optional: Show cleaned features and vectorization
            with st.expander("Show Preprocessed Text"):
                 st.code(preprocess_text(email_input), language='text')

    else:
        st.warning("Please enter some text to classify.")

st.markdown("---")
st.caption("Developed for a machine learning portfolio showcase using Scikit-learn and Streamlit.")
