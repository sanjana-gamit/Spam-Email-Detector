import streamlit as st
import joblib
import string
import re
# NOTE: If you used NLTK stop words in your train.py, you must import them here too.
# For simplicity, we are using the same basic preprocessing function as in the training script.

# --- 1. Load Saved Components ---
# These files should be placed in the same directory as this app.py file
try:
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Error: Model or Vectorizer file not found. Ensure 'spam_detector_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop()


# --- 2. Preprocessing Function (Must Match Training Script) ---
punctuations = string.punctuation
remove_chars = punctuations + string.digits
stop_words = set() # Matches the training script's constraint

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


# --- 3. Prediction Function ---
def predict_spam(email_text):
    """Processes text and returns a spam/ham prediction."""
    # Step 1: Preprocess the raw text
    processed_text = preprocess_text(email_text)

    # Step 2: Vectorize the processed text
    vector = vectorizer.transform([processed_text])

    # Step 3: Predict the class
    prediction = model.predict(vector)[0]
    
    # Step 4: Get probability for confidence (optional but good for a demo)
    probability = model.predict_proba(vector)[0]
    confidence = max(probability)

    return "SPAM" if prediction == 1 else "HAM", confidence


# --- 4. Streamlit App Layout ---
st.set_page_config(page_title="Spam Email Detector Demo", layout="wide")
st.title("🤖 AI-Powered Spam Email Detector")
st.markdown("Enter an email below to see if our Multinomial Naive Bayes model classifies it as SPAM or HAM (Legitimate).")

# Text input for the user
email_input = st.text_area(
    "Paste the email body here:", 
    "Subject: Your account has been temporarily disabled. Click this link to confirm your details and restore access.", 
    height=200
)

# Run prediction when button is clicked
if st.button("Classify Email", type="primary"):
    if email_input:
        with st.spinner('Analyzing email...'):
            prediction, confidence = predict_spam(email_input)
            
            st.subheader("Classification Result:")

            if prediction == "SPAM":
                st.markdown(f"## 🚨 Predicted Class: <span style='color:red;'>{prediction}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"## ✅ Predicted Class: <span style='color:green;'>{prediction}</span>", unsafe_allow_html=True)
            
            st.info(f"Confidence: {confidence*100:.2f}%")
            
            # Optional: Show a brief explanation of the clean text
            with st.expander("Show Cleaned Text (Features)"):
                 st.code(preprocess_text(email_input), language='text')

    else:
        st.warning("Please enter some text to classify.")

st.markdown("---")
st.caption("Project built using Python, Scikit-learn, and Streamlit. [Link to GitHub Repo]")
