import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import string
import nltk
# NOTE: If running this locally, you may need to install/download NLTK resources:
# import nltk
# nltk.download('stopwords')

# --- 1. Data Loading and Initial Cleaning ---
file_path = "spam_ham_dataset.csv.zip/spam_ham_dataset.csv"
df = pd.read_csv(file_path)

# Drop redundant index and string label column
df.drop(columns=['Unnamed: 0', 'label'], inplace=True)
df.rename(columns={'label_num': 'spam'}, inplace=True)


# --- 2. NLP Preprocessing Function ---
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except Exception:
    # Using an empty set if NLTK stopwords can't be loaded (common in some envs)
    stop_words = set()

punctuations = string.punctuation
# Combine punctuation and digits into a list of characters to remove
remove_chars = punctuations + string.digits

def preprocess_text(text):
    """Cleans and tokenizes the email text."""
    # Convert to lowercase
    text = text.lower()

    # Remove the common "Subject:" tag and newline characters
    if text.startswith('subject:'):
        text = text[len('subject:'):]
    text = text.replace('\r\n', ' ')

    # Remove punctuation and digits
    text = ''.join([char for char in text if char not in remove_chars])

    # Tokenization and Stop word removal
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and word]

    # Rejoin tokens into a cleaned string
    return ' '.join(tokens)

# Apply preprocessing to create the clean_text column
df['clean_text'] = df['text'].apply(preprocess_text)


# --- 3. Split Data ---
X = df['clean_text']
y = df['spam']
# Splitting data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 4. Feature Extraction (TF-IDF Vectorization) ---
# TF-IDF converts text into numerical features, weighting words by importance
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# --- 5. Model Training (Multinomial Naive Bayes) ---
# Multinomial Naive Bayes is excellent for text classification
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# --- 6. Prediction and Evaluation ---
y_pred = model.predict(X_test_tfidf)

print("--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- 7. Sample Prediction for Demo ---
def predict_spam(email_text, model, vectorizer):
    """Predicts if a new email is spam or ham."""
    processed_text = preprocess_text(email_text)
    vector = vectorizer.transform([processed_text])
    prediction = model.predict(vector)[0]
    return "SPAM (1)" if prediction == 1 else "HAM (0)"

sample_email = "Claim your free money now! Click the link below to win a prize. Call 123456789."
prediction = predict_spam(sample_email, model, tfidf_vectorizer)

print("\n--- Sample Prediction ---")
print(f"Email: '{sample_email}'")
print(f"Prediction: {prediction}")
