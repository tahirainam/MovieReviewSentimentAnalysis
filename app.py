import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing function (same as training)
def preprocess(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analysis")
review = st.text_area("Enter a movie review:", height=200)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = preprocess(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = "👍 Positive" if prediction == 1 else "👎 Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
