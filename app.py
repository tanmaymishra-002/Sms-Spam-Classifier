import streamlit as st
import pickle
import string
import nltk

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

stop_words = set(stopwords.words("english"))


def transform_text(text):
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    y = []

    # Keep only alphanumeric tokens
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load trained model and TF-IDF vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")


if st.button("Predict"):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. TF-IDF vectorization
    vector_input = tfidf.transform([transformed_sms])

    # 3. Convert sparse matrix to dense
    result = model.predict(vector_input.toarray())[0]

    # 4. Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")