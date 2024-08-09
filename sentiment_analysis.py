import joblib
import streamlit as st
from nltk.tokenize import word_tokenize
import string
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    # Lowercase text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back into a string
    return ' '.join(tokens)

# Load the model and vectorizer
model_filename = 'sentiment_model.joblib'
vectorizer_filename = 'vectorizer.joblib'

model = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)

# Streamlit UI
st.title('IMDB Sentiment Analysis')
st.write('Enter a movie review and see its sentiment prediction.')

# Text input
user_input = st.text_area('Movie Review', '')

if st.button('Predict'):
    if user_input:
        # Preprocess the user input
        processed_input = preprocess_text(user_input)
        # Transform the input using the fitted vectorizer
        input_vector = vectorizer.transform([processed_input])
        # Predict sentiment
        prediction = model.predict(input_vector)[0]
        # Map numeric prediction to sentiment label
        sentiment = 'positive' if prediction == 'positive' else 'negative'
        st.write(f'Sentiment: {sentiment.capitalize()}')
    else:
        st.write('Please enter a review.')
