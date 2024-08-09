import joblib
import streamlit as st
from nltk.tokenize import word_tokenize
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

model_filename = 'sentiment_model.joblib'
vectorizer_filename = 'vectorizer.joblib'

model = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)

st.title('IMDB Sentiment Analysis')
st.write('Enter a movie review and see its sentiment prediction.')
user_input = st.text_area('Movie Review', '')

if st.button('Predict'):
    if user_input:
        processed_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([processed_input])
        
        prediction = model.predict(input_vector)[0]
        
        sentiment = 'positive' if prediction == 'positive' else 'negative'
        st.write(f'Sentiment: {sentiment.capitalize()}')
    else:
        st.write('Please enter a review.')
