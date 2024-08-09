import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

data = pd.read_csv('/mnt/data/IMDB Dataset.csv')

print("Preprocessing reviews...")
data['review'] = data['review'].progress_apply(preprocess_text)

X = data['review']
y = data['sentiment']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Vectorizing text data...")
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training model...")
model = LogisticRegression()
with tqdm(total=len(X_train_vec)) as pbar:
    for i in range(len(X_train_vec)):
        model.partial_fit(X_train_vec[i:i+1], [y_train.iloc[i]], classes=['positive', 'negative'])
        pbar.update(1)

model_filename = 'sentiment_model.joblib'
vectorizer_filename = 'vectorizer.joblib'

joblib.dump(model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)

print(f"Model saved as {model_filename}")
print(f"Vectorizer saved as {vectorizer_filename}")

print("Evaluating model...")
y_pred = model.predict(X_test_vec)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
