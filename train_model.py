import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# Load dataset
data = pd.read_csv('data/videos.csv')

# Vectorize the video titles
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['title'])

# Train a NearestNeighbors model
model = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X)

# Save the model and vectorizer
joblib.dump((model, vectorizer, data), 'models/model.pkl')