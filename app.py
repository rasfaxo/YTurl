from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import logging
import requests

app = Flask(__name__)
CORS(app)
YOUTUBE_API_KEY = 'AIzaSyDl9evGCdGM0RhxrAuc3kTUOsPMRf1k2uw'

# Setup logging
logging.basicConfig(level=logging.INFO)

def search_youtube(query):
    url = f'https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key={YOUTUBE_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json().get('items', [])
        if result:
            return result[0]['snippet']['title']
    return None

# Load trained model and vectorizer
try:
    model, vectorizer, data = joblib.load('models/model.pkl')
except Exception as e:
    logging.error(f"Error loading model and vectorizer: {e}")
    model, vectorizer, data = None, None, None

def is_valid_youtube_url(url):
    youtube_regex = re.compile(
        r'^(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$'
    )
    return youtube_regex.match(url)

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    if model is None or vectorizer is None or data is None:
        return jsonify({"error": "Model not loaded"}), 500
    url = request.args.get('url')
    if not url or not is_valid_youtube_url(url):
        return jsonify({"error": "Invalid YouTube URL"}), 400

    # Find the title of the input URL
    input_title = data[data['url'] == url]['title'].values
    if len(input_title) == 0:
        # If URL not found, use a default title or the first title in the dataset
        input_title = data['title'].values[:1]
        logging.info(f"URL not found in dataset, using default title: {input_title[0]}")
    else:
        input_title = input_title[:1]

    # Transform the input title to vector
    input_vector = vectorizer.transform(input_title)

    # Determine the number of neighbors to use
    n_neighbors = min(5, len(data))

    try:
        # Find the nearest neighbors
        distances, indices = model.kneighbors(input_vector, n_neighbors=n_neighbors)
    except ValueError as e:
        logging.error(f"Error finding nearest neighbors: {e}")
        return jsonify({"error": str(e)}), 500

    # Get the recommended video URLs and titles
    recommendations = data.iloc[indices[0]].to_dict(orient='records')
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)