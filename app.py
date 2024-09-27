from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)

# Load trained model and vectorizer
model, vectorizer, data = joblib.load('models/model.pkl')

def is_valid_youtube_url(url):
    youtube_regex = re.compile(
        r'^(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$'
    )
    return youtube_regex.match(url)

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    url = request.args.get('url')
    if not url or not is_valid_youtube_url(url):
        return jsonify([]), 400

    # Find the title of the input URL
    input_title = data[data['url'] == url]['title'].values
    if len(input_title) == 0:
        return jsonify([])

    # Transform the input title to vector
    input_vector = vectorizer.transform(input_title)

    # Determine the number of neighbors to use
    n_neighbors = min(5, len(data))

    try:
        # Find the nearest neighbors
        distances, indices = model.kneighbors(input_vector, n_neighbors=n_neighbors)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    # Get the recommended video URLs and titles
    recommendations = data.iloc[indices[0]].to_dict(orient='records')
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)