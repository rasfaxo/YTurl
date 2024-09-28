from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import logging
import requests
import pandas as pd
app = Flask(__name__)
CORS(app)
YOUTUBE_API_KEY = 'AIzaSyDl9evGCdGM0RhxrAuc3kTUOsPMRf1k2uw'

# Mengatur logging
logging.basicConfig(level=logging.INFO)

def search_youtube(query, max_results=5):
    url = f'https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key={YOUTUBE_API_KEY}&maxResults={max_results}'
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json().get('items', [])
        return [{'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}", 'title': item['snippet']['title']} for item in result if item['id']['kind'] == 'youtube#video']
    return []

# Memuat model dan vectorizer yang telah dilatih
try:
    model, vectorizer, data = joblib.load('models/model.pkl')
except Exception as e:
    logging.error(f"Kesalahan saat memuat model dan vectorizer: {e}")
    model, vectorizer, data = None, None, None

def is_valid_youtube_url(url):
    youtube_regex = re.compile(
        r'^(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$'
    )
    return youtube_regex.match(url)

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    if model is None or vectorizer is None or data is None:
        return jsonify({"error": "Model tidak dimuat"}), 500
    url = request.args.get('url')
    if not url or not is_valid_youtube_url(url):
        return jsonify({"error": "URL YouTube tidak valid"}), 400

    # Mencari judul dari URL yang dimasukkan
    input_title = data[data['url'] == url]['title'].values
    if len(input_title) == 0:
        # Jika URL tidak ditemukan, mencari judul video di YouTube
        logging.info(f"URL tidak ditemukan dalam dataset, mencari judul video di YouTube.")
        video_info = search_youtube(url)
        if not video_info:
            return jsonify({"error": "Video tidak ditemukan di YouTube"}), 404
        video_title = video_info[0]['title']
        input_title = [video_title]
        logging.info(f"Judul video ditemukan dari YouTube: {video_title}")
    else:
        input_title = input_title[:1]

    # Mendapatkan URL dan judul video rekomendasi dari YouTube
    recommendations = search_youtube(input_title[0], max_results=5)
    
    logging.info(f"Rekomendasi: {recommendations}")
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)