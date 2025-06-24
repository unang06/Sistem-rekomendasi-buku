import requests
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# --- SETUP NLTK (AMAN UNTUK STREAMLIT CLOUD) ---
nltk.data.path.append("/tmp")

try:
    stop_words = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords', download_dir="/tmp")
    stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus simbol & angka
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

# --- GOOGLE BOOKS API CONFIG ---
API_KEY = "AIzaSyCWtre-ogfgAL55nyLnNCnLmeo4cGUH7O8"
BASE_URL = "https://www.googleapis.com/books/v1/volumes"

def search_books(query, max_results=40):
    params = {
        'q': query,
        'key': API_KEY,
        'maxResults': max_results
    }
    try:
        response = requests.get(BASE_URL, params=params)  # harus di sini
        response.raise_for_status()

        # ✨ Debug print di bawah ini harus berada **setelah** response berhasil
        print(f"[DEBUG] QUERY: {query}")
        print(f"[DEBUG] STATUS: {response.status_code}")
        print(f"[DEBUG] URL: {response.url}")

        data = response.json()
        print(f"[DEBUG] JSON KEYS: {list(data.keys())}")
        print(f"[DEBUG] TOTAL_ITEMS: {data.get('totalItems')}")

        books = []
        for item in data.get('items', []):
            volume_info = item.get('volumeInfo', {})
            book = {
                'id': item.get('id'),
                'title': volume_info.get('title', 'No title'),
                'authors': ', '.join(volume_info.get('authors', ['Unknown'])),
                'genres': ', '.join(volume_info.get('categories', ['Unknown'])),
                'description': volume_info.get('description', 'No description'),
                'published_year': volume_info.get('publishedDate', '')[:4],
                'average_rating': volume_info.get('averageRating', 0),
                'thumbnail': volume_info.get('imageLinks', {}).get('thumbnail', '')
            }
            books.append(book)
        return pd.DataFrame(books)

    except Exception as e:
        print(f"❌ Exception in search_books: {e}")
        return pd.DataFrame()

def create_recommender(df):
    if df.empty:
        return None

    df['features'] = df['genres'].fillna('') + ' ' + df['description'].fillna('')
    df['features'] = df['features'].apply(preprocess_text)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim
