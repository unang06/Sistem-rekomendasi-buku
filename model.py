def search_books(query, max_results=40):
    params = {
        'q': query,
        'key': API_KEY,
        'maxResults': max_results
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()

        # Debug log setelah response berhasil
        print(f"[DEBUG] Status: {response.status_code}")
        print(f"[DEBUG] Full URL: {response.url}")
        data = response.json()
        print(f"[DEBUG] JSON keys: {list(data.keys())}")
        print(f"[DEBUG] Total items: {data.get('totalItems', 0)}")

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
        print(f"❌ Exception in search_books: {str(e)}")
        return pd.DataFrame()
