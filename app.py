# app.py
import streamlit as st
from model import search_books, create_recommender

# Load CSS
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# UI
st.markdown('<div class="title">📚 CARI BUKU FAVORITMU DISINI</div>', unsafe_allow_html=True)

search_query = st.text_input('🔍 Cari buku berdasarkan judul, penulis, atau genre:', placeholder='Contoh: Romance')

if st.button('Cari Buku'):
    with st.spinner('🔎 Sedang mencari buku...'):
        books_df = search_books(search_query)
        if not books_df.empty:
            st.session_state.books_df = books_df
            st.session_state.cosine_sim = create_recommender(books_df)
            st.success(f"📘 Ditemukan {len(books_df)} buku!")

            st.subheader("📚 Daftar Buku yang Ditemukan")
            for _, row in books_df.iterrows():
                with st.expander(f"📗 {row['title']} - {row['authors']}"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if row['thumbnail']:
                            st.image(row['thumbnail'], width=130)
                    with col2:
                        st.markdown(f"""
                        <div class='book-box'>
                        <strong>Judul:</strong> {row['title']}  \n
                        <strong>Penulis:</strong> {row['authors']}  \n
                        <strong>Genre:</strong> {row['genres']}  \n
                        <strong>Tahun:</strong> {row['published_year']}  \n
                        <strong>Rating:</strong> {row['average_rating']} ⭐  \n
                        <strong>Deskripsi:</strong> {row['description'][:300]}...
                        </div>
                        """, unsafe_allow_html=True)

            st.subheader(f"✨ Rekomendasi Berdasarkan Genre")
            try:
                cosine_sim = st.session_state.cosine_sim
                idx = 0
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
                book_indices = [i[0] for i in sim_scores]

                cols = st.columns(5)
                for col, i in zip(cols, book_indices):
                    book = books_df.iloc[i]
                    with col:
                        st.image(book['thumbnail'], width=100)
                        st.markdown(f"**{book['title']}**")
                        st.caption(f"👤 {book['authors']}")
                        st.caption(f"⭐ {book['average_rating']}")
            except Exception as e:
                st.error(f"❌ Gagal membuat rekomendasi: {str(e)}")
        else:
            st.warning("🙁 Tidak menemukan buku yang cocok.")

# Rekomendasi berdasarkan pilihan
if 'books_df' in st.session_state:
    books_df = st.session_state.books_df
    cosine_sim = st.session_state.cosine_sim

    selected_book = st.selectbox('📌 Pilih buku favorit Anda:', books_df['title'].values)

    if st.button('✨ Dapatkan Rekomendasi'):
        with st.spinner('🔄 Menghasilkan rekomendasi...'):
            try:
                idx = books_df[books_df['title'] == selected_book].index[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
                book_indices = [i[0] for i in sim_scores]

                st.subheader(f"📖 Rekomendasi untuk: *{selected_book}*")
                cols = st.columns(5)
                for col, i in zip(cols, book_indices):
                    book = books_df.iloc[i]
                    with col:
                        st.image(book['thumbnail'], width=100)
                        st.markdown(f"**{book['title']}**")
                        st.caption(f"👤 {book['authors']}")
                        st.caption(f"⭐ {book['average_rating']}")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat membuat rekomendasi: {str(e)}")
