"""
Step 4: Vector/Search Index Creation

Purpose:
Convert text chunks into a searchable index.

Technology used:
- scikit-learn `TfidfVectorizer`
- SciPy `hstack`

Why:
When a patient asks a question, the app needs to quickly find which PDF chunks
are relevant. TF-IDF is fast and works locally, which is useful for large PDFs.

Note:
This project originally tried HuggingFace embeddings + FAISS, but the large
medical encyclopedia took too long to index during upload. TF-IDF is simpler and
faster for this Streamlit app.
"""

from dataclasses import dataclass

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfIndex:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    matrix: object
    chunks: list
    page_count: int
    chunk_count: int


def create_tfidf_index(chunks, page_count: int):
    chunk_texts = [chunk.page_content for chunk in chunks]

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        stop_words="english",
        max_features=90000,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=45000,
    )

    word_matrix = word_vectorizer.fit_transform(chunk_texts)
    char_matrix = char_vectorizer.fit_transform(chunk_texts)
    matrix = hstack([word_matrix, char_matrix]).tocsr()

    return TfidfIndex(
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        matrix=matrix,
        chunks=chunks,
        page_count=page_count,
        chunk_count=len(chunks),
    )
