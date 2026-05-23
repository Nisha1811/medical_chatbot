"""
Step 5: Retrieval

Purpose:
Find the most relevant PDF chunks for the user's question.

Technology used:
- TF-IDF matrix multiplication
- Word and character vectorizers

Why:
RAG works by retrieving only the useful text instead of sending the entire PDF
to the LLM. This reduces hallucination and keeps answers grounded in the report
or medical book.
"""

import re

from scipy.sparse import hstack


def build_search_query(user_query: str) -> str:
    words = re.findall(r"[A-Za-z]+", user_query.lower())
    generic_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "about",
        "tell",
        "me",
        "explain",
        "purpose",
        "uses",
        "use",
        "types",
        "type",
        "list",
    }

    focused_words = [word for word in words if word not in generic_words]
    return " ".join(focused_words) or user_query


def retrieve_context(index, query: str, k: int = 10):
    search_query = build_search_query(query)
    word_query = index.word_vectorizer.transform([search_query])
    char_query = index.char_vectorizer.transform([search_query])
    query_vector = hstack([word_query, char_query]).tocsr()

    scores = (index.matrix @ query_vector.T).toarray().ravel()
    ranked_indexes = scores.argsort()[-k:][::-1]
    selected_indexes = set()

    for chunk_index in ranked_indexes:
        if scores[chunk_index] <= 0:
            continue

        selected_indexes.add(chunk_index)

        # Include neighboring chunks for fuller context.
        if chunk_index > 0:
            selected_indexes.add(chunk_index - 1)
        if chunk_index + 1 < len(index.chunks):
            selected_indexes.add(chunk_index + 1)

    return [index.chunks[i] for i in sorted(selected_indexes)]
