"""
Step 3: Text Chunking

Purpose:
Break long PDF text into smaller chunks.

Technology used:
- LangChain `RecursiveCharacterTextSplitter`

Why:
Large PDFs can contain thousands of words. We cannot send the whole PDF to the
LLM every time. Smaller chunks make search faster and let us send only the most
relevant parts to the model.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=260,
    )

    return splitter.split_documents(documents)
