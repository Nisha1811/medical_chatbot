"""
Step 2: PDF Text Extraction

Purpose:
Read the uploaded PDF and extract text from each page.

Technology used:
- LangChain `PyPDFLoader`
- Python `tempfile`

Why:
The LLM cannot directly understand a PDF file. First, the PDF must be converted
into plain text. PyPDFLoader returns one document object per page.
"""

import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def extract_pdf_documents(file_payloads: tuple[tuple[str, bytes], ...]):
    documents = []
    temp_paths: list[str] = []

    try:
        for file_name, file_bytes in file_payloads:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                temp_paths.append(tmp.name)

            loader = PyPDFLoader(temp_paths[-1])
            page_documents = loader.load()

            for doc in page_documents:
                if doc.page_content.strip():
                    doc.metadata["source"] = file_name
                    documents.append(doc)
    finally:
        for temp_path in temp_paths:
            Path(temp_path).unlink(missing_ok=True)

    return documents
