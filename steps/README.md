# Medical Chatbot RAG Steps

This folder explains the project in separate steps.

The real application is merged into `app.py`, but these files show the same idea
piece by piece so you can explain the workflow to someone clearly.

## Step Order

1. `01_pdf_upload_step.py`
   - How the user uploads PDFs in Streamlit.

2. `02_pdf_text_extraction_step.py`
   - How PDF text is extracted page by page.

3. `03_text_chunking_step.py`
   - Why long PDF text is split into smaller chunks.

4. `04_vector_index_step.py`
   - How searchable vectors/indexes are created using TF-IDF.

5. `05_retrieval_step.py`
   - How the app finds the most relevant chunks for a user question.

6. `06_rag_answer_step.py`
   - How retrieved context and the user question are sent to the LLM.

7. `07_lab_report_mode_step.py`
   - How lab-report answers are handled differently from encyclopedia answers.

## Full Flow

```text
PDF upload
    -> PDF text extraction
    -> text chunking
    -> TF-IDF vector index
    -> retrieve relevant chunks
    -> build prompt
    -> Groq LLM answer
    -> show answer and sources in Streamlit
```

## Why Everything Is Merged In app.py

For a small Streamlit project, keeping everything in one file is easy to run and
demo. These separate files are for explanation and learning. Later, the same
logic can be converted into real Python modules if the project becomes larger.
