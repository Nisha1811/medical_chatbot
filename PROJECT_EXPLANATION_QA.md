# Medical Chatbot Project Explanation and Q&A

This document explains the project step by step. It is useful when presenting
the project to a teacher, interviewer, client, or teammate.

## Project Summary

This project is a medical RAG chatbot built with Streamlit. It allows users to
upload medical PDFs or lab reports and ask questions based on the uploaded
document. The app retrieves relevant text from the PDF and sends it to a Groq
LLM to generate a patient-friendly answer.

RAG means Retrieval-Augmented Generation.

In simple words:

```text
PDF -> extract text -> split into chunks -> search relevant chunks -> send to LLM -> answer
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web app UI |
| LangChain | Prompt templates, PDF loader utilities, text splitting |
| PyPDFLoader | Reads text from PDF files |
| RecursiveCharacterTextSplitter | Splits long text into smaller chunks |
| scikit-learn TF-IDF | Creates local searchable text index |
| SciPy sparse matrix | Efficiently stores and combines TF-IDF vectors |
| Groq ChatGroq | Connects to Groq Llama model for answer generation |
| HTML/CSS | Improves Streamlit UI design |

---

# Step 1: PDF Upload

## What Happens

The user uploads a PDF file from the Streamlit sidebar. The PDF can be:

- medical encyclopedia
- blood report
- sugar test report
- HbA1c report
- lipid profile
- thyroid report
- any readable medical PDF

The uploaded file is converted into bytes using:

```python
file.getvalue()
```

## Technology Used

Streamlit:

```python
st.file_uploader()
```

## Why We Use It

Streamlit provides a simple file upload box in the browser. It makes it easy for
patients or users to upload reports without building a separate frontend.

## Common Questions and Answers

**Q1. Why do we upload PDF files?**

A. The chatbot needs a document as knowledge source. By uploading a PDF, the app
can answer based on that document instead of giving random general answers.

**Q2. Can the app read any PDF?**

A. It can read PDFs that contain selectable text. If the PDF is a scanned image,
OCR is needed, which is not currently added.

**Q3. Can multiple PDFs be uploaded?**

A. Yes, the app supports multiple PDFs. It combines readable text from all
selected PDFs.

**Q4. Why not directly send the PDF to the LLM?**

A. LLMs cannot directly process large PDF files efficiently. The PDF must first
be converted into text and searched.

---

# Step 2: PDF Text Extraction

## What Happens

The uploaded PDF is saved temporarily and read page by page. Each page becomes a
document object with text and metadata such as page number and source file name.

## Technology Used

LangChain PDF loader:

```python
PyPDFLoader
```

Temporary file handling:

```python
tempfile.NamedTemporaryFile()
```

## Why We Use It

The app needs plain text from the PDF before it can search or answer questions.
`PyPDFLoader` makes it easy to extract text from each page.

## Common Questions and Answers

**Q1. What is PyPDFLoader?**

A. PyPDFLoader is a LangChain document loader that extracts text from PDF files.

**Q2. Why do we save the uploaded file temporarily?**

A. PyPDFLoader expects a file path. Streamlit gives uploaded files as bytes, so
we save them temporarily and then read them.

**Q3. What happens if a PDF page has no text?**

A. The app skips empty pages. If the whole PDF has no readable text, the app
shows a warning.

**Q4. Why do we keep metadata like source and page?**

A. Metadata helps show sources later, such as which PDF and page were used for
the answer.

---

# Step 3: Text Chunking

## What Happens

The extracted PDF text is split into smaller chunks.

Example:

```text
Full PDF text -> chunk 1 -> chunk 2 -> chunk 3 -> ...
```

## Technology Used

LangChain:

```python
RecursiveCharacterTextSplitter
```

## Why We Use It

PDFs can be very large. The app should not send the full PDF to the LLM for every
question. Instead, it splits the document into smaller chunks and searches only
the relevant chunks.

## Common Questions and Answers

**Q1. What is chunking?**

A. Chunking means dividing a large text into smaller parts.

**Q2. Why is chunking important in RAG?**

A. It makes search faster and allows the app to send only relevant information to
the LLM.

**Q3. What is chunk overlap?**

A. Chunk overlap means some text is repeated between neighboring chunks. This
prevents important information from being cut off between two chunks.

**Q4. What chunk size is used?**

A. The app uses chunks around 1500 characters with overlap. This gives enough
context while keeping search manageable.

---

# Step 4: Search Index / Vector Index

## What Happens

Each text chunk is converted into a numeric representation using TF-IDF. These
numeric representations are stored in memory as a searchable index.

## Technology Used

scikit-learn:

```python
TfidfVectorizer
```

SciPy:

```python
hstack
```

## Why We Use It

The app needs to quickly find which chunks match the user question. TF-IDF works
like a local search engine.

## What Is TF-IDF?

TF-IDF means Term Frequency-Inverse Document Frequency.

Simple meaning:

```text
TF-IDF = keyword importance scoring
```

It gives higher importance to useful words and lower importance to very common
words.

## Common Questions and Answers

**Q1. Which vector database is used?**

A. No external vector database is used. The project uses an in-memory TF-IDF
search index with scikit-learn and SciPy.

**Q2. Is TF-IDF an embedding model?**

A. No. TF-IDF is a traditional keyword-based search method. It is not an AI
embedding model.

**Q3. Why not use FAISS?**

A. FAISS with HuggingFace embeddings was slower for the large medical
encyclopedia PDF on this machine. TF-IDF was faster and simpler for local
testing.

**Q4. Is TF-IDF good enough?**

A. For many document search tasks, yes. It is fast and works well when user
questions contain words that appear in the document. For advanced semantic
search, embeddings can be added later.

**Q5. Where is the index stored?**

A. It is stored in memory while the Streamlit app is running. It is recreated
when PDFs are processed.

---

# Step 5: Retrieval

## What Happens

When the user asks a question, the app converts the question into a TF-IDF query
vector and compares it with all document chunks. The most relevant chunks are
selected.

## Technology Used

- TF-IDF query vector
- Matrix multiplication
- Ranking by score

## Why We Use It

The LLM should receive only the text that is relevant to the question. Retrieval
reduces unnecessary context and improves answer accuracy.

## Common Questions and Answers

**Q1. What is retrieval?**

A. Retrieval means searching the PDF and selecting the most relevant text chunks
for the user question.

**Q2. How does the app know which chunk is relevant?**

A. It compares the words in the question with the words in each chunk using
TF-IDF scores.

**Q3. Why does the app include neighboring chunks?**

A. Sometimes important explanation starts before or continues after the matched
chunk. Neighboring chunks provide fuller context.

**Q4. What happens if nothing matches?**

A. The app responds: "I couldn't find this in the documents."

---

# Step 6: Prompt Building

## What Happens

The app creates a prompt containing:

- instructions for the LLM
- retrieved PDF context
- user question

## Technology Used

LangChain:

```python
PromptTemplate
```

## Why We Use It

Prompt templates keep the LLM behavior controlled. They tell the model to answer
only from the document and avoid unsafe medical advice.

## Common Questions and Answers

**Q1. What is a prompt?**

A. A prompt is the instruction sent to the LLM.

**Q2. Why do we use different prompts for encyclopedia and lab reports?**

A. A medical book and a patient lab report need different answer styles. Lab
reports need values, units, reference ranges, and high/low explanation.

**Q3. How does the app reduce hallucination?**

A. The prompt tells the model to use only the retrieved context and say when the
answer is not found.

**Q4. Can the model still make mistakes?**

A. Yes. That is why the app includes medical safety instructions and tells users
to consult a qualified doctor.

---

# Step 7: Groq LLM Answer Generation

## What Happens

The final prompt is sent to Groq's Llama model. The model generates a readable
answer for the user.

## Technology Used

Groq:

```python
ChatGroq
```

Model:

```text
llama-3.1-8b-instant
```

## Why We Use It

Groq provides fast LLM responses. The Llama model can convert retrieved medical
text into easy-to-understand answers.

## Common Questions and Answers

**Q1. What does Groq do in this project?**

A. Groq runs the LLM that generates the final answer.

**Q2. Does Groq search the PDF?**

A. No. The local TF-IDF retriever searches the PDF. Groq only writes the final
answer from the retrieved text.

**Q3. Why use temperature 0.2?**

A. A low temperature makes answers more focused and less random, which is better
for medical explanations.

**Q4. Why set max tokens?**

A. It allows longer answers instead of very short replies.

---

# Step 8: Lab Report Mode

## What Happens

In Lab Report mode, the app uses the same PDF processing pipeline but a different
prompt. It explains patient report values carefully.

It can answer questions like:

- Which values are abnormal?
- Explain this report in simple words.
- Is fasting sugar high?
- What does HbA1c mean?
- What should I ask my doctor?

## Technology Used

- Same PDF extraction
- Same chunking
- Same TF-IDF retrieval
- Special lab report prompt

## Why We Use It

Lab reports contain values, units, and reference ranges. The app must explain
those clearly without diagnosing or prescribing medicine.

## Common Questions and Answers

**Q1. Can the app diagnose a patient?**

A. No. It can explain report values and suggest doctor follow-up, but it should
not diagnose.

**Q2. Can it tell high and low values?**

A. Yes, if the report contains reference ranges or high/low flags.

**Q3. What if reference range is missing?**

A. The app should say that the range is not shown in the report.

**Q4. Can it work with blood sugar reports?**

A. Yes. It can explain fasting sugar, random sugar, HbA1c, and similar values if
they are present in the PDF.

**Q5. Can it read scanned lab reports?**

A. Not reliably yet. OCR would be needed for scanned image PDFs.

---

# Step 9: Streamlit UI

## What Happens

Streamlit displays:

- sidebar settings
- mode selector
- PDF upload box
- status cards
- chat messages
- source pages

## Technology Used

Streamlit plus custom CSS.

## Why We Use It

Streamlit makes it easy to build a Python web app quickly without separate HTML,
CSS, JavaScript, or backend routing.

## Common Questions and Answers

**Q1. Why use Streamlit?**

A. It is simple, fast, and ideal for machine learning or AI demos.

**Q2. Is Streamlit frontend or backend?**

A. It acts as both for this project. Python code controls the UI and app logic.

**Q3. Why use custom CSS?**

A. CSS improves the design and makes the UI look more professional.

**Q4. Can this be converted into a full web app later?**

A. Yes. Later, the frontend can be built with React and the backend with FastAPI.

---

# app.py Explanation

`app.py` is the main file where all steps are merged into one working Streamlit
application.

## What app.py Contains

1. Imports
2. Page configuration
3. CSS styling
4. API key loading
5. Groq LLM setup
6. PDF processing function
7. TF-IDF index creation
8. Retrieval function
9. Prompt templates
10. Sidebar mode selection
11. PDF upload / data folder selection
12. Chat interface
13. Answer generation
14. Source display

## Why Everything Is in app.py

For a small demo project, one file is easier to run and explain. The separate
`steps/` folder shows the same logic step by step for learning.

## Common Questions and Answers for app.py

**Q1. What is the role of app.py?**

A. It is the main Streamlit application file. It connects UI, PDF processing,
retrieval, prompts, and Groq answer generation.

**Q2. Why did we create separate step files if app.py already has everything?**

A. The step files are for explanation and learning. `app.py` is the final merged
working version.

**Q3. Where is the Groq API key used?**

A. The app reads `GROQ_API_KEY` from Streamlit secrets or environment variables
and passes it to ChatGroq.

**Q4. What are the modes in app.py?**

A. Medical Encyclopedia, Lab Report, and General AI.

**Q5. What is Medical Encyclopedia mode?**

A. It is used for medical reference PDFs. The app answers based on the selected
medical book or document.

**Q6. What is Lab Report mode?**

A. It is used for patient lab reports. The app explains report values, high/low
flags, and doctor follow-up points.

**Q7. What is General AI mode?**

A. It sends the question directly to Groq without PDF retrieval.

**Q8. Does app.py use a database?**

A. It does not use a traditional database. It creates an in-memory TF-IDF search
index from the uploaded PDFs.

**Q9. What happens when the app restarts?**

A. The in-memory index is cleared and must be rebuilt by selecting or uploading
PDFs again.

**Q10. What is the biggest limitation?**

A. Scanned PDFs are not handled well because OCR is not added. Also, TF-IDF is
keyword-based, not semantic like embeddings.

---

# Final Project Explanation in Short

This project is a Streamlit-based medical RAG chatbot. The user uploads a PDF,
the app extracts text, splits it into chunks, creates a TF-IDF search index,
retrieves relevant chunks for the user's question, and sends those chunks to a
Groq Llama model to generate an answer. It supports both medical encyclopedia
questions and patient lab report explanation, while avoiding diagnosis and
prescription advice.
