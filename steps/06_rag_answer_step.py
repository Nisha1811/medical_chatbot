"""
Step 6: RAG Answer Generation

Purpose:
Send the user's question plus retrieved PDF context to the LLM.

Technology used:
- LangChain `PromptTemplate`
- Groq `ChatGroq`

Why:
The LLM should answer from the PDF context, not from random memory. This is the
"generation" part of Retrieval-Augmented Generation.
"""

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


RAG_PROMPT = PromptTemplate(
    template="""You are a careful medical assistant AI.

Use only the provided context to answer the question.
If the answer is not present in the context, say:
"I couldn't find this in the documents."

Give a useful, detailed answer. Include definitions, causes, symptoms,
treatment, prevention, and cautions when the context supports it.

Do not diagnose or prescribe medicine.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


def format_context(docs, max_chars: int = 14000) -> str:
    context_parts = []
    used_chars = 0

    for doc in docs:
        source = doc.metadata.get("source", "Uploaded PDF")
        page = doc.metadata.get("page")
        page_label = f"page {page + 1}" if isinstance(page, int) else "page unknown"
        text = doc.page_content.strip()
        if not text:
            continue

        part = f"Source: {source}, {page_label}\n{text}"
        if used_chars + len(part) > max_chars:
            remaining = max_chars - used_chars
            if remaining > 500:
                context_parts.append(part[:remaining])
            break

        context_parts.append(part)
        used_chars += len(part)

    return "\n\n---\n\n".join(context_parts)


def generate_rag_answer(api_key: str, docs, question: str) -> str:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1200,
        groq_api_key=api_key,
    )

    context = format_context(docs)
    prompt = RAG_PROMPT.format(context=context, question=question)
    return llm.invoke(prompt).content
