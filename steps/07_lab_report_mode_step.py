"""
Step 7: Lab Report Mode

Purpose:
Explain uploaded patient reports such as CBC, blood sugar, HbA1c, lipid profile,
LFT, KFT, or thyroid reports.

Technology used:
- Same PDF extraction, chunking, and retrieval pipeline
- A different prompt for patient-friendly lab report explanation

Why:
A lab report is different from a medical encyclopedia. The app should focus on
test values, units, reference ranges, high/low flags, and doctor follow-up
questions. It should not diagnose the patient.
"""

from langchain.prompts import PromptTemplate


LAB_REPORT_PROMPT = PromptTemplate(
    template="""You are a careful medical report explainer for patients.

Use only the uploaded lab report context to answer. Do not invent values.
If a value, unit, or reference range is not present, say that it is not shown in
the report.

When answering:
- Start with a short plain-language summary.
- List important values found in the report.
- If reference ranges or high/low flags are shown, explain which values appear
  high, low, or within range.
- Explain what those tests generally relate to.
- Mention what the patient should discuss with a doctor.

Do not diagnose, prescribe medicine, or make treatment decisions.

Lab report context:
{context}

Patient question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


def build_lab_question(user_query: str) -> str:
    clean_query = user_query.strip()

    if not clean_query:
        return (
            "Explain this lab report in simple language. Highlight important "
            "values, abnormal high or low results if reference ranges are shown, "
            "and what the patient should discuss with a doctor."
        )

    return clean_query
