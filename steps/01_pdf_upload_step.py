"""
Step 1: PDF Upload

Purpose:
The user gives the app a PDF file. This can be a medical encyclopedia PDF or a
patient lab report PDF.

Technology used:
- Streamlit `st.file_uploader`

Why:
Streamlit gives a simple browser-based upload box. The uploaded file can be read
as bytes and passed to the next step.
"""

import streamlit as st


uploaded_files = st.file_uploader(
    "Upload medical PDFs or lab reports",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    file_payloads = tuple((file.name, file.getvalue()) for file in uploaded_files)
    st.write(f"{len(file_payloads)} PDF file(s) selected.")
else:
    file_payloads = ()
    st.info("Upload a PDF to continue.")
