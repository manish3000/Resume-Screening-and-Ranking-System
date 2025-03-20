import streamlit as st
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords (only required once)
nltk.download("stopwords")

# Load English stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Removes punctuation and stopwords from text."""
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.lower().split()  # Convert to lowercase
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(filtered_words)

def extract_text_from_pdf(pdf_file):
    """Extracts and cleans text from a PDF file."""
    text = ""
    pdf_bytes = pdf_file.getvalue()  # Get the binary content

    # Open the PDF using PyMuPDF's memory buffer mode
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")  

    for page in doc:
        text += page.get_text()
    
    return clean_text(text)  # Clean extracted text

def calculate_cosine_similarity(text1, text2):
    """Computes the cosine similarity between two text documents."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(vectors[0], vectors[1])
    
    return similarity_score[0][0]  # Extract the similarity value

def main():
    """Streamlit App"""
    st.title("Resume Ranking Based on Job Description")

    uploaded_files = st.file_uploader("Upload one or more Resume PDFs", type="pdf", accept_multiple_files=True)
    job_description = st.text_area("Enter Job Description:")

    if uploaded_files and job_description:
        cleaned_job_desc = clean_text(job_description)
        results = []

        for uploaded_file in uploaded_files:
            pdf_text = extract_text_from_pdf(uploaded_file)
            similarity_score = calculate_cosine_similarity(pdf_text, cleaned_job_desc)
            
            # Store results as (score, file name)
            results.append((similarity_score, uploaded_file.name))

        # Sort by highest similarity score
        results.sort(reverse=True, key=lambda x: x[0])

        # Display results
        st.subheader("Ranked Resumes (Higher Score First)")
        for rank, (score, file_name) in enumerate(results, start=1):
            st.write(f"**{rank}. {file_name}** - Similarity Score: **{score:.4f}**")

if __name__ == "__main__":
    main()
