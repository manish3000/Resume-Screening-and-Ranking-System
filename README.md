# Resume-Screening-and-Ranking-System
A Streamlit web application that automates the process of comparing resumes against a given job description.

# AI-powered Resume Screening and Ranking System



A **Streamlit** web application that automates the process of comparing resumes against a given job description. This project leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques (such as **TF-IDF** and **cosine similarity**) to rank resumes in order of relevance. By reducing manual screening efforts, it helps recruiters identify the most suitable candidates quickly and efficiently.

---

## Table of Contents
1. [Features](#features)  
2. [Project Structure](#project-structure)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [How It Works](#how-it-works)  
6. [Future Enhancements](#future-enhancements)  
7. [Contributing](#contributing)  
8. [Contact](#contact)  

---

## Features
- **Multiple Resume Uploads:** Upload one or more PDF resumes simultaneously.
- **Job Description Input:** A dedicated text area for entering job descriptions.
- **Automated Text Extraction:** Utilizes **PyMuPDF (Fitz)** to extract text from PDF files.
- **Text Preprocessing:** Cleans the extracted text by removing punctuation and stopwords using **NLTK**.
- **Similarity Calculation:** Employs **TF-IDF vectorization** and **cosine similarity** to determine how closely resumes match the job description.
- **Ranked Output:** Displays resumes sorted by similarity scores, with the highest matches at the top.
- **User-Friendly Interface:** Built with **Streamlit**, providing an interactive and intuitive experience.

---

## Project Structure
```plaintext
.
├── app.py                # Main Streamlit application
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
└── ...
```
> **Note:** The file names and structure might vary depending on your setup.

---

## Installation

### Prerequisites
- **Python 3.8+**  
- **pip** (Python package manager)

### Steps
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/manish3000/Resume-Screening-and-Ranking-System
   cd resume-ranking-ai
   ```

2. **Create and Activate a Virtual Environment:** (recommended)
   ```bash
   # On Windows:
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Stopwords:** (Only required once)
   ```python
   import nltk
   nltk.download('stopwords')
   ```

---

## Usage

1. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

2. **Access the App:**  
   Open your web browser and navigate to the URL displayed in the terminal (usually [http://localhost:8501](http://localhost:8501)).

3. **Upload Resumes:**  
   Click on **"Browse files"** or drag and drop your PDF resumes into the designated area.

4. **Enter Job Description:**  
   Provide the job description in the text area.

5. **View Ranked Results:**  
   The app will display a ranked list of resumes based on their similarity scores, with the best-matching resumes at the top.

---

## How It Works

1. **Text Extraction:**  
   The application uses **PyMuPDF (Fitz)** to extract text from each uploaded PDF resume.

2. **Text Preprocessing:**  
   The extracted text is cleaned using **NLTK** by removing punctuation and stopwords and converting text to lowercase.

3. **Vectorization:**  
   The cleaned text is transformed into numerical vectors using **TF-IDF**, which captures the importance of each word.

4. **Cosine Similarity Calculation:**  
   The similarity between each resume vector and the job description vector is computed using **cosine similarity**. A higher similarity score indicates a closer match.

5. **Ranking:**  
   Resumes are sorted in descending order based on their similarity scores, simplifying the identification of top candidates.

---

## Future Enhancements
- **Deep Learning Integration:** Incorporate advanced models like **BERT** or **GPT** for improved contextual understanding.
- **Multilingual Support:** Extend capabilities to handle resumes and job descriptions in multiple languages.
- **File Format Compatibility:** Add support for additional file formats such as `.docx`, `.txt`, etc.
- **Advanced Customization:** Allow recruiters to assign weights to specific skills or experiences.
- **Bias Detection and Mitigation:** Develop strategies to detect and reduce potential biases in resume ranking.

---

## Contributing
Contributions are welcome! If you would like to contribute to this project, follow these steps:
1. **Fork** the repository.
2. **Create** a new feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Commit** your changes:
   ```bash
   git commit -m 'Add new feature'
   ```
4. **Push** to your branch:
   ```bash
   git push origin feature/my-feature
   ```
5. **Open** a pull request on GitHub.

---

## Contact
For questions, feedback, or collaboration inquiries, please contact:
- **Name:** Manish Kujur
- **Email:** [manishkujur05@gmail.com](mailto:manishkujur05@gmail.com)
- **LinkedIn:** [Manish Kujur](https://www.linkedin.com/in/manish-kujur-bb7615239/)

Feel free to open an issue on GitHub if you encounter any bugs or have feature requests. Your feedback and support are greatly appreciated!

---
