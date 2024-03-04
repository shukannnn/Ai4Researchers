## Ai4Researchers

This presentation introduces an AI chatbot designed to assist researchers in answering research paper-related queries. The chatbot is capable of extracting text from various document formats, including PDF, DOCX, PPTX, and LaTeX, and provides intelligent responses based on the content.

### Tech Stack

- *Python*: Core programming language for development.
- *Streamlit*: UI development tool for creating interactive web applications.
- *PyPDF2, python-docx, pptx*: Libraries for parsing and extracting text from document formats.
- *pylatexenc*: Library for converting LaTeX equations into human-readable text.
- *PyTesseract*: Optical character recognition (OCR) tool for extracting text from images.
- *Transformers (Hugging Face)*: Library for natural language processing (NLP) tasks, including question-answering and image captioning.
- *PyTorch*: Deep learning framework for tasks such as image captioning using pre-trained models.
- *NLTK*: Toolkit for natural language processing tasks like tokenization and part-of-speech tagging.
- *Aspose.Slides*: Library for extracting images from PPTX files.
- *OpenAI GPT-3*: Language model for generating responses to user queries and conducting conversations.
- *Langchain*: Library for advanced text processing tasks such as text splitting, embeddings, and conversational chains.

### External Dependencies

Ensure the following dependencies are installed externally:

- *Tesseract OCR*: Install Tesseract OCR for PyTesseract to work properly.
    - Install via: sudo apt install tesseract-ocr
- *TeX/LaTeX Distribution*: Install a TeX/LaTeX distribution for pylatexenc.
    - Example: TeX Live - [Download Link](https://www.tug.org/texlive/)

### Installation Instructions

1. Create a virtual environment with Python 3.9:
    
    conda create -p venv python==3.9
    

2. Activate the virtual environment:
    
    conda activate venv
    

3. Install the required Python packages using pip:
    
    pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai huggingface_hub InstructorEmbedding sentence_transformers==2.2.2 tiktoken aspose.slides spire.pdf python-pptx python-docx
    

4. Install external dependencies:
    - Install Tesseract OCR :
        
        sudo apt install tesseract-ocr
        ``

### Usage

1. Run the Streamlit app:
    
    streamlit run chatbot.py

2. Upload research papers in PDF, DOCX, PPTX, or LaTeX formats.
3. Click on the "Process" button to analyze the uploaded research papers.
4. Ask questions related to the content of the research papers in the text input box.
5. The chatbot will analyze the documents and provide answers based on the content.
6. Download the question-answer history in the form of a .txt file by clicking the download button.

### Note

- Ensure that the uploaded documents contain relevant research papers with readable text. The effectiveness of the chatbot depends on the quality and relevance of the content provided.
- For better performance, it is recommended to provide clear and concise questions related to the content of the research papers.

### Conclusion

This AI chatbot provides researchers with a convenient tool for accessing information within research papers and obtaining quick answers to their queries. With its ability to handle various document formats and employ advanced NLP techniques, it serves as a valuable assistant in the research process.
