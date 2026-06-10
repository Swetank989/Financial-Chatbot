Project:
AI-Powered Financial Document Analysis System
Overview:
The system extracts information from uploaded PDF documents, retrieves relevant content using semantic search, and generates intelligent responses using the Llama 3 Large Language Model running locally through Ollama.
Features:
Document Upload:
    Upload financial reports in PDF format.
    Automatic text extraction from uploaded documents.

Question Answering
    Ask natural language questions about the document.
    Retrieves relevant information using semantic search.
    Generates context-aware answers using Llama 3.

Document Summarization
    Generates concise summaries of uploaded financial documents.
    Highlights important business and financial information.

Financial Analysis
    Performs high-level financial analysis.
    Identifies:Revenue trends, Profitability insights, Business risks, Growth opportunities
  
Frontend:
    Streamlit

Backend:
    Python

AI / Machine Learning:
    Sentence Transformers,
     all-MiniLM-L6-v2 Embedding Model,
     Ollama,
     Llama 3

Vector Database:
    FAISS

PDF Processing:
    pdfplumber

Installation:
    Clone Repository and Install Dependencies

Install Ollama:
Download and install Ollama:

    https://ollama.com
    Pull Llama 3 model:
        bash
        ollama pull llama3

Running the Application:

    bash
    streamlit run app.py


Built By:
Swetank Singh

B.Tech Engineering Physics
Indian Institute of Technology Roorkee
