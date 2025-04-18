# Clause AI

Clause AI is a privacy-focused legal document processing tool that offers two key features:
1. **PDF Anonymization**: Automatically detect and anonymize sensitive information in legal documents
2. **Privacy Q&A**: Ask questions about data privacy laws and regulations with AI-powered answers

## Overview

Clause AI helps legal professionals, compliance officers, and privacy specialists handle sensitive legal documents by:
- Anonymizing personal and sensitive information in PDFs
- Providing accurate answers to complex questions about privacy regulations like GDPR and DPDPA 2023
- Using advanced NLP techniques to extract, process, and understand legal text

## Features

### PDF Anonymization
- Upload PDFs up to 20MB in size
- Choose from multiple anonymization strategies:
  - **Redact**: Replace sensitive information with "[REDACTED]"
  - **Mask**: Replace with "X" characters
  - **Pseudonymize**: Replace with random identifiers
- Preview PDFs before processing
- Download anonymized text files

### Privacy Q&A
- Ask questions about data privacy laws and regulations
- Get concise, expert responses based on actual legal texts
- View relevant source chunks for transparency
- Built on a RAG (Retrieval-Augmented Generation) architecture

## Technical Architecture

### How It Works

#### PDF Anonymization Pipeline
1. **Text Extraction**: Uses pdfplumber for native PDF text and falls back to OCR (pytesseract) when needed
2. **Named Entity Recognition**: Applies SpaCy's transformer-based NER model to detect sensitive entities
3. **Context-Aware Filtering**: Preserves critical legal entities while targeting personal information
4. **Anonymization**: Applies the selected strategy (redact/mask/pseudonymize) to identified entities

#### Privacy Q&A System
1. **Document Processing**: Chunks documents into semantically meaningful segments
2. **Embedding**: Converts text chunks into vector embeddings using Sentence Transformers
3. **Retrieval**: Uses ChromaDB for efficient vector storage and similarity search
4. **Reranking**: Applies a cross-encoder to improve retrieval precision
5. **Response Generation**: Leverages Groq's LLM API with retrieved context to generate accurate answers

### Technical Components
- **Streamlit**: For the web interface
- **SpaCy**: For advanced NER (Named Entity Recognition) with `en_core_web_trf` model
- **Sentence Transformers**: Using `all-mpnet-base-v2` for semantic text embedding
- **ChromaDB**: For vector storage and retrieval
- **Groq LLM API**: Using `meta-llama/llama-4-scout-17b-16e-instruct` for generating high-quality responses
- **PDF Processing**: Using pdfplumber and pytesseract for text extraction
- **Cross-Encoder**: Using `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking

## Installation

### Prerequisites
- Python 3.8+
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) for PDF processing (Windows users)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for OCR capabilities
- At least 8GB RAM for optimal performance
- 2GB free disk space for vector database and models

### Detailed Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/clause-ai.git
cd clause-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download SpaCy models:
```bash
python -m spacy download en_core_web_trf
```

5. Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your-groq-api-key
```

6. Set up Tesseract OCR path (Windows only):
   - After installing Tesseract, add it to your PATH or specify the path in `anonymize.py`
   - Modify the path in `app.py` if needed: `poppler_path=r'path\to\poppler\bin'`

7. Create necessary directories:
```bash
mkdir -p data-privacy-pdf
mkdir -p project_root/chroma_db
```

8. Place your legal PDF documents in the `data-privacy-pdf` directory:
   - Make sure PDFs are properly formatted
   - Ensure file permissions allow read access

## Usage

### Preprocessing Documents

Before using the Privacy Q&A feature, you'll need to preprocess your legal documents:

```bash
python pre_process.py
```

This will:
- Extract text from PDFs in the `data-privacy-pdf` directory
- Process and chunk the text
- Store embeddings in ChromaDB

### Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

### Example Q&A Queries

Here are some example questions you can ask the Privacy Q&A system:

1. **Basic Definitions**
   - "What is the definition of 'Data Principal' under DPDPA 2023?"
   - "How does the DPDPA define 'personal data'?"

2. **Compliance Requirements**
   - "What are the key obligations of Data Fiduciaries under DPDPA 2023?"
   - "What consent requirements apply to processing sensitive personal data?"

3. **Rights and Protections**
   - "What rights do data principals have under DPDPA?"
   - "How can users request access to their personal data?"

4. **Specific Regulations**
   - "What security measures are required for financial institutions under RBI guidelines?"
   - "What are the cybersecurity requirements under SEBI regulations?"

5. **Comparative Analysis**
   - "How do data localization requirements differ between DPDPA and earlier drafts?"
   - "What are the key differences between DPDPA 2023 and the Personal Data Protection Bill 2019?"

## Troubleshooting

### Common Issues

#### PDF Text Extraction Fails
- **Symptom**: No text is extracted from PDFs or OCR produces gibberish
- **Solutions**:
  - Ensure Tesseract OCR is properly installed and accessible
  - Check PDF isn't locked/encrypted
  - Try converting scanned PDFs to a higher resolution
  - Update the Tesseract path in the code if necessary

#### ChromaDB Collection Errors
- **Symptom**: "No collection found" or empty results in Q&A
- **Solutions**:
  - Run `pre_process.py` to initialize the database
  - Check the path in `persist_directory` is correct and accessible
  - Ensure you have adequate disk space for the vector database

#### Out of Memory Errors
- **Symptom**: Application crashes when processing large documents
- **Solutions**:
  - Split large PDFs into smaller files
  - Reduce `max_chunk_length` in `DataPrivacyProcessor` class
  - Increase system swap space

#### Groq API Connection Issues
- **Symptom**: "Error generating response" in Privacy Q&A
- **Solutions**:
  - Verify your API key is valid and properly set in the `.env` file
  - Check internet connectivity
  - Ensure you haven't exceeded API rate limits

#### SpaCy Model Loading Errors
- **Symptom**: "Failed to load en_core_web_trf" error
- **Solutions**:
  - Reinstall the model: `python -m spacy download en_core_web_trf`
  - Check system has adequate RAM (model requires at least 4GB)

## Customizing for Different Privacy Laws

Clause AI can be extended to support additional privacy laws and regulations:

### Adding New Documents

1. Place new PDF documents in the `data-privacy-pdf` directory
2. Edit `pre_process.py` to include the new file paths:
```python
pdf_paths = [
    # Existing paths
    "./data-privacy-pdf/RBI-Guidelines.pdf",
    # Add your new document
    "./data-privacy-pdf/your-new-document.pdf",
]
```
3. Run the preprocessing script:
```bash
python pre_process.py
```

### Customizing Entity Recognition

To recognize jurisdiction-specific entities:

1. Open `anonymize.py` and modify the `preserve_entities` set:
```python
preserve_entities = {"India", "Reserve Bank of India", "Aadhaar", "Your New Entity"}
```

2. To adjust which entity types are anonymized, modify the filtering logic:
```python
if ent.label_ not in ["PERSON", "GPE", "ORG", "DATE", "YOUR_NEW_TYPE"]:
    continue
```

### Tuning the Q&A System

For better results with different legal systems:

1. Adjust the system prompt in `summarize.py` to reflect the legal context:
```python
self.system_prompt = """You are a data privacy expert specializing in [YOUR JURISDICTION] law...
```

2. Fine-tune retrieval parameters in `retrieve` method:
```python
def retrieve(self, query: str, top_k: int = 15, min_similarity: float = 0.3):
    # Adjust parameters based on your corpus
```

## Project Structure

- `app.py`: Main Streamlit application
- `anonymize.py`: PDF anonymization functionality
- `summarize.py`: Text processing, embedding, and RAG functionality
- `pre_process.py`: Document preprocessing script
- `requirements.txt`: Project dependencies
- `data-privacy-pdf/`: Directory containing privacy laws and regulations PDFs

## Supported Documents

The system comes pre-configured to work with:
- DPDPA 2023
- RBI Guidelines
- SEBI Circular on Cybersecurity
- Aadhaar Act 2016
- IT Act 2000
- Personal Data Protection Bill 2019
- RTI Act

## Limitations

- PDF anonymization works best with text-based PDFs
- Large PDFs may take longer to process
- The Q&A system is limited to the documents it has been trained on
- Responses are only as accurate as the underlying document corpus

## Contributors

Developed by:
- Tavish Shetty
- Suryadev Sudheer
- Rishikesh Vadodaria

## License

MIT
