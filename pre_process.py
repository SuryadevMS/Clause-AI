from summarize import DataPrivacyProcessor, PrivacyRetriever
import logging
import nltk
nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

processor = DataPrivacyProcessor()
pdf_paths = [
    "./data-privacy-pdf/RBI-Guidelines.pdf",
    "./data-privacy-pdf/2024-0118-Policy-SEBI_Circular_on_Cybersecurity_and_Cyber_Resilience_Framework_(CSCRF)_for_SEBI_Regulated.pdf",
    "./data-privacy-pdf/Aadhaar_Act_2016_as_amended.pdf",
    "./data-privacy-pdf/DPDPA - 2023.pdf",
    "./data-privacy-pdf/it_act_2000_updated.pdf",
    "./data-privacy-pdf/Personal Data Protection Bill, 2019.pdf",
    "./data-privacy-pdf/rti-act.pdf"
]
chunks = processor.extract_text_from_pdfs(pdf_paths)
logger.info(f"Extracted {len(chunks)} chunks from PDFs")
if chunks:
    logger.info(f"Sample chunk: {chunks[0][:200]}...")
retriever = PrivacyRetriever(chunks=chunks, embedder=processor.embedder, persist_directory="F:/College/SEM 12/Capstone Project/Clause AI/chroma")
print(f"Stored {retriever.collection.count()} chunks in ChromaDB")