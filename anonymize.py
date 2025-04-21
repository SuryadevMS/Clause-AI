import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import spacy
import random
import os
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load SpaCy transformer model
try:
    nlp = spacy.load("en_core_web_trf")
except Exception as e:
    logger.error(f"Failed to load en_core_web_trf. Ensure it is installed: {e}")
    raise


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using pdfplumber, falling back to OCR if insufficient text is extracted.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text.
    """
    text = ""
    try:
        # Try pdfplumber first
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 10:
                    text += page_text + "\n"
        
        # If little or no text extracted, use OCR
        if len(text.strip()) < 100:
            logger.warning(f"pdfplumber extracted insufficient text from {pdf_path}. Falling back to OCR.")
            try:
                images = convert_from_path(pdf_path)
                for image in images:
                    ocr_text = pytesseract.image_to_string(image, lang="eng")
                    if ocr_text and len(ocr_text.strip()) > 10:
                        text += ocr_text + "\n"
                logger.info(f"OCR extracted {len(text)} characters from {pdf_path}")
            except Exception as e:
                logger.error(f"OCR failed for {pdf_path}: {e}")
                return text
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
    
    if not text.strip():
        logger.warning(f"No extractable text found in {pdf_path}.")
    else:
        logger.info(f"Extracted {len(text)} characters from {pdf_path}")
    return text

def chunk_text(text: str, max_length: int = 1000) -> List[str]:
    """
    Split text into chunks to avoid memory issues with SpaCy.
    
    Args:
        text (str): Input text.
        max_length (int): Maximum length of each chunk.
    
    Returns:
        List[str]: List of text chunks.
    """
    sentences = text.split("\n")
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_length = len(sentence)
        
        if current_length + sentence_length <= max_length:
            current_chunk += sentence + "\n"
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + "\n"
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def ai_anonymize_text(text: str, strategy: str = "redact") -> str:
    """
    Anonymize text using SpaCy's transformer-based NER model with context-aware filtering.
    
    Args:
        text (str): Input text.
        strategy (str): Anonymization strategy ("redact", "pseudonymize", "mask").
    
    Returns:
        str: Anonymized text.
    """
    # Chunk text to handle large inputs
    chunks = chunk_text(text, max_length=1000)
    anonymized_text = ""
    
    # Context-aware filtering: Preserve certain entities in legal contexts
    preserve_entities = {"India", "Reserve Bank of India", "Aadhaar"}  # Add more as needed
    
    for chunk in chunks:
        try:
            doc = nlp(chunk)
            segments = []
            last_end = 0
            
            # Identify and filter entities
            for ent in doc.ents:
                if ent.label_ not in ["PERSON", "GPE", "ORG", "DATE"]:
                    continue
                if ent.text in preserve_entities:
                    continue  # Skip entities critical to legal context
                
                start, end = ent.start_char, ent.end_char
                entity_text = ent.text
                entity_type = ent.label_
                
                if last_end < start:
                    segments.append({"text": chunk[last_end:start], "action": "keep"})
                
                segments.append({"text": entity_text, "action": strategy, "type": entity_type})
                last_end = end
            
            if last_end < len(chunk):
                segments.append({"text": chunk[last_end:], "action": "keep"})
            
            # Apply anonymization
            for segment in segments:
                text_segment = segment["text"]
                action = segment["action"]
                
                if action == "keep":
                    anonymized_text += text_segment
                elif action == "pseudonymize":
                    anonymized_text += f"{segment['type']}_{random.randint(1, 1000)}"
                elif action == "mask":
                    anonymized_text += "X" * len(text_segment)
                elif action == "redact":
                    anonymized_text += "[REDACTED]"
        
        except Exception as e:
            logger.error(f"Failed to process chunk: {chunk[:50]}...: {e}")
            anonymized_text += chunk  # Fallback to original chunk
    
    logger.info(f"Anonymized text length: {len(anonymized_text)} characters")
    return anonymized_text

def anonymize_pdf(pdf_path: str, output_path: str, strategy: str = "redact") -> None:
    """
    Process a PDF, anonymize its text, and save the result to a file.
    
    Args:
        pdf_path (str): Path to the input PDF.
        output_path (str): Path to save the anonymized text.
        strategy (str): Anonymization strategy ("redact", "pseudonymize", "mask").
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    original_text = extract_text_from_pdf(pdf_path)
    if not original_text:
        logger.error("No text could be extracted from the PDF.")
        return
    
    anonymized_text = ai_anonymize_text(original_text, strategy)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(anonymized_text)
        logger.info(f"Anonymized PDF saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save anonymized text to {output_path}: {e}")

if __name__ == "__main__":
    # Example usage
    pdf_path = "./data-privacy-pdf/DPDPA - 2023.pdf"
    output_path = "./anonymized_DPDPA_2023.txt"
    anonymize_pdf(pdf_path, output_path, strategy="redact")