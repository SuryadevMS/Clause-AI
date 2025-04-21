import numpy as np
import chromadb
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, BartForConditionalGeneration
from typing import List, Tuple
import logging
import nltk
import requests
import os
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Download required NLTK data
nltk.download('wordnet')
nltk.download('punkt_tab')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name, device='cpu')
        # Ensure model is fully loaded
        self.embedder.eval()
        logger.info(f"Loaded SentenceTransformer model {model_name} on CPU")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.embedder.encode(input, show_progress_bar=False)
        return embeddings.tolist()

class DataPrivacyProcessor:
    def __init__(self):
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        
    def extract_text_from_pdfs(self, pdf_paths: List[str]) -> List[str]:
        all_chunks = []
        for pdf_path in pdf_paths:
            try:
                full_text = ""
                # Try pdfplumber first
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text and len(text.strip()) > 10:
                            full_text += text + "\n"
                
                # If little or no text extracted, use OCR
                if len(full_text.strip()) < 100:
                    logger.warning(f"pdfplumber extracted insufficient text from {pdf_path}. Falling back to OCR.")
                    images = convert_from_path(pdf_path)  # Convert PDF to images
                    full_text = ""
                    for image in images:
                        text = pytesseract.image_to_string(image, lang='eng')
                        if text and len(text.strip()) > 10:
                            full_text += text + "\n"
                
                # Split into sentences and create chunks
                sentences = sent_tokenize(full_text)
                current_chunk = ""
                current_length = 0
                max_chunk_length = 500  # Max characters per chunk
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence or len(sentence) < 10:
                        continue
                    sentence_length = len(sentence)
                    
                    if current_length + sentence_length <= max_chunk_length:
                        current_chunk += sentence + " "
                        current_length += sentence_length
                    else:
                        if current_chunk:
                            all_chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                        current_length = sentence_length
                
                if current_chunk:
                    all_chunks.append(current_chunk.strip())
                
                logger.info(f"Extracted {len(all_chunks)} chunks from {pdf_path}")
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
        return all_chunks

class PrivacyRetriever:
    def __init__(self, chunks: List[str], embedder: SentenceTransformer, persist_directory: str = "F:/College/SEM 12/Capstone Project/Clause AI/chroma"):
        self.chunks = chunks
        self.embedder = embedder
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        except Exception as e:
            logger.error(f"Failed to initialize CrossEncoder: {e}")
            raise
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.embedding_function = SentenceTransformerEmbeddingFunction()
            self.collection = self.client.get_or_create_collection(
                name="privacy_chunks",
                embedding_function=self.embedding_function
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client or collection: {e}")
            raise
        
        try:
            import numpy as np
            logger.info(f"NumPy version: {np.__version__}")
        except ImportError:
            logger.error("NumPy is not installed. Please install it with 'pip install numpy'")
            raise ImportError("NumPy is required for ChromaDB and sentence-transformers")
        
        try:
            chunk_count = self.collection.count()
            logger.info(f"ChromaDB collection contains {chunk_count} chunks")
            if chunk_count == 0 and chunks:
                logger.info(f"Encoding {len(chunks)} chunks with SentenceTransformer")
                embeddings = self.embedder.encode(chunks, show_progress_bar=True)
                if not isinstance(embeddings, np.ndarray):
                    logger.error("Embeddings are not a NumPy array. Check sentence-transformers compatibility.")
                    raise ValueError("Invalid embeddings format")
                logger.info(f"Generated embeddings shape: {embeddings.shape}")
                self.collection.add(
                    ids=[f"chunk_{i}" for i in range(len(chunks))],
                    embeddings=embeddings.tolist(),
                    metadatas=[{"text": chunk} for chunk in chunks]
                )
                logger.info(f"Stored {len(chunks)} chunks in ChromaDB")
        except chromadb.errors.InternalError as ce:
            logger.error(f"ChromaDB internal error, possible database corruption: {ce}")
            raise
        except Exception as e:
            logger.error(f"Failed to access or populate ChromaDB collection: {e}")
            raise
        
    def retrieve(self, query: str, top_k: int = 10, min_similarity: float = 0.3):
        try:
            query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2
            )
            
            candidates = []
            for text, distance in zip(results["metadatas"][0], results["distances"][0]):
                similarity = 1 - distance
                if similarity >= min_similarity:
                    candidates.append((text["text"], similarity))
            
            if not candidates:
                logger.warning(f"No chunks above min_similarity {min_similarity} for query: {query}")
                return []
            
            chunk_texts = [chunk for chunk, _ in candidates]
            rerank_scores = self.reranker.predict([(query, chunk) for chunk in chunk_texts])
            reranked = sorted(zip(chunk_texts, rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]
            
            retrieved = [(chunk, float(score)) for chunk, score in reranked]
            logger.info(f"Retrieved {len(retrieved)} chunks for query '{query}': {[f'{chunk[:50]}... ({score:.3f})' for chunk, score in retrieved]}")
            return retrieved
            
        except ValueError as ve:
            logger.error(f"Query encoding failed: {ve}")
            return []
        except Exception as e:
            logger.error(f"Retrieval failed unexpectedly: {e}")
            return []

class PrivacyRAG:
    def __init__(self, retriever, max_tokens=2048):
        self.retriever = retriever
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")  # Still used for truncation
        self.system_prompt = """You are a data privacy expert. Using only the provided context from data privacy laws and guidelines, answer the query accurately, concisely, and directly in formal legal language. Do not speculate or include information beyond the context or question."""

    def _truncate_context(self, context: str, query: str) -> str:
        try:
            prompt_tokens = len(self.tokenizer.encode(self.system_prompt))
            query_tokens = len(self.tokenizer.encode(query))
            available_tokens = self.max_tokens - prompt_tokens - query_tokens - 100
            context_tokens = self.tokenizer.encode(context)
            if len(context_tokens) > available_tokens:
                context_tokens = context_tokens[:available_tokens]
                return self.tokenizer.decode(context_tokens, skip_special_tokens=True)
            return context
        except Exception as e:
            logger.error(f"Context truncation failed: {e}")
            return context

    def generate_response(self, query: str) -> str:
        try:
            context_chunks_with_scores = self.retriever.retrieve(query)
            if not context_chunks_with_scores:
                logger.info(f"No relevant chunks found for query: {query}")
                return "No relevant information found in the provided context."
            context_chunks = [chunk for chunk, _ in context_chunks_with_scores]
            context = "\n\n".join(context_chunks)
            context = self._truncate_context(context, query)
            full_prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuery:\n{query}"

            # Call Groq API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            }
            payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9
            }
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"].strip()
            logger.info(f"Generated response for '{query}': {answer}")
            return answer
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"