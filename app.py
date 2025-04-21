import streamlit as st
from anonymize import anonymize_pdf
from summarize import DataPrivacyProcessor, PrivacyRetriever, PrivacyRAG
import os
import logging
from pdf2image import convert_from_path
import tempfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app
st.set_page_config(page_title="Clause AI", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["PDF Anonymization", "Privacy Q&A"])

# Initialize session state for Q&A history
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Main title
st.title("Clause AI")

if page == "PDF Anonymization":
    st.header("PDF Anonymization", help="Upload a PDF to anonymize sensitive information like names and addresses.")
    st.info("Upload a PDF file to anonymize sensitive information. The output will be a text file with anonymized content.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Supported file size: up to 20MB. Only PDF format is accepted.",
        accept_multiple_files=False
    )

    # Anonymization strategy
    strategy = st.selectbox(
        "Choose anonymization strategy",
        ["redact", "mask", "pseudonymize"],
        help="Redact: Replace with [REDACTED]; Mask: Replace with X's; Pseudonymize: Replace with random identifiers."
    )

    if uploaded_file:
        # Check file size (20MB limit)
        file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        if file_size_mb > 20:
            st.error("File size exceeds 20MB limit. Please upload a smaller file.")
        else:
            # Display PDF preview
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(uploaded_file.getbuffer())
                temp_pdf_path = temp_pdf.name
                try:
                    images = convert_from_path(temp_pdf_path, first_page=1, last_page=1, poppler_path=r'E:\Release-24.08.0-0\poppler-24.08.0\Library\bin')
                    st.image(images[0], caption="PDF Preview (First Page)", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate PDF preview: {e}. Ensure Poppler is installed and added to PATH.")

            if st.button("Anonymize", help="Click to start the anonymization process."):
                output_path = "anonymized_output.txt"
                progress_bar = st.progress(0)
                try:
                    progress_bar.progress(50)
                    anonymize_pdf(temp_pdf_path, output_path, strategy=strategy)
                    with open(output_path, "r", encoding="utf-8") as f:
                        anonymized_text = f.read()
                    progress_bar.progress(100)
                    st.success("Anonymization completed!")
                    st.text_area("Anonymized Text", anonymized_text, height=300)
                    st.download_button(
                        "Download Anonymized Text",
                        anonymized_text,
                        file_name="anonymized_output.txt",
                        mime="text/plain",
                        help="Download the anonymized text as a .txt file."
                    )
                except Exception as e:
                    st.error(f"Error during anonymization: {e}")
                finally:
                    progress_bar.empty()
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)
                    if os.path.exists(output_path):
                        os.remove(output_path)

else:
    st.header("Privacy Q&A", help="Ask questions about data privacy laws and regulations.")
    st.info("Enter a question about data privacy (e.g., GDPR, DPDPA 2023) to get a concise, expert response.")

    # Initialize RAG
    processor = DataPrivacyProcessor()
    persist_directory = "F:/College/SEM 12/Capstone Project/Clause AI/chroma"
    os.makedirs(persist_directory, exist_ok=True)
    retriever = PrivacyRetriever(chunks=[], embedder=processor.embedder, persist_directory=persist_directory)
    rag = PrivacyRAG(retriever)

    # Display ChromaDB status
    chunk_count = retriever.collection.count()
    st.write(f"ChromaDB contains {chunk_count} chunks")
    if chunk_count == 0:
        st.error("ChromaDB is empty. Please run preprocess.py with relevant PDFs.")

    # Debug button to view sample chunks
    if st.button("View Sample Chunks"):
        if chunk_count > 0:
            sample = retriever.collection.peek(5)
            for i, meta in enumerate(sample["metadatas"]):
                st.write(f"**Chunk {i+1}**: {meta['text'][:200]}...")
        else:
            st.warning("No chunks available to display.")

    # Query input
    query = st.text_input(
        "Enter your question",
        placeholder="e.g., What are the obligations of Data Fiduciaries under DPDPA 2023?",
        help="Ask specific questions about data privacy laws or guidelines."
    )

    if st.button("Get Answer", help="Click to generate a response based on privacy laws."):
        if not query:
            st.error("Please enter a question.")
        else:
            try:
                with st.spinner("Generating response..."):
                    # Retrieve chunks for debugging
                    retrieved_chunks = retriever.retrieve(query, top_k=5)
                    response = rag.generate_response(query)
                st.write("**Answer:**")
                st.write(response)
                # Display retrieved chunks for debugging
                if retrieved_chunks:
                    with st.expander("Retrieved Chunks"):
                        for i, (chunk, score) in enumerate(retrieved_chunks):
                            st.write(f"**Chunk {i+1} (Score: {score:.3f})**: {chunk[:500]}...")
                else:
                    st.warning("No relevant chunks retrieved. Try rephrasing the query, lowering the similarity threshold, or checking PDF text extraction.")
                # Store in history
                st.session_state.qa_history.append({"query": query, "response": response})
            except Exception as e:
                st.error(f"Error generating response: {e}. Check logs for details or verify Groq API key.")

# Cleanup temporary files
if os.path.exists(tempfile.gettempdir()):
    shutil.rmtree(tempfile.gettempdir(), ignore_errors=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Tavish Shetty, Suryadev Sudheer and Rishikesh Vadodaria | Powered by Streamlit and Hugging Face")