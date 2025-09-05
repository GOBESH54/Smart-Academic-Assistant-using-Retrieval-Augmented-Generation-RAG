RAG System with Gemini API
A document-based Retrieval-Augmented Generation (RAG) system built using Streamlit, supporting local model downloads, FAISS-powered vector search, and answering user questions over uploaded documents using the Gemini API.

Features
Gemini API Integration: Supports Gemini 1.5 Flash, Pro, 2.0 Flash, and latest variants for generative Q&A over documents.

Local Sentence Transformer Embeddings: Efficiently downloads and manages sentence transformer models for in-environment operation.

Multi-format Document Support: Extracts text from PDF, DOCX, and TXT files.

Smart Text Chunking: Splits documents into overlapping text chunks for accurate retrieval.

FAISS Vector Database: Stores chunk embeddings and enables fast similarity search.

Streamlit UI: Allows document upload, model selection, configuration of chunk size/overlap/results, and Q&A chat interface.

Persistence: Embedding index is saved/reloaded automatically for session continuity.

Robust Error Handling: Feedback for invalid API keys, document extraction failures, and model init errors.

Quickstart
1. Clone the Repository
bash
git clone https://github.com/your-username/rag-gemini-streamlit.git
cd rag-gemini-streamlit
2. Install Dependencies
Ensure Python 3.9+ is installed. Then, install the required libraries:

bash
pip install -r requirements.txt
Main dependencies include:

streamlit

google-generativeai

sentence-transformers

faiss-cpu

PyPDF2

python-docx

numpy

torch

3. Run the Application
bash
streamlit run main.py
4. Obtain Gemini API Key
Go to Google AI Studio

Sign in with your Google account

Click "Create API Key" and copy-paste it into the app sidebar ("Enter Gemini API Key")

Usage Guide
Configure Settings:

Enter your Gemini API key.

Choose a Gemini model.

Adjust chunk size, overlap, and the number of contexts to retrieve.

Upload a Document:

Supported formats: PDF, DOCX, TXT.

Click "Process Document" after upload.

Ask Questions:

Type a question in the chat box.

Receive answers generated from the retrieved document context.

Review Chat History:

Chat interface maintains message history for reference.

Clear History:

Use the "Clear Chat History" button to restart your Q&A session.

Advanced Settings
Chunk Size & Overlap: Control granularity and redundancy in document chunking.

Model Choice: Select Gemini version per performance and cost tradeoff.

Results Returned: Adjust number of retrieved chunks per query for context-rich responses.

Persistent Index
Document embeddings and FAISS index are saved as current_document_index.faiss and metadata for reuse and session restoration.

Troubleshooting
API Key Validation: Key must start with AIzaSy and be 39 characters.

Network Issues: Ensure reliable internet during model and API operations.

Document Support: If extraction fails, verify the file format and encoding.

License
MIT License.

Support
Open an issue or pull request for bug reports, improvements, or questions.
