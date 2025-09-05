import streamlit as st
import google.generativeai as genai
import os
import tempfile
import shutil
from pathlib import Path
import PyPDF2
import docx
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import torch
import json
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalModelManager:
    """Manages local model downloads and storage within the virtual environment"""
    
    def __init__(self, models_dir: str = "local_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def download_sentence_transformer(self, model_name: str = "all-MiniLM-L6-v2"):
        """Download sentence transformer model locally"""
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            logger.info(f"Model {model_name} already exists locally")
            return str(model_path)
        
        logger.info(f"Downloading {model_name} to local directory...")
        
        # Download to temporary location first
        temp_model = SentenceTransformer(model_name)
        
        # Save to local directory
        temp_model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        return str(model_path)
    
    def load_local_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load model from local directory"""
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            logger.info("Model not found locally, downloading...")
            self.download_sentence_transformer(model_name)
        
        return SentenceTransformer(str(model_path))

class DocumentProcessor:
    """Handles document processing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return ""
        return text
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT: {e}")
            return ""
    
    def process_document(self, file_path: str, file_type: str) -> str:
        """Process document based on file type"""
        if file_type.lower() == 'pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_type.lower() in ['docx', 'doc']:
            return self.extract_text_from_docx(file_path)
        elif file_type.lower() == 'txt':
            return self.extract_text_from_txt(file_path)
        else:
            return ""

class TextChunker:
    """Handles text chunking for better retrieval"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks

class VectorStore:
    """Manages vector storage and retrieval using FAISS"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.texts = []
        self.embeddings = None
    
    def build_index(self, texts: List[str], embeddings: np.ndarray):
        """Build FAISS index from embeddings"""
        self.texts = texts
        self.embeddings = embeddings
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Built FAISS index with {len(texts)} documents")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for most similar documents"""
        if self.index is None:
            return []
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                results.append({
                    'text': self.texts[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def save_index(self, path: str):
        """Save index and metadata to disk"""
        save_data = {
            'texts': self.texts,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None
        }
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, f"{path}_index.faiss")
        
        # Save metadata
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(save_data, f)
    
    def load_index(self, path: str):
        """Load index and metadata from disk"""
        try:
            # Load FAISS index
            if os.path.exists(f"{path}_index.faiss"):
                self.index = faiss.read_index(f"{path}_index.faiss")
            
            # Load metadata
            if os.path.exists(f"{path}_metadata.json"):
                with open(f"{path}_metadata.json", 'r') as f:
                    data = json.load(f)
                    self.texts = data['texts']
                    if data['embeddings']:
                        self.embeddings = np.array(data['embeddings'])
            
            logger.info("Successfully loaded saved index")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

class GeminiRAGSystem:
    """Complete RAG system using Gemini API"""
    
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-1.5-flash"):
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        
        # Validate API key format
        if not self.validate_api_key_format(gemini_api_key):
            raise ValueError("Invalid API key format. API key should start with 'AIzaSy' and be 39 characters long.")
        
        # Configure Gemini API
        try:
            genai.configure(api_key=gemini_api_key)
            # Use updated model name (fixed from gemini-pro)
            self.gemini_model = genai.GenerativeModel(model_name)
            
            # Test API key validity
            self.test_api_connection()
            
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini API with model {model_name}: {str(e)}")
        
        # Initialize components
        self.model_manager = LocalModelManager()
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker()
        self.vector_store = VectorStore()
        
        # Load sentence transformer locally
        self.sentence_model = None
        
        logger.info(f"RAG System initialized successfully with model: {model_name}")
    
    def validate_api_key_format(self, api_key: str) -> bool:
        """Validate API key format"""
        if not api_key:
            return False
        if not api_key.startswith('AIzaSy'):
            return False
        if len(api_key) != 39:
            return False
        return True
    
    def test_api_connection(self):
        """Test API connection with a simple request"""
        try:
            test_response = self.gemini_model.generate_content("Hello")
            logger.info(f"✅ Gemini API connection successful with model: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Gemini API connection failed: {e}")
            raise ValueError(f"API connection failed with model {self.model_name}: {str(e)}")
    
    def initialize_sentence_model(self):
        """Initialize sentence transformer model locally"""
        if self.sentence_model is None:
            logger.info("Loading sentence transformer model...")
            self.sentence_model = self.model_manager.load_local_model()
            logger.info("Sentence transformer model loaded successfully")
    
    def process_uploaded_file(self, uploaded_file) -> bool:
        """Process uploaded file and build vector store"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Extract text based on file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            text = self.document_processor.process_document(tmp_file_path, file_extension)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            if not text:
                st.error("Could not extract text from the uploaded file")
                return False
            
            # Initialize sentence model if not already done
            self.initialize_sentence_model()
            
            # Chunk text
            chunks = self.text_chunker.chunk_text(text)
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Generate embeddings
            embeddings = self.sentence_model.encode(chunks)
            
            # Build vector store
            self.vector_store.build_index(chunks, embeddings)
            
            # Save index for future use
            self.vector_store.save_index("current_document")
            
            logger.info("Document processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            st.error(f"Error processing file: {e}")
            return False
    
    def retrieve_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context for the query"""
        if self.sentence_model is None:
            self.initialize_sentence_model()
        
        # Generate query embedding
        query_embedding = self.sentence_model.encode([query])
        
        # Search for relevant documents
        results = self.vector_store.search(query_embedding, k=k)
        
        if not results:
            return "No relevant context found."
        
        # Combine relevant contexts
        context = "\n\n".join([result['text'] for result in results])
        return context
    
    def generate_answer(self, query: str) -> str:
        """Generate answer using Gemini API with retrieved context"""
        try:
            # Retrieve relevant context
            context = self.retrieve_relevant_context(query)
            
            # Create prompt with context
            prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, please say so.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response using Gemini
            response = self.gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def load_existing_index(self) -> bool:
        """Load existing index if available"""
        return self.vector_store.load_index("current_document")

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="RAG System with Local Dependencies", layout="wide")
    
    st.title("🤖 RAG System with Gemini API")
    st.markdown("Upload a document and ask questions about its content")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Gemini API Key input with validation
        gemini_api_key = st.text_input(
            "Enter Gemini API Key", 
            type="password", 
            help="Get your API key from Google AI Studio: https://makersuite.google.com/app/apikey"
        )
        
        if gemini_api_key:
            # Validate API key format
            if not gemini_api_key.startswith('AIzaSy') or len(gemini_api_key) != 39:
                st.error("❌ Invalid API key format. API key should start with 'AIzaSy' and be 39 characters long.")
                st.stop()
            else:
                st.success("✅ API key format looks correct")
        else:
            st.warning("Please enter your Gemini API key to continue")
            st.info("📝 **How to get API key:**\n1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)\n2. Sign in with Google account\n3. Click 'Create API Key'\n4. Copy and paste here")
            st.stop()
        
        # Model selection
        st.subheader("Model Selection")
        model_options = {
            "Gemini 1.5 Flash (Recommended)": "gemini-1.5-flash",
            "Gemini 1.5 Pro": "gemini-1.5-pro", 
            "Gemini 2.0 Flash": "gemini-2.0-flash-exp",
            "Gemini 1.5 Pro (Latest)": "gemini-1.5-pro-latest"
        }
        
        selected_model_name = st.selectbox(
            "Choose Gemini Model:",
            options=list(model_options.keys()),
            index=0,
            help="Gemini 1.5 Flash is recommended for most use cases - fast and cost-effective"
        )
        
        selected_model = model_options[selected_model_name]
        
        # Advanced settings
        st.subheader("Advanced Settings")
        chunk_size = st.slider("Chunk Size", 100, 1000, 500)
        overlap = st.slider("Chunk Overlap", 0, 100, 50)
        num_results = st.slider("Number of Retrieved Contexts", 1, 10, 3)
    
    # Initialize RAG system with selected model and error handling
    if 'rag_system' not in st.session_state or st.session_state.get('current_model') != selected_model:
        try:
            with st.spinner(f"Initializing RAG system with {selected_model_name}..."):
                st.session_state.rag_system = GeminiRAGSystem(gemini_api_key, selected_model)
                st.session_state.rag_system.text_chunker = TextChunker(chunk_size, overlap)
                st.session_state.current_model = selected_model
                st.success(f"✅ RAG system initialized with {selected_model_name}")
        except ValueError as e:
            st.error(f"❌ Failed to initialize: {str(e)}")
            st.info("💡 **Troubleshooting:**\n- Check your API key\n- Try selecting a different model\n- Ensure you have internet connection")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.stop()
    
    # File upload section
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, DOCX, TXT"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                success = st.session_state.rag_system.process_uploaded_file(uploaded_file)
                if success:
                    st.success("Document processed successfully!")
                    st.session_state.document_processed = True
    
    # Try to load existing index
    elif not hasattr(st.session_state, 'document_processed'):
        with st.spinner("Checking for existing document..."):
            if st.session_state.rag_system.load_existing_index():
                st.info("Loaded previously processed document")
                st.session_state.document_processed = True
    
    # Q&A Section
    if hasattr(st.session_state, 'document_processed') and st.session_state.document_processed:
        st.header("❓ Ask Questions")
        
        # Chat interface
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the document"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response = st.session_state.rag_system.generate_answer(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    else:
        st.info("Please upload a document to start asking questions")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This system runs entirely in your virtual environment with locally downloaded models.")

if __name__ == "__main__":
    main()
