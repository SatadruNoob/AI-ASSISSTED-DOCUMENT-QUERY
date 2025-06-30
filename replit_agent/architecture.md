# Architecture Overview

## 1. Overview

This repository contains a Document RAG (Retrieval-Augmented Generation) Pipeline application built with Streamlit, LangChain, and Mistral AI. Users can upload PDF documents, process them with OCR, and query their content using natural language. The system uses ChromaDB for vector storage and a relational database (PostgreSQL or SQLite) for document/query metadata. The app is designed for containerized environments and supports both custom and HuggingFace embeddings.

## 2. System Architecture

The system is modular, with these main components:

1. **User Interface Layer**: Streamlit web interface for uploads, queries, and results
2. **Document Processing Layer**: Handles PDF extraction, OCR, chunking, and deduplication
3. **Vector Storage Layer**: ChromaDB for persistent vector storage
4. **Metadata Database Layer**: PostgreSQL/SQLite for document and query metadata
5. **RAG Pipeline Layer**: Retrieval and generation logic
6. **LLM Integration Layer**: Mistral AI API for text generation

### Architecture Diagram

```
┌────────────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│   User Interface   │────▶│  Document Processor │────▶│   Vector Store    │
│    (Streamlit)     │     │ (OCR/Text Splitter) │     │    (ChromaDB)     │
└────────────────────┘     └─────────────────────┘     └─────────┬─────────┘
           ▲                                                      │
           │                                                      │
           │                                                      ▼
┌────────────────────┐                              ┌───────────────────────┐
│    LLM Service     │◀─────────────────────────────│    RAG Pipeline       │
│    (Mistral AI)    │                              │ (Retrieval+Gen)      │
└────────────────────┘                              └─────────┬─────────────┘
           ▲                                                  │
           │                                                  │
           │                                                  ▼
┌────────────────────┐                              ┌───────────────────────┐
│ Metadata Database  │◀─────────────────────────────│   Metadata Manager    │
│ (PostgreSQL/SQLite)│                              │ (db_manager.py)      │
└────────────────────┘                              └───────────────────────┘
```

## 3. Key Components

### 3.1 User Interface (app.py)
- Streamlit-based UI for uploads, queries, and results
- Session state for chat history, processed files, and pipeline state
- Sidebar for document upload and processing settings
- Tabs for querying and database management

### 3.2 Document Processing (ocr_utils.py, vector_store_utils.py)
- PDF text extraction (with and without OCR)
- Chunking with user-configurable size/overlap
- Deduplication using content hashes
- Metadata extraction and cleaning
- Updates both ChromaDB and the metadata database

### 3.3 Vector Storage (ChromaDB)
- Persistent storage in a fixed directory (not per-session)
- Supports both custom and HuggingFace embeddings
- Collection-based organization for logical separation

### 3.4 Metadata Database (db_manager.py)
- Stores document metadata, chunk counts, OCR status, and query history
- Uses PostgreSQL if DATABASE_URL is set, otherwise falls back to SQLite
- Keeps ChromaDB and metadata DB in sync (see sync_chunks.py)

### 3.5 Embedding Strategy (simple_embeddings.py, HuggingFaceEmbeddings)
- Custom deterministic embedding for non-OCR text
- HuggingFace embeddings for OCR and advanced use cases
- Embedding choice is modular and can be extended

### 3.6 RAG Pipeline (rag_pipeline.py)
- Initializes vector store, embeddings, and LLM
- Retrieves relevant chunks using MMR search
- Constructs context and prompts for Mistral AI
- Handles retries and error management
- Stores query/response in metadata DB

## 4. Data Flow

1. **Document Ingestion**:
   - User uploads PDFs via Streamlit
   - Files are saved to an uploads directory
   - Text is extracted (with/without OCR), chunked, and deduplicated
   - Chunks are embedded and stored in ChromaDB
   - Metadata is stored/updated in the database

2. **Query Processing**:
   - User submits a query
   - Query is embedded and used to retrieve relevant chunks from ChromaDB
   - Retrieved context is sent to Mistral AI for answer generation
   - Query, response, and timing are logged in the database

3. **Database Management**:
   - Users can sync chunk counts, rebuild the vector DB, and view document/query history
   - Sync and rebuild operations keep ChromaDB and the metadata DB consistent

## 5. Deployment & Configuration

- **Containerization**: Uses Docker/DevContainer for reproducible environments
- **System Dependencies**: Installed via system-packages.txt in the Dockerfile
- **Python Dependencies**: Managed via requirements-global.txt and pyproject.toml
- **Streamlit Port**: Runs on 8501 (configurable)
- **Environment Variables**: Used for API keys and DB configuration
- **Persistent Storage**: ChromaDB and uploads directory are persistent within the container

## 6. External Dependencies
- **Streamlit**: UI framework
- **LangChain**: RAG pipeline and vector store integration
- **ChromaDB**: Vector database
- **PDF2Image & pytesseract**: PDF and OCR processing
- **NumPy**: Embedding calculations
- **Mistral AI API**: LLM for answer generation
- **PostgreSQL/SQLite**: Metadata storage

## 7. Notes
- The architecture now supports persistent, multi-session document storage and querying
- All document and query metadata is tracked in a relational database
- The system is designed for extensibility and can be adapted for other LLMs or embedding models