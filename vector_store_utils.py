import os
import re
import time
import json
import hashlib
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb
import numpy as np

# Import our SimpleEmbeddings class
from simple_embeddings import SimpleEmbeddings


# Import our database manager
import db_manager

def sanitize_collection_name(name: str) -> str:
    """Sanitize collection name for Chroma DB"""
    name = name.replace(" ", "_")
    name = re.sub(r'[^a-zA-Z0-9_\-]', '', name)
    return name[:63]

def compute_hash(content: str) -> str:
    """Compute SHA-256 hash of the document content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_existing_hashes(chroma_store) -> set:
    """Get existing document hashes from Chroma DB"""
    existing_hashes = set()
    try:
        results = chroma_store._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        for metadata in metadatas:
            if metadata and "content_hash" in metadata:
                existing_hashes.add(metadata["content_hash"])
    except Exception as e:
        print(f"⚠️ Error retrieving existing hashes from Chroma: {str(e)}")
    return existing_hashes

def extract_clean_metadata(raw_metadata, file_path):
    """Extract and clean metadata from PDF"""
    cleaned = {
        "source": file_path,
        "file_name": os.path.basename(file_path)
    }
    if raw_metadata:
        for key, value in raw_metadata.items():
            # Only include serializable and clean fields
            if isinstance(key, str) and isinstance(value, (str, int, float)):
                cleaned[key] = str(value)
    return cleaned

def load_documents_from_directory(docs_path: str):
    """Load documents with clean, reliable PDF metadata handling"""
    docs = []
    for root, _, files in os.walk(docs_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(".pdf"):
                try:
                    # Process PDF file
                    raw_reader = PdfReader(file_path)
                    pdf_metadata = extract_clean_metadata(raw_reader.metadata, file_path)
                    page_count = len(raw_reader.pages)

                    # Generate a document hash based on the first page content
                    first_page_text = raw_reader.pages[0].extract_text() if page_count > 0 else ""
                    document_hash = compute_hash(first_page_text)

                    # Store document info in the PostgreSQL database
                    try:
                        # Use folder name as collection name if not specified
                        folder_name = os.path.basename(os.path.dirname(file_path))
                        col_name = sanitize_collection_name(folder_name) if folder_name else "main_collection"

                        # Add document to database
                        db_manager.add_document(
                            filename=os.path.basename(file_path),
                            file_path=file_path,
                            collection_name=col_name,
                            doc_hash=document_hash,
                            page_count=page_count,
                            processed_with_ocr=False
                        )
                    except Exception as db_err:
                        print(f"⚠️ Error adding document to database: {str(db_err)}")

                    # Process each page
                    for page_num, page in enumerate(raw_reader.pages):
                        text = page.extract_text()
                        if text and text.strip():  # Avoid empty pages
                            # Add the content hash to metadata
                            content_hash = compute_hash(text)
                            page_metadata = {
                                **pdf_metadata, 
                                "page": page_num,
                                "doc_hash": document_hash,
                                "content_hash": content_hash
                            }
                            docs.append(Document(
                                page_content=text,
                                metadata=page_metadata
                            ))
                except Exception as e:
                    print(f"⚠️ Error loading {file_path}: {str(e)}")
            elif file.endswith(".txt"):
                try:
                    # Process text file
                    loader = TextLoader(file_path)
                    text_docs = loader.load()

                    # Generate a document hash
                    text_content = "\n".join([doc.page_content for doc in text_docs])
                    document_hash = compute_hash(text_content)

                    # Store document info in database
                    try:
                        folder_name = os.path.basename(os.path.dirname(file_path))
                        col_name = sanitize_collection_name(folder_name) if folder_name else "main_collection"

                        db_manager.add_document(
                            filename=os.path.basename(file_path),
                            file_path=file_path,
                            collection_name=col_name,
                            doc_hash=document_hash,
                            page_count=1,  # Text files are treated as single page
                            processed_with_ocr=False
                        )
                    except Exception as db_err:
                        print(f"⚠️ Error adding text document to database: {str(db_err)}")

                    # Add document hash to metadata
                    for doc in text_docs:
                        if not doc.metadata:
                            doc.metadata = {}
                        doc.metadata["doc_hash"] = document_hash
                        doc.metadata["content_hash"] = compute_hash(doc.page_content)
                        doc.metadata["file_name"] = os.path.basename(file_path)
                        doc.metadata["source"] = file_path

                    docs.extend(text_docs)
                except Exception as e:
                    print(f"⚠️ Error loading text file {file_path}: {str(e)}")

    return docs

def split_and_prepare_documents(docs, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into smaller chunks for Chroma."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_splits = text_splitter.split_documents(docs)

    # Track chunks per document for database updates
    doc_chunk_counts = {}

    # Add additional metadata to chunks
    for i, chunk in enumerate(all_splits):
        # Add chunk number to metadata
        chunk.metadata["chunk_id"] = i
        # Add timestamp to metadata
        chunk.metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Ensure content_hash is in metadata
        if "content_hash" not in chunk.metadata:
            chunk.metadata["content_hash"] = compute_hash(chunk.page_content)

        # Track chunks per document using doc_hash
        if "doc_hash" in chunk.metadata:
            doc_hash = chunk.metadata["doc_hash"]
            if doc_hash in doc_chunk_counts:
                doc_chunk_counts[doc_hash] += 1
            else:
                doc_chunk_counts[doc_hash] = 1

    # Update database with chunk counts
    for doc_hash, chunk_count in doc_chunk_counts.items():
        try:
            # Get document by hash
            doc_info = db_manager.get_document_by_hash(doc_hash)
            if doc_info:
                # Update document with chunk count
                db_manager.add_document(
                    filename=doc_info['filename'],
                    file_path=doc_info['file_path'],
                    collection_name=doc_info['collection_name'],
                    doc_hash=doc_info['doc_hash'],
                    page_count=doc_info['page_count'],
                    chunk_count=chunk_count,
                    processed_with_ocr=doc_info['processed_with_ocr']
                )
        except Exception as e:
            print(f"⚠️ Error updating chunk count in database: {str(e)}")

    return all_splits

def update_vector_store(docs_path: str, persist_dir: str, collection_name: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Update the vector store with new documents.

    This function loads documents from a directory, splits them into chunks,
    and appends new chunks to an existing ChromaDB collection or creates a
    new collection if it doesn't exist. It ensures documents survive system restarts
    by storing them in a fixed location.

    Args:
        docs_path: Directory containing the documents to process
        persist_dir: Directory to store the ChromaDB database
        collection_name: Name of the ChromaDB collection to use
        chunk_size: Size of text chunks in characters (default: 1000)
        chunk_overlap: Number of overlapping characters between chunks (default: 200)
    """
    # Load documents
    print(f"🔍 Loading documents from: {docs_path}")
    try:
        docs = load_documents_from_directory(docs_path)

        if not docs:
            print("⚠️ No documents loaded.")
            return 0
    except Exception as e:
        print(f"⚠️ Error loading documents: {str(e)}")
        raise Exception(f"Failed to load documents: {str(e)}")

    print(f"📄 Loaded {len(docs)} document pages.")
    print(f"✂️ Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    chunks = split_and_prepare_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"🧩 Created {len(chunks)} chunks.")

    # Initialize embeddings
    embeddings = SimpleEmbeddings()

    # Connect to persistent Chroma DB
    print(f"💾 Connecting to Chroma DB at {persist_dir} with collection: {collection_name}")
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    # Check if the collection already exists in the database
    existing_collections = chroma_client.list_collections()
    collection_exists = any(col.name == collection_name for col in existing_collections)

    # Initialize Chroma store with the client
    chroma_store = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # Get existing hashes to avoid duplicates
    existing_hashes = get_existing_hashes(chroma_store)
    print(f"ℹ️ Found {len(existing_hashes)} existing document chunks in the database.")

    # Filter out chunks that already exist
    new_chunks = []
    for chunk in chunks:
        chunk_hash = compute_hash(chunk.page_content)
        if chunk_hash not in existing_hashes:
            chunk.metadata["content_hash"] = chunk_hash
            # Add upload timestamp for tracking
            chunk.metadata["upload_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            new_chunks.append(chunk)

    if new_chunks:
        try:
            print(f"➕ Adding {len(new_chunks)} new chunks to Chroma DB.")
            # Process chunks in smaller batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(new_chunks), batch_size):
                batch = new_chunks[i:i + batch_size]
                try:
                    chroma_store.add_documents(batch)
                    print(f"✅ Added batch {i//batch_size + 1}/{(len(new_chunks)-1)//batch_size + 1}")
                except Exception as batch_error:
                    print(f"⚠️ Error in batch {i//batch_size + 1}: {str(batch_error)}")
                    continue
            
            chroma_store.persist()
            print("✅ Chroma DB updated successfully!")
        except Exception as e:
            print(f"❌ Error updating vector store: {str(e)}")
            return 0

        # Update chunk counts in PostgreSQL for each source document
        try:
            # Group chunks by source document
            doc_chunk_counts = {}
            for chunk in new_chunks:
                if 'source' in chunk.metadata:
                    source = chunk.metadata['source']
                    if source not in doc_chunk_counts:
                        doc_chunk_counts[source] = 0
                    doc_chunk_counts[source] += 1

            # Update each document's chunk count in the database
            for source, count in doc_chunk_counts.items():
                filename = os.path.basename(source)
                doc_hash = compute_hash(source)

                # Get current document from database
                doc_in_db = db_manager.get_document_by_hash(doc_hash)

                if doc_in_db:
                    # Add to existing chunk count
                    current_count = doc_in_db.get('chunk_count', 0) or 0
                    db_manager.add_document(
                        filename=filename,
                        file_path=source,
                        collection_name=collection_name,
                        doc_hash=doc_hash,
                        page_count=doc_in_db.get('page_count', 0),
                        chunk_count=current_count + count,
                        processed_with_ocr=doc_in_db.get('processed_with_ocr', False)
                    )
                    print(f"📊 Updated chunk count for {filename}: {current_count} → {current_count + count}")
                else:
                    # Create new document entry
                    db_manager.add_document(
                        filename=filename,
                        file_path=source,
                        collection_name=collection_name,
                        doc_hash=doc_hash,
                        chunk_count=count,
                        processed_with_ocr=False
                    )
                    print(f"📊 Created new document entry for {filename} with {count} chunks")
        except Exception as e:
            print(f"⚠️ Error updating chunk counts in database: {str(e)}")
    else:
        print("ℹ️ No new chunks to add.")

    # Return summary information about the update
    return len(new_chunks)