import os
import re
import time
import hashlib
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import numpy as np

# Import database manager
import db_manager

def sanitize_collection_name(name: str) -> str:
    """Sanitize collection name for Chroma DB"""
    name = name.replace(" ", "_")
    name = re.sub(r'[^a-zA-Z0-9_\-]', '', name)
    return name[:63]

def compute_hash(content: str) -> str:
    """Compute a hash for document content to avoid duplicates"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def extract_text_from_pdf_ocr(pdf_path):
    """Extract text from PDF using OCR"""
    try:
        images = convert_from_path(pdf_path, dpi=100)  # Lower DPI to save memory
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
            del image  # Free memory for each image
        del images  # Free memory for the list
        import gc; gc.collect()
        return text
    except Exception as e:
        print(f"❌ OCR failed for {pdf_path}: {str(e)}")
        return ""

def get_existing_hashes(chroma_store):
    """Get existing document hashes from Chroma DB"""
    try:
        results = chroma_store._collection.get(include=["metadatas"])
        return {md["content_hash"] for md in results.get("metadatas", []) if md and "content_hash" in md}
    except Exception as e:
        print(f"⚠️ Failed to get existing hashes: {str(e)}")
        return set()

def ocr_and_update_chroma(doc_dir, persist_dir, collection_name="default_collection", chunk_size: int = 1000, chunk_overlap: int = 200):
    """Process documents with OCR and update Chroma DB
    
    This function processes PDF documents with OCR to extract text,
    then stores the chunks in a persistent ChromaDB collection.
    New chunks are appended to the existing collection, ensuring data
    survives across system restarts.
    
    Args:
        doc_dir: Directory containing the documents to process
        persist_dir: Directory to store the ChromaDB database
        collection_name: Name of the ChromaDB collection to use
        chunk_size: Size of text chunks in characters (default: 1000)
        chunk_overlap: Number of overlapping characters between chunks (default: 200)
    """
    collection_name = sanitize_collection_name(collection_name)
    print(f"🗂 Using collection name for OCR data: `{collection_name}`")
    
    # Connect to persistent Chroma DB
    print(f"🔍 Connecting to Chroma DB at {persist_dir}")
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    # Check if the collection already exists
    existing_collections = chroma_client.list_collections()
    collection_exists = any(col.name == collection_name for col in existing_collections)
    print(f"ℹ️ Collection '{collection_name}' exists: {collection_exists}")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
   
    
    # Initialize Chroma store with the client
    chroma_store = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # Get existing hashes to avoid duplicates
    existing_hashes = get_existing_hashes(chroma_store)
    print(f"ℹ️ Found {len(existing_hashes)} existing OCR document chunks in the database.")
    
    # Configure the text splitter with user-defined settings
    print(f"🔪 Using chunk size: {chunk_size}, overlap: {chunk_overlap} for OCR text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    new_chunks = []

    print("🔎 Scanning and applying OCR to all PDFs...")

    # Track OCR document processing in the database
    processed_docs = {}

    for i, file in enumerate(sorted(os.listdir(doc_dir)), 1):
        path = os.path.join(doc_dir, file)
        if file.lower().endswith(".pdf"):
            print(f"▶️ [{i}] Processing: {file}")
            
            # Generate a document hash for database tracking
            # We'll use the file path as a unique identifier
            doc_hash = compute_hash(path)
            
            # Check if we've already processed this document with OCR
            doc_in_db = db_manager.get_document_by_hash(doc_hash)
            if doc_in_db and doc_in_db.get('processed_with_ocr', False):
                print(f"ℹ️ Document already processed with OCR according to database.")
                # Still process it again but tracking we've seen it
                processed_docs[doc_hash] = True
            
            # Extract text using OCR
            text = extract_text_from_pdf_ocr(path)

            if not text.strip():
                print("   ⚠️ OCR returned empty text.")
                continue

            # Process document chunks
            chunks = text_splitter.split_documents([Document(page_content=text, metadata={})])
            filtered_chunks = []
            chunk_count = 0

            for chunk in chunks:
                chunk_hash = compute_hash(chunk.page_content)
                if chunk_hash not in existing_hashes:
                    chunk.metadata = {
                        "source": path,
                        "file_name": file,
                        "content_hash": chunk_hash,
                        "doc_hash": doc_hash,  # Add document hash for tracking
                        "section": "ocr_recovered",
                        "upload_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    filtered_chunks.append(chunk)
                    # Add to existing hashes to prevent duplicates in this batch
                    existing_hashes.add(chunk_hash)
                    chunk_count += 1

            # Update document in database with OCR status
            try:
                # Use the document's folder as the collection name if needed
                folder_name = os.path.basename(os.path.dirname(path))
                col_name = collection_name
                
                # Check if document exists in database
                if doc_in_db:
                    # Update with OCR status
                    db_manager.add_document(
                        filename=os.path.basename(path),
                        file_path=path,
                        collection_name=col_name,
                        doc_hash=doc_hash,
                        page_count=doc_in_db.get('page_count', 0),
                        chunk_count=doc_in_db.get('chunk_count', 0) + chunk_count,
                        processed_with_ocr=True
                    )
                else:
                    # Add new document with OCR status
                    db_manager.add_document(
                        filename=os.path.basename(path),
                        file_path=path,
                        collection_name=col_name,
                        doc_hash=doc_hash,
                        page_count=0,  # Unknown page count from OCR
                        chunk_count=chunk_count,
                        processed_with_ocr=True
                    )
                processed_docs[doc_hash] = True
                
            except Exception as db_err:
                print(f"⚠️ Error updating document in database: {str(db_err)}")

            if filtered_chunks:
                print(f"   ➕ {len(filtered_chunks)} new chunks to add.")
                new_chunks.extend(filtered_chunks)
            else:
                print("   ℹ️ All OCR chunks already exist in Chroma.")

    if new_chunks:
        print(f"\n🚀 Adding {len(new_chunks)} new OCR-recovered chunks from {len(processed_docs)} documents...")
        try:
            print("DEBUG: About to call chroma_store.add_documents()")
            chroma_store.add_documents(new_chunks)
            print("DEBUG: Finished chroma_store.add_documents()")
        except Exception as e:
            print(f"❌ Error during add_documents: {e}")
        print("✅ Chroma DB updated successfully!")
        
        # Update chunk counts in PostgreSQL for each source document
        try:
            # Group chunks by document (doc_hash)
            doc_chunk_counts = {}
            for chunk in new_chunks:
                if 'doc_hash' in chunk.metadata:
                    doc_hash = chunk.metadata['doc_hash']
                    if doc_hash not in doc_chunk_counts:
                        doc_chunk_counts[doc_hash] = {
                            'count': 0,
                            'source': chunk.metadata.get('source', ''),
                            'file_name': chunk.metadata.get('file_name', '')
                        }
                    doc_chunk_counts[doc_hash]['count'] += 1
            
            # Update each document's chunk count in the database
            for doc_hash, info in doc_chunk_counts.items():
                count = info['count']
                source = info['source']
                filename = info['file_name']
                
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
                        processed_with_ocr=True
                    )
                    print(f"📊 Updated OCR chunk count for {filename}: {current_count} → {current_count + count}")
                else:
                    # Create new document entry
                    db_manager.add_document(
                        filename=filename,
                        file_path=source,
                        collection_name=collection_name,
                        doc_hash=doc_hash,
                        chunk_count=count,
                        processed_with_ocr=True
                    )
                    print(f"📊 Created new document entry for {filename} with {count} OCR chunks")
        except Exception as e:
            print(f"⚠️ Error updating OCR chunk counts in database: {str(e)}")
    else:
        print("✅ No new OCR chunks to add.")
        
    return len(new_chunks)
    return len(new_chunks)
