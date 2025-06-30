import os
import sys
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import db_manager

def sync_chunk_counts(persist_dir="/home/runner/workspace/chroma_db"):
    """Sync document chunk counts between ChromaDB and PostgreSQL."""
    print(f"Connecting to ChromaDB at: {persist_dir}")
    
    # Initialize the client
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    # Get all collections
    collections = chroma_client.list_collections()
    print(f"Found {len(collections)} collections in ChromaDB")
    
    total_updated = 0
    
    # Loop through each collection
    for collection in collections:
        collection_name = collection.name
        print(f"\nProcessing collection: {collection_name}")
        
        # Initialize with HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        chroma_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        # Get all documents
        try:
            docs = chroma_store.get()
            if docs and 'documents' in docs:
                # Group documents by source
                source_counts = {}
                
                # Process metadata to count chunks per document
                for metadata in docs.get('metadatas', []):
                    if metadata and 'source' in metadata:
                        source = metadata['source']
                        file_name = metadata.get('file_name', os.path.basename(source))
                        doc_hash = metadata.get('doc_hash', None)
                        
                        # Use document path hash if doc_hash not available
                        if not doc_hash:
                            from vector_store_utils import compute_hash
                            doc_hash = compute_hash(source)
                        
                        # Initialize counters
                        if doc_hash not in source_counts:
                            source_counts[doc_hash] = {
                                'path': source,
                                'file_name': file_name,
                                'count': 0,
                                'processed_with_ocr': metadata.get('section') == 'ocr_recovered'
                            }
                        
                        # Increment counter for this document
                        source_counts[doc_hash]['count'] += 1
                
                # Update PostgreSQL database with correct counts
                for doc_hash, info in source_counts.items():
                    # Check if document exists in database
                    db_doc = db_manager.get_document_by_hash(doc_hash)
                    
                    if db_doc:
                        # Update existing document if count differs
                        if db_doc.get('chunk_count', 0) != info['count']:
                            db_manager.add_document(
                                filename=info['file_name'],
                                file_path=info['path'],
                                collection_name=collection_name,
                                doc_hash=doc_hash,
                                page_count=db_doc.get('page_count', 0),
                                chunk_count=info['count'],  # Set accurate count from ChromaDB
                                processed_with_ocr=db_doc.get('processed_with_ocr', False) or info['processed_with_ocr']
                            )
                            print(f"Updated {info['file_name']} chunk count: {db_doc.get('chunk_count', 0)} → {info['count']}")
                            total_updated += 1
                    else:
                        # Add new document record
                        db_manager.add_document(
                            filename=info['file_name'],
                            file_path=info['path'],
                            collection_name=collection_name,
                            doc_hash=doc_hash,
                            chunk_count=info['count'],
                            processed_with_ocr=info['processed_with_ocr']
                        )
                        print(f"Added new document record for {info['file_name']} with {info['count']} chunks")
                        total_updated += 1
                
        except Exception as e:
            print(f"Error processing collection {collection_name}: {str(e)}")
    
    print(f"\nSync complete. Updated {total_updated} document records.")
    return total_updated

if __name__ == "__main__":
    # Use command line argument for persist_dir if provided
    persist_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/runner/workspace/chroma_db"
    sync_chunk_counts(persist_dir)