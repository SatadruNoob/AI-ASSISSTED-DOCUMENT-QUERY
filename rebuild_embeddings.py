import os
import shutil
import chromadb
import argparse
from vector_store_utils import update_vector_store, sanitize_collection_name
from ocr_utils import ocr_and_update_chroma
import db_manager

def rebuild_vector_database(persist_dir="/home/runner/workspace/chroma_db", 
                           uploads_dir="/home/runner/workspace/uploads",
                           collection_name="main_collection",
                           enable_ocr=True,
                           chunk_size=1000,
                           chunk_overlap=200):
    """
    Rebuilds the ChromaDB vector database with improved embeddings.
    This function deletes the existing ChromaDB database and recreates it
    using the documents in the uploads directory with the new embedding algorithm.
    
    Args:
        persist_dir: Path to the ChromaDB database directory
        uploads_dir: Path to the uploaded documents
        collection_name: Name of the collection to create
        enable_ocr: Whether to use OCR to process documents
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    print(f"🔄 Rebuilding vector database at {persist_dir}")
    print(f"📁 Using uploads directory: {uploads_dir}")
    print(f"📊 Collection name: {collection_name}")
    print(f"⚙️ Settings: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, ocr={enable_ocr}")
    
    # Get information about existing collections
    try:
        # Connect to ChromaDB to get existing collections
        client = chromadb.PersistentClient(path=persist_dir)
        existing_collections = client.list_collections()
        print(f"Found {len(existing_collections)} existing collections")
        
        # Close the client connection
        client = None
    except Exception as e:
        print(f"Error connecting to existing ChromaDB: {str(e)}")
        existing_collections = []
    
    # Backup collection information from the database
    collections_in_db = db_manager.get_all_collections()
    print(f"Found {len(collections_in_db)} collections in the PostgreSQL database")
    
    # Remove the existing Chroma DB directory
    try:
        # Delete the directory
        if os.path.exists(persist_dir):
            print(f"🗑️ Removing existing ChromaDB directory: {persist_dir}")
            shutil.rmtree(persist_dir)
            print(f"✅ Removed {persist_dir}")
        
        # Recreate the directory
        os.makedirs(persist_dir, exist_ok=True)
        print(f"✅ Created new ChromaDB directory: {persist_dir}")
    except Exception as e:
        print(f"❌ Error removing/recreating ChromaDB directory: {str(e)}")
        return False
    
    # Now process all documents in the uploads directory
    if not os.path.exists(uploads_dir):
        print(f"❌ Uploads directory {uploads_dir} doesn't exist")
        return False
    
    # Sanitize the collection name
    safe_collection_name = sanitize_collection_name(collection_name)
    
    # Process all documents with the new embedding model
    print(f"🔍 Processing documents in {uploads_dir}")
    
    try:
        # Process regular documents
        num_chunks = update_vector_store(
            uploads_dir, 
            persist_dir, 
            safe_collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print(f"✅ Added {num_chunks} new chunks from regular document processing")
        
        # Process with OCR if enabled
        if enable_ocr:
            ocr_chunks = ocr_and_update_chroma(
                uploads_dir, 
                persist_dir,
                safe_collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            print(f"✅ Added {ocr_chunks} new chunks from OCR processing")
        
        # After processing all documents and OCR, clean up old DB records
        try:
            from chroma_stats import get_chroma_stats
            stats = get_chroma_stats(persist_dir)
            all_sources = []
            for col in stats.values():
                all_sources.extend(col.get('sources', []))
            removed = db_manager.remove_documents_not_in_sources(all_sources)
            print(f"🧹 Database cleanup complete. Removed {removed} old document records.")
        except Exception as cleanup_err:
            print(f"⚠️ Error during database cleanup: {cleanup_err}")

        print("🚀 Vector database rebuild complete!")
        return True
    except Exception as e:
        print(f"❌ Error during vector database rebuild: {str(e)}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Rebuild the vector database with improved embeddings")
    parser.add_argument("--persist-dir", default="/home/runner/workspace/chroma_db", 
                       help="Directory where ChromaDB is stored")
    parser.add_argument("--uploads-dir", default="/home/runner/workspace/uploads", 
                       help="Directory containing uploaded documents")
    parser.add_argument("--collection", default="main_collection", 
                       help="Collection name")
    parser.add_argument("--chunk-size", type=int, default=1000, 
                       help="Chunk size for document splitting")
    parser.add_argument("--chunk-overlap", type=int, default=200, 
                       help="Chunk overlap for document splitting")
    parser.add_argument("--no-ocr", action="store_true", 
                       help="Disable OCR processing")
    
    args = parser.parse_args()
    
    # Run the rebuild function
    rebuild_vector_database(
        persist_dir=args.persist_dir,
        uploads_dir=args.uploads_dir,
        collection_name=args.collection,
        enable_ocr=not args.no_ocr,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )