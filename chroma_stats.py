import os
import chromadb
import json
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma


def get_chroma_stats(persist_dir="/home/runner/workspace/chroma_db"):
    """Get statistics about ChromaDB collections and documents."""
    print(f"Connecting to ChromaDB at: {persist_dir}")

    # Initialize the client
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    # Get all collections
    collections = chroma_client.list_collections()
    print(f"Found {len(collections)} collections in ChromaDB")

    stats = {}

    # Loop through each collection and get document counts
    for collection in collections:
        collection_name = collection.name
        print(f"\nCollection: {collection_name}")

        # Initialize with HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        chroma_store = Chroma(client=chroma_client,
                              collection_name=collection_name,
                              embedding_function=embeddings)

        # Get all documents (be careful with large collections)
        try:
            docs = chroma_store.get()
            if docs and 'documents' in docs:
                doc_count = len(docs['documents'])
                print(f"  Documents: {doc_count}")

                # Count unique document sources
                sources = set()
                for metadata in docs.get('metadatas', []):
                    if metadata and 'source' in metadata:
                        sources.add(metadata['source'])

                print(f"  Unique sources: {len(sources)}")
                print(f"  Sources: {list(sources)}")

                stats[collection_name] = {
                    'chunk_count': doc_count,
                    'source_count': len(sources),
                    'sources': list(sources)
                }
            else:
                print("  No documents found")
                stats[collection_name] = {
                    'chunk_count': 0,
                    'source_count': 0,
                    'sources': []
                }
        except Exception as e:
            print(f"  Error accessing collection: {str(e)}")
            stats[collection_name] = {'error': str(e)}

    return stats


if __name__ == "__main__":
    stats = get_chroma_stats()
    print("\n=== SUMMARY ===")
    print(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
