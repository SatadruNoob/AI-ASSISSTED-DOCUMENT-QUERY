# Workaround for Streamlit watcher error with torch
import sys
import os
if "TORCH_DISABLE_WATCHER" not in os.environ:
    os.environ["TORCH_DISABLE_WATCHER"] = "1"
    sys.modules["torch.classes"] = type(sys)("torch.classes")

import os
import time
import tempfile
import streamlit as st
import chromadb
from rag_pipeline import initialize_rag_pipeline, process_query
from ocr_utils import ocr_and_update_chroma
from vector_store_utils import update_vector_store, sanitize_collection_name
import db_manager


# === ADD SECRET VALIDATION HERE ===
# Validate required environment variables
required_vars = [
    "HUGGINGFACEHUB_API_TOKEN",
    "LANGSMITH_API_KEY",
    "MISTRAL_API_KEY"
]

missing_vars = [var for var in required_vars if var not in os.environ]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.stop()  # This will halt the app if variables are missing


# Set page configuration
st.set_page_config(
    page_title="Document RAG Pipeline",
    page_icon="📚",
    layout="wide",
)

# Create a persistent directory for the Chroma DB
PERSISTENT_DIR = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(PERSISTENT_DIR, exist_ok=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = PERSISTENT_DIR
    print(f"DEBUG: Chroma DB persistent directory: {st.session_state.persist_dir}")

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Check if Chroma DB has existing data
try:
    # Use the default collection name
    collection_name = "main_collection"
    # Initialize the RAG pipeline with any existing data
    st.session_state.pipeline = initialize_rag_pipeline(
        st.session_state.persist_dir,
        collection_name
    )

    # Check if the collection exists with data
    client = chromadb.PersistentClient(path=st.session_state.persist_dir)
    existing_collections = client.list_collections()
    has_collections = any(col.name == collection_name for col in existing_collections)

    if has_collections:
        # Set documents_processed flag if the collection exists
        st.session_state.documents_processed = True
        print(f"INFO: Found existing collection '{collection_name}' - Ready for queries")

        # Auto-sync ChromaDB with PostgreSQL database on startup
        try:
            from sync_chunks import sync_chunk_counts
            updated = sync_chunk_counts(st.session_state.persist_dir)
            print(f"INFO: Auto-synced document counts on startup - updated {updated} records")
        except Exception as e:
            print(f"WARNING: Error during auto-sync on startup: {str(e)}")
    else:
        st.session_state.documents_processed = False
        print("INFO: No existing collections found - Need to process documents first")
except Exception as e:
    st.session_state.documents_processed = False
    st.session_state.pipeline = None
    print(f"DEBUG: Could not initialize from existing data: {str(e)}")


# Set environment variables for API keys - In production, these should be fetched from secure storage
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "TYPE_YOUR_TOKEN_NUMBER")
# os.environ["LANGSMITH_TRACING"] = os.environ.get("LANGSMITH_TRACING", "true")
# os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY", "TYPE_YOUR_TOKEN_NUMBER")
# os.environ["MISTRAL_API_KEY"] = os.environ.get("MISTRAL_API_KEY", "TYPE_YOUR_TOKEN_NUMBER")

# Title and description
st.title("🔍 Document RAG Pipeline with Mistral AI")
st.markdown("""
This application allows you to upload PDF documents, process them with OCR capabilities, 
and query them using a LangChain RAG pipeline powered by Mistral AI.
""")

# Sidebar for document upload and processing
with st.sidebar:
    st.header("📄 Document Processing")

    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    collection_name = st.text_input(
        "Collection Name", 
        "default_collection",
        help="Name for your document collection in the vector database"
    )

    # Advanced settings expander
    with st.expander("Advanced Processing Settings"):
        col1, col2 = st.columns(2)

        with col1:
            chunk_size = st.number_input(
                "Chunk Size (characters)",
                min_value=100,
                max_value=4000,
                value=1000,
                step=100,
                help="Number of characters per chunk. Smaller chunks are better for specific queries, larger chunks provide more context."
            )

        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=1000,
                value=200,
                step=50,
                help="Number of overlapping characters between chunks. Helps maintain context between chunks."
            )

        st.info("ℹ️ Recommended settings: Chunk size 1000-2000 for general documents, with an overlap of 10-20% of chunk size.")

    enable_ocr = st.checkbox(
        "Enable OCR", 
        value=True,
        help="Use Optical Character Recognition for processing documents"
    )

    process_button = st.button("Process Documents")

    if process_button and uploaded_files:
        with st.spinner("Processing documents..."):
            # Create an uploads directory if it doesn't exist
            uploads_dir = os.path.join(os.getcwd(), "uploads")
            os.makedirs(uploads_dir, exist_ok=True)

            # Save uploaded files to the uploads directory
            file_paths = []
            for uploaded_file in uploaded_files:
                # Create a unique filename to avoid overwriting
                base_name = os.path.splitext(uploaded_file.name)[0]
                file_ext = os.path.splitext(uploaded_file.name)[1]
                file_path = os.path.join(uploads_dir, uploaded_file.name)

                # Save the file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

                # Add the file to the processed files set for tracking
                st.session_state.processed_files.add(uploaded_file.name)

            # Process files
            try:
                # Check if collection name was provided, else use "main_collection"
                if collection_name == "default_collection":
                    collection_name = "main_collection"

                # Sanitize the collection name
                safe_collection_name = sanitize_collection_name(collection_name)
                st.session_state.collection_name = safe_collection_name

                # Update vector store with user-defined chunk settings
                update_vector_store(
                    uploads_dir, 
                    st.session_state.persist_dir, 
                    safe_collection_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                # Perform OCR if enabled (with user-defined chunk settings)
                if enable_ocr:
                    ocr_and_update_chroma(
                        uploads_dir, 
                        st.session_state.persist_dir,
                        safe_collection_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

                # Initialize RAG pipeline
                st.session_state.pipeline = initialize_rag_pipeline(
                    st.session_state.persist_dir,
                    safe_collection_name
                )

                st.session_state.documents_processed = True
                st.success(f"✅ {len(file_paths)} documents processed successfully!")

            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")

# Create tabs for main application and database management
tab_query, tab_db = st.tabs(["Query Documents", "Database Management"])

with tab_query:
    # Main content area
    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("💬 Chat with Your Documents")

        # Query input
        query = st.text_input("Ask a question about your documents", key="query_input")

        # Process query button
        query_button = st.button("Send Query")

        if query_button and query:
            if not st.session_state.documents_processed:
                st.warning("⚠️ Please upload and process documents first.")
            else:
                with st.spinner("Generating response..."):
                    # Process the query
                    try:
                        # Check if pipeline is initialized
                        if st.session_state.pipeline is not None:
                            start_time = time.time()
                            result = process_query(st.session_state.pipeline, query)
                            end_time = time.time()

                            # Add to chat history
                            st.session_state.chat_history.append({
                                "query": query,
                                "result": result,
                                "time": end_time - start_time
                            })

                            # Force refresh
                            st.rerun()
                        else:
                            st.error("❌ Pipeline is not initialized. Please upload and process documents first.")
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")

    with col2:
        st.header("📋 Results")

        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            with st.expander(f"Q: {chat['query']}", expanded=(i == len(st.session_state.chat_history) - 1)):
                # Query details
                st.markdown(f"**Query:** {chat['query']}")
                st.markdown(f"**Processing Time:** {chat['time']:.2f} seconds")

                # Response tabs
                tab1, tab2 = st.tabs(["Answer", "Retrieved Chunks"])

                with tab1:
                    st.markdown(f"**Response:**\n{chat['result']['answer']}")

                with tab2:
                    st.markdown("**Retrieved Document Chunks:**")
                    # Display document information in a more streamlined way without nested expanders
                    for j, doc in enumerate(chat['result']['source_documents']):
                        st.markdown(f"**Chunk {j+1} - {doc.metadata.get('file_name', 'Unknown')}**")
                        st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                        st.markdown("**Content:**")
                        st.markdown(doc.page_content)
                        st.markdown("---")

with tab_db:
    st.header("📊 Database Management")

    # Add sync button at the top of the Database tab
    if st.button("🔄 Sync Document Counts", help="Synchronize document chunk counts between ChromaDB and PostgreSQL"):
        with st.spinner("Synchronizing chunk counts..."):
            try:
                # Import and run the sync function
                from sync_chunks import sync_chunk_counts
                updated = sync_chunk_counts(st.session_state.persist_dir)
                st.success(f"✅ Sync complete! Updated {updated} document records.")
                # Force refresh to show updated counts
                time.sleep(1)  # Brief pause to ensure database writes complete
                st.rerun()
            except Exception as e:
                st.error(f"❌ Sync error: {str(e)}")

    # Add advanced management options in an expander
    with st.expander("⚙️ Advanced Management Options"):
        st.warning("⚠️ The following operations are potentially destructive and may take some time to complete.")

        # Rebuild database button
        if st.button("🔄 Rebuild Vector Database", help="Rebuilds the entire vector database with improved embeddings. This will process all documents again."):
            with st.spinner("Rebuilding vector database... This may take a while."):
                try:
                    # Import and run the rebuild function
                    from rebuild_embeddings import rebuild_vector_database
                    success = rebuild_vector_database(
                        persist_dir=st.session_state.persist_dir,
                        uploads_dir=os.path.join(os.getcwd(), "uploads"),
                        collection_name="main_collection",
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    if success:
                        st.success("✅ Vector database successfully rebuilt!")
                        # Re-initialize pipeline
                        st.session_state.pipeline = initialize_rag_pipeline(
                            st.session_state.persist_dir,
                            "main_collection"
                        )
                        time.sleep(1)  # Brief pause to ensure everything is ready
                        st.rerun()
                    else:
                        st.error("❌ Failed to rebuild vector database.")
                except Exception as e:
                    st.error(f"❌ Error during database rebuild: {str(e)}")

    # Create tabs for different database views
    db_tab1, db_tab2 = st.tabs(["Documents", "Query History"])

    with db_tab1:
        st.subheader("📑 Indexed Documents")

        # Get all collections from the database
        try:
            collections = db_manager.get_all_collections()

            if collections:
                # Create select box for collection
                selected_collection = st.selectbox(
                    "Select a Collection", 
                    options=collections,
                    help="View documents in a specific collection"
                )

                # Get documents for the selected collection
                docs = db_manager.get_documents_by_collection(selected_collection)

                if docs:
                    # Display documents as a table
                    st.write(f"Found {len(docs)} documents in collection: {selected_collection}")

                    # Convert to a more displayable format
                    display_docs = []
                    for doc in docs:
                        display_docs.append({
                            "Filename": doc["filename"],
                            "Pages": doc["page_count"] or 0,
                            "Chunks": doc["chunk_count"] or 0,
                            "OCR": "✅" if doc["processed_with_ocr"] else "❌",
                            "Upload Date": doc["upload_date"]
                        })

                    st.dataframe(display_docs, use_container_width=True)

                    # Add a download button for the document list
                    st.download_button(
                        "Download Document List",
                        data=str(display_docs),
                        file_name=f"{selected_collection}_document_list.txt",
                        mime="text/plain"
                    )
                else:
                    st.info(f"No documents found in collection: {selected_collection}")
            else:
                st.info("No document collections found in the database.")
        except Exception as e:
            st.error(f"Error accessing document database: {str(e)}")

    with db_tab2:
        st.subheader("🔍 Recent Queries")

        try:
            # Get recent queries
            queries = db_manager.get_recent_queries(limit=20)

            if queries:
                # Display queries as a table
                st.write(f"Showing {len(queries)} most recent queries")

                # Convert to a more displayable format
                display_queries = []
                for q in queries:
                    display_queries.append({
                        "Query": q["query_text"],
                        "Collection": q["collection_name"],
                        "Time (ms)": q["processing_time"] or 0,
                        "Date": q["query_date"]
                    })

                st.dataframe(display_queries, use_container_width=True)
            else:
                st.info("No query history found in the database.")
        except Exception as e:
            st.error(f"Error accessing query history: {str(e)}")

# Footer
st.markdown("---")

# Display the persist directory path (for debug purposes)
st.markdown(f"**DB Storage Path:** `{st.session_state.persist_dir}`")

st.markdown(
    "Built with LangChain, Mistral AI, and Streamlit. "
    "This application uses a RAG pipeline to process and query documents."
)
