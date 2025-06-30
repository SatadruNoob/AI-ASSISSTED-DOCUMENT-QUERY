import os
import time
import json
from typing import Dict, Any, List
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai.chat_models import ChatMistralAI
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
import db_manager

# Add tenacity import
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

def initialize_rag_pipeline(persist_dir: str, collection_name: str = "default_collection"):
    """Initialize the RAG pipeline components"""

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    # Initialize vector store
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # Initialize Mistral LLM
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.2,
        max_tokens=1024
    )

    return {
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "llm": llm
    }

# Retry decorator for LLM invocation
def is_rate_limit_error(exception):
    msg = str(exception).lower()
    return "429" in msg or "rate limit" in msg

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception(is_rate_limit_error)
)
def invoke_llm_with_retry(llm, prompt):
    return llm.invoke(prompt)

def process_query(pipeline: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Process a user query through the RAG pipeline"""
    start_time = time.time()
    
    try:
        # Validate pipeline components
        if not isinstance(pipeline, dict) or not all(k in pipeline for k in ["vectorstore", "llm"]):
            raise ValueError("Invalid pipeline configuration")
            
        vectorstore = pipeline["vectorstore"]
        llm = pipeline["llm"]

        # Get collection name safely
        collection_name = "main_collection"
        try:
            if hasattr(vectorstore, '_collection') and hasattr(vectorstore._collection, 'name'):
                collection_name = vectorstore._collection.name
        except Exception as e:
            print(f"Warning: Could not get collection name: {str(e)}")

        # Analysis to determine retrieval strategy
        query_analysis = analyze_query_complexity(query)
        k = 8 if query_analysis["complexity"] == "complex" else 5

        # Perform document retrieval with error handling
        try:
            docs = vectorstore.max_marginal_relevance_search(
                query, 
                k=k,
                fetch_k=k*3,
                lambda_mult=0.7
            )
            if not docs:
                return {
                    "answer": "No relevant documents found for your query.",
                    "source_documents": [],
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
        except Exception as e:
            print(f"Error during document retrieval: {str(e)}")
            raise

        # Build context from retrieved documents with document titles
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('file_name', 'Unknown Document')
            formatted_docs.append(f"Document {i+1} ({source}):\n{doc.page_content}")

        context = "\n\n---\n\n".join(formatted_docs)

        # Construct the improved prompt
        prompt = f"""You are a helpful assistant answering questions based on the provided document sections.

        CONTEXT INFORMATION:
        {context}

        USER QUESTION:
        {query}

        Instructions:
        1. Answer the question ONLY based on the information in the provided context. 
        2. Look carefully at ALL document sections before determining if you have enough information.
        3. If you can't find a direct answer but there's relevant information, synthesize what's available.
        4. If none of the documents contain relevant information, respond with "I don't have enough information to answer this question based on the provided documents."
        5. Include specific information from the documents to support your answer.
        6. Keep your answer concise and focused on the question.
        7. DO NOT make up or hallucinate any information not found in the documents.
        8. If the documents seem to contradict each other, acknowledge this in your answer.

        ANSWER:
        """

        # Generate answer with improved error handling
        try:
            # Use retry logic for LLM invocation
            response = invoke_llm_with_retry(llm, prompt)
            
            # Handle different response formats
            if hasattr(response, 'content'):
                answer = response.content
            elif isinstance(response, dict):
                if 'content' in response:
                    answer = response['content']
                else:
                    answer = json.dumps(response)  # Safely handle dict response
            elif isinstance(response, str):
                answer = response
            else:
                answer = str(response)
                
            # Validate answer is not empty
            if not answer or not answer.strip():
                answer = "I apologize, but I couldn't generate a meaningful response."
                
        except Exception as e:
            print(f"Error in LLM response generation: {str(e)}")
            answer = f"I apologize, but I encountered an error while processing your query. Error: {str(e)}"

        # Calculate processing time in milliseconds
        processing_time = int((time.time() - start_time) * 1000)

        # Store query in the database
        try:
            db_manager.add_query_history(
                query_text=query,
                collection_name=collection_name,
                response_text=answer,
                processing_time=processing_time
            )
        except Exception as e:
            print(f"⚠️ Error storing query in database: {str(e)}")

        return {
            "answer": answer,
            "source_documents": docs,
            "processing_time_ms": processing_time
        }

    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        return {
            "answer": f"I apologize, but I encountered an error while processing your query: {str(e)}",
            "source_documents": [],
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

def analyze_query_complexity(query: str) -> Dict[str, Any]:
    """Analyze the complexity of the query to determine retrieval strategy"""
    # Simple analysis for demonstration
    words = query.split()
    complexity = "simple" if len(words) < 8 else "complex"

    return {
        "complexity": complexity,
        "word_count": len(words),
        "question_words": [w for w in words if w.lower() in ["what", "why", "how", "when", "where", "who"]]
    }