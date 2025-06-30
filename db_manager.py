import os
import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, create_engine, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Any, Optional

# Get database URL from environment, fallback to SQLite if not set
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    print("⚠️ DATABASE_URL not set. Falling back to local SQLite database 'local.db'.")
    DATABASE_URL = "sqlite:///local.db"

# Create database engine
engine = create_engine(DATABASE_URL)

# Create declarative base
Base = declarative_base()

# Define Document model
class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    collection_name = Column(String(255), nullable=False)
    doc_hash = Column(String(64), nullable=False, unique=True)
    page_count = Column(Integer, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    processed_with_ocr = Column(Boolean, default=False)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'file_path': self.file_path,
            'collection_name': self.collection_name,
            'doc_hash': self.doc_hash,
            'page_count': self.page_count,
            'chunk_count': self.chunk_count,
            'processed_with_ocr': self.processed_with_ocr,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }

# Define Query History model
class QueryHistory(Base):
    __tablename__ = 'query_history'
    
    id = Column(Integer, primary_key=True)
    query_text = Column(Text, nullable=False)
    collection_name = Column(String(255), nullable=False)
    query_date = Column(DateTime, default=datetime.datetime.utcnow)
    response_text = Column(Text, nullable=True)
    processing_time = Column(Integer, nullable=True)  # Time in milliseconds
    
    def to_dict(self):
        return {
            'id': self.id,
            'query_text': self.query_text,
            'collection_name': self.collection_name,
            'query_date': self.query_date.isoformat() if self.query_date else None,
            'response_text': self.response_text,
            'processing_time': self.processing_time
        }

# Create all tables in the database
Base.metadata.create_all(engine)

# Create a session factory
Session = sessionmaker(bind=engine)

def add_document(
    filename: str,
    file_path: str,
    collection_name: str,
    doc_hash: str,
    page_count: Optional[int] = None,
    chunk_count: Optional[int] = None,
    processed_with_ocr: bool = False
) -> Document:
    """Add a new document to the database"""
    session = Session()
    try:
        # Check if document with the same hash already exists
        existing_doc = session.query(Document).filter_by(doc_hash=doc_hash).first()
        if existing_doc:
            # Update last accessed time
            existing_doc.last_accessed = datetime.datetime.utcnow()
            session.commit()
            return existing_doc
        
        # Create new document record
        document = Document(
            filename=filename,
            file_path=file_path,
            collection_name=collection_name,
            doc_hash=doc_hash,
            page_count=page_count,
            chunk_count=chunk_count,
            processed_with_ocr=processed_with_ocr
        )
        
        session.add(document)
        session.commit()
        return document
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_document_by_hash(doc_hash: str) -> Optional[Dict[str, Any]]:
    """Get a document by its hash"""
    session = Session()
    try:
        document = session.query(Document).filter_by(doc_hash=doc_hash).first()
        if document:
            # Update last accessed time
            document.last_accessed = datetime.datetime.utcnow()
            session.commit()
            return document.to_dict()
        return None
    finally:
        session.close()

def get_documents_by_collection(collection_name: str) -> List[Dict[str, Any]]:
    """Get all documents in a collection"""
    session = Session()
    try:
        documents = session.query(Document).filter_by(collection_name=collection_name).all()
        return [doc.to_dict() for doc in documents]
    finally:
        session.close()

def get_all_collections() -> List[str]:
    """Get a list of all collection names"""
    session = Session()
    try:
        collections = session.query(Document.collection_name).distinct().all()
        return [col[0] for col in collections]
    finally:
        session.close()

def add_query_history(
    query_text: str,
    collection_name: str,
    response_text: Optional[str] = None,
    processing_time: Optional[int] = None
) -> QueryHistory:
    """Add a query to the history"""
    session = Session()
    try:
        query = QueryHistory(
            query_text=query_text,
            collection_name=collection_name,
            response_text=response_text,
            processing_time=processing_time
        )
        
        session.add(query)
        session.commit()
        return query
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_recent_queries(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent queries from history"""
    session = Session()
    try:
        queries = session.query(QueryHistory).order_by(QueryHistory.query_date.desc()).limit(limit).all()
        return [query.to_dict() for query in queries]
    finally:
        session.close()

def remove_documents_not_in_sources(valid_sources: list):
    """Remove documents from the database whose file_path is not in valid_sources."""
    session = Session()
    try:
        deleted = session.query(Document).filter(~Document.file_path.in_(valid_sources)).delete(synchronize_session=False)
        session.commit()
        print(f"Removed {deleted} old document records from database.")
        return deleted
    finally:
        session.close()