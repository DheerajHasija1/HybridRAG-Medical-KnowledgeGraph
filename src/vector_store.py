import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def create_vectorstore_from_pdf(pdf_path, persist_dir="faiss_vectorstore"):
    """Create and cache FAISS vectorstore from PDF"""
    
    # Check if vectorstore already exists
    if os.path.exists(persist_dir):
        print(f"✅ Loading existing vectorstore from {persist_dir}")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.load_local(
            persist_dir, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"✅ Vectorstore loaded with {vectorstore.index.ntotal} vectors")
        return vectorstore
    
    print(f"📄 Creating FAISS vectorstore from PDF: {pdf_path}")
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased for better context
        chunk_overlap=200,  # Increased overlap
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📋 Created {len(chunks)} chunks from PDF")
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save vectorstore
    vectorstore.save_local(persist_dir)
    print(f"✅ FAISS vectorstore created and saved to {persist_dir}")
    
    return vectorstore

def similarity_search(vectorstore, query, k=10):
    """Search similar documents"""
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []
    