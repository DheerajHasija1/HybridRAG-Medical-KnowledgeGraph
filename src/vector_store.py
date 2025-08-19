import os
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        self.vectorstore = None

    def create_vectorstore_from_pdf(self, pdf_path):
        """Create FAISS vectorstore from PDF. (No disk caching)"""
        print(f"📄 Creating FAISS vectorstore from PDF: {pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"📋 Created {len(chunks)} chunks from PDF")
        
        # Create FAISS vectorstore
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("✅ FAISS vectorstore created successfully")
        
        return self.vectorstore

    def similarity_search(self, query, k=3):
        """Search for similar documents"""
        if self.vectorstore is None:
            return []
        
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
