import os
from langchain_community.vectorstores import Chroma
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

    def create_vectorstore_from_pdf_cached(self, pdf_path, persist_dir="./chroma_db"):
        """Create or load existing vector store from disk cache"""
        
        # Check if vector DB already exists
        if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
            print(f"✅ Loading existing vector DB from {persist_dir}")
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
        else:
            print(f"📄 Creating new vector DB from PDF: {pdf_path}")
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            print(f"📋 Created {len(chunks)} chunks from PDF")
            
            # Create vector store and persist to disk
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            self.vectorstore.persist()  # Save to disk
            print(f"💾 Vector DB saved to {persist_dir}")
        
        return self.vectorstore

    def similarity_search(self, query, k=3):
        if self.vectorstore is None:
            return []
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
        