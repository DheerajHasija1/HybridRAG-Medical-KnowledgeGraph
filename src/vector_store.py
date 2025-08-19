import os
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle  # For saving/loading FAISS index

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

    def create_vectorstore_from_pdf_cached(self, pdf_path, persist_dir="./faiss_db"):
        """Create or load existing vector store from disk cache"""
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "faiss.pkl")
        os.makedirs(persist_dir, exist_ok=True)

        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            print(f"✅ Loading existing FAISS vector DB from {persist_dir}")
            # Load index and docstore
            with open(meta_path, "rb") as f:
                stored = pickle.load(f)
            self.vectorstore = FAISS.load_local(
                faiss_path, 
                self.embeddings,
                stored,
            )
        else:
            print(f"📄 Creating new vector DB from PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            print(f"📋 Created {len(chunks)} chunks from PDF")
            
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            self.vectorstore.save_local(faiss_path)
            # Store extra metadata (docstore etc.)
            with open(meta_path, "wb") as f:
                pickle.dump(self.vectorstore.docstore, f)
            print(f"💾 FAISS Vector DB saved to {persist_dir}")
        return self.vectorstore

    def similarity_search(self, query, k=3):
        if self.vectorstore is None:
            return []
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

