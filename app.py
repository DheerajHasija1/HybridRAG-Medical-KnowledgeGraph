import streamlit as st
import os
from src.vector_store import VectorStore
from src.knowledge_graph import KnowledgeGraph
from src.hybrid_agent import HybridRAGAgent

PDF_PATH = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"

@st.cache_resource(show_spinner=False)
def initialize_system_cached():
    if not os.path.exists(PDF_PATH):
        st.error(f"PDF file not found: {PDF_PATH}")
        return None

    vs = VectorStore()
    vs.create_vectorstore_from_pdf(PDF_PATH)

    kg = KnowledgeGraph()
    kg.build_graph_from_pdf_cached(PDF_PATH)

    agent = HybridRAGAgent(vs, kg)
    return agent

def main():
    st.title("Medical Encyclopedia Hybrid RAG")
    st.write("Ask questions from Gale Encyclopedia of Medicine")

    with st.spinner("Loading system..."):
        agent = initialize_system_cached()

    if agent is None:
        st.stop()

    st.success("System ready!")

    st.subheader("Ask Medical Questions")
    query = st.text_input("Your question:")

    if st.button("Search") and query:
        with st.spinner("Searching..."):
            try:
                result = agent.query(query)

                st.subheader("AI Response")
                st.write(result["response"])

                with st.expander("Search Details"):
                    st.write("Vector Search Results:")
                    for i, doc in enumerate(result["vector_results"], 1):
                        st.write(f"{i}. {doc[:200]}...")

                    st.write("Knowledge Graph Context:")
                    st.write(result["graph_context"])

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
