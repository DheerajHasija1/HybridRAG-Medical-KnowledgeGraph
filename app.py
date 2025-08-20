import streamlit as st
import os
from dotenv import load_dotenv
from src.vector_store import create_vectorstore_from_pdf
from src.knowledge_graph import load_or_create_knowledge_graph
from src.hybrid_agent import HybridRAGAgent
import time

# Load environment variables
load_dotenv()

# Configuration
PDF_PATH = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"

st.set_page_config(
    page_title=" Medical Knowledge Assistant",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def initialize_system_cached():
    """Initialize the complete system with caching"""
    try:
        print("🚀 Initializing system...")
        vs = create_vectorstore_from_pdf(PDF_PATH)
        print(f"✅ Vector store loaded: {vs.index.ntotal} chunks")
        kg = load_or_create_knowledge_graph(PDF_PATH)
        stats = kg.get_graph_stats()
        print(f"✅ Knowledge graph loaded: {stats['nodes']} nodes, {stats['edges']} edges")
        agent = HybridRAGAgent(vs, kg)
        print("✅ System initialization complete!")
        return agent
    except Exception as e:
        print(f"❌ System initialization error: {e}")
        st.error(f"System initialization failed: {str(e)}")
        return None

def display_source_details(vector_results, graph_results, query, route, message_idx=0):
    """Display detailed source breakdown in expandable sections"""
    unique_id = int(time.time() * 1000000) % 1000000  # Unique timestamp-based ID
    
    with st.expander("🔍 **View Detailed Sources & Search Breakdown**", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📚 Vector Database Results")
            if vector_results:
                st.success(f"✅ Found {len(vector_results)} relevant chunks")
                for i, result in enumerate(vector_results[:3], 1):
                    with st.container():
                        st.write(f"**Chunk {i}:**")
                        preview = result[:300].replace('\n', ' ').strip()
                        st.text_area(
                            f"Content Preview {i}",
                            preview + "...",
                            height=100,
                            key=f"vector_{i}_{message_idx}_{unique_id}"
                        )
                if len(vector_results) > 3:
                    st.info(f"📄 +{len(vector_results) - 3} more chunks available")
            else:
                st.warning("❌ No relevant text chunks found")
        
        with col2:
            st.subheader("🕸️ Knowledge Graph Relations")
            if graph_results:
                st.success(f"✅ Found {len(graph_results)} entity relations")
                for i, result in enumerate(graph_results, 1):
                    st.code(f"{result}", language="text")
            else:
                st.warning("❌ No entity relationships found")

def main():
    st.title("🏥 Medical Knowledge Assistant")
    
    # Initialize session state
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = []
    
    if "current_session" not in st.session_state:
        st.session_state.current_session = {"id": 0, "messages": [], "title": "New Chat"}
    
    if "session_counter" not in st.session_state:
        st.session_state.session_counter = 0

    # Check API key
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        st.warning("⚠️ GROQ_API_KEY not found in .env file.")

    # Initialize system
    agent = initialize_system_cached()
    if agent is None:
        st.error("❌ System could not be initialized.")
        st.stop()

    # Sidebar with chat history
    with st.sidebar:
        st.header("💬 Chat History")
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            # Save current session if it has messages
            if st.session_state.current_session["messages"]:
                # Check if already exists
                exists = False
                for i, session in enumerate(st.session_state.chat_sessions):
                    if session["id"] == st.session_state.current_session["id"]:
                        st.session_state.chat_sessions[i] = st.session_state.current_session.copy()
                        exists = True
                        break
                if not exists:
                    st.session_state.chat_sessions.append(st.session_state.current_session.copy())
            
            # Start new session
            st.session_state.session_counter += 1
            st.session_state.current_session = {
                "id": st.session_state.session_counter, 
                "messages": [], 
                "title": "New Chat"
            }
            st.rerun()
        
        st.divider()
        
        # Display previous chat sessions
        for i, session in enumerate(reversed(st.session_state.chat_sessions[-10:])):  # Show last 10
            session_title = session.get("title", f"Chat {session['id']}")[:25]
            if st.button(f"💭 {session_title}", 
                        key=f"load_session_{session['id']}_{i}", 
                        use_container_width=True):
                
                # Save current session first
                if st.session_state.current_session["messages"]:
                    current_exists = False
                    for j, existing in enumerate(st.session_state.chat_sessions):
                        if existing["id"] == st.session_state.current_session["id"]:
                            st.session_state.chat_sessions[j] = st.session_state.current_session.copy()
                            current_exists = True
                            break
                    if not current_exists:
                        st.session_state.chat_sessions.append(st.session_state.current_session.copy())
                
                # Load selected session
                st.session_state.current_session = session.copy()
                st.rerun()
        
        st.divider()
        
        # Clear all chats
        if st.button("🗑️ Clear All Chats", use_container_width=True, key="clear_all_btn"):
            st.session_state.chat_sessions = []
            st.session_state.current_session = {"id": 0, "messages": [], "title": "New Chat"}
            st.session_state.session_counter = 0
            st.rerun()

    # Initialize current session messages
    if not st.session_state.current_session["messages"]:
        st.session_state.current_session["messages"] = [{
            "role": "assistant",
            "content": "👋 Hello! I'm your medical knowledge assistant. I can help you find information about symptoms, diseases, treatments, and medical conditions."
        }]

    # Display current chat messages
    for idx, message in enumerate(st.session_state.current_session["messages"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
        # Show detailed sources if available
        if message["role"] == "assistant" and "source_details" in message:
            display_source_details(
                message["source_details"]["vector_results"],
                message["source_details"]["graph_results"],
                message["source_details"]["query"],
                message["source_details"]["route"],
                idx
            )

    # Chat input
    if prompt := st.chat_input("Ask your medical question..."):
        # Add user message
        st.session_state.current_session["messages"].append({"role": "user", "content": prompt})
        
        # Update session title with first user message
        if st.session_state.current_session["title"] == "New Chat":
            st.session_state.current_session["title"] = prompt[:30] + "..." if len(prompt) > 30 else prompt
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching medical knowledge..."):
                try:
                    response, source_details = agent.process_query_with_details(prompt)
                    
                    # Display comprehensive response
                    st.markdown(response)
                    
                    # Display detailed source breakdown
                    display_source_details(
                        source_details["vector_results"],
                        source_details["graph_results"],
                        source_details["query"],
                        source_details["route"],
                        len(st.session_state.current_session["messages"])
                    )
                    
                    # Store in session with source details
                    st.session_state.current_session["messages"].append({
                        "role": "assistant",
                        "content": response,
                        "source_details": source_details
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}\n\nPlease try again or rephrase your question."
                    st.error(error_msg)
                    st.session_state.current_session["messages"].append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
