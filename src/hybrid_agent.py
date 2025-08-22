# src/hybrid_agent.py

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os

class HybridRAGAgent:
    def __init__(self, vectorstore, knowledge_graph):
        self.vectorstore = vectorstore
        self.knowledge_graph = knowledge_graph
        self.llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.prompt_template = PromptTemplate(
    template="""
You are a specialized medical assistant. Your main role is to answer health and medical questions.

Instructions:
• If the user says anything showing gratitude ("thanks", "thank you", "shukriya", "appreciate", "great job", "dhanyavad"), reply with a short, friendly message (for example: "You're welcome!", "Glad I could help!").
• If greeted ("hi", "hello", "hey", etc.), respond politely (example: "Hello! How can I assist you today?")
• If the question is medical/health-related (symptoms, diseases, treatments, conditions, wellness):
    – Give a brief definition (1-2 sentences)
    – List 3-4 key points as bullet points (use double newlines between bullets, e.g. "• Point 1\n\n• Point 2\n\n• Point 3")
    – Add a disclaimer only for medical advice or diagnosis, not general info
• For non-medical topics (finance, sports, tech), politely decline and suggest the user asks health-related questions

Context from PDF:
{context}

Knowledge Graph Relations:
{relations}

User Question: {question}

Answer:
""",
    input_variables=["context", "relations", "question"]
)


    def process_query_with_details(self, query):
        try:
            # Vector search
            vector_docs = self.vectorstore.similarity_search(query, k=5)
            vector_results = [doc.page_content for doc in vector_docs]

            # Graph search
            graph_results = self.knowledge_graph.query_graph(query, max_results=5)

            # Summarise context
            context_text = "\n---\n".join(vector_results[:2]) if vector_results else ""
            relations_text = "\n".join(graph_results) if graph_results else ""

            # LLM answer using context and graph
            final_prompt = self.prompt_template.format(
                context=context_text,
                relations=relations_text,
                question=query
            )
            
            # GROQ Completion
            response = self.llm.invoke(final_prompt)
            answer = response.content

            source_details = {
                "vector_results": vector_results,
                "graph_results": graph_results,
                "query": query,
                "route": self._get_route(vector_results, graph_results)
            }
            return answer, source_details
        except Exception as e:
            error_response = f"Error processing query: {str(e)}"
            empty_details = {
                "vector_results": [],
                "graph_results": [],
                "query": query,
                "route": "error"
            }
            return error_response, empty_details

    def _get_route(self, vector_results, graph_results):
        if graph_results and vector_results:
            return "both"
        elif vector_results:
            return "vector_only"
        elif graph_results:
            return "graph_only"
        return "none"
