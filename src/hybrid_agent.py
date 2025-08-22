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
You are a specialized medical knowledge assistant. Your primary role is to help with medical, health, and healthcare-related questions only.

INSTRUCTIONS:
1. First, determine if the question is medical/health-related:
   - Medical topics: symptoms, diseases, treatments, medications, health conditions, anatomy, medical procedures, healthcare, wellness
   - Non-medical topics: finance, technology, entertainment, sports, politics, general knowledge, etc.

2. If question is NOT medical/health-related:
   - Politely decline and redirect: "I'm sorry, but I specialize only in medical and healthcare topics. For questions about [topic], I'd recommend consulting appropriate experts in that field. Is there anything health-related I can help you with instead?"

3. If question IS medical/health-related:
   - Keep response concise and under 150 words
   - Use bullet points for key information
   - Structure: Brief definition + 3-4 key points maximum
   - Only add disclaimer for medical advice questions, not general information

4. Response Format (KEEP SHORT):
   - 1-2 sentence definition
   - Maximum 3-4 bullet points
   - No lengthy explanations

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
