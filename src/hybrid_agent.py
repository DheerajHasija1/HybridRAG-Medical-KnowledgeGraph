from langgraph.graph import StateGraph
from typing import TypedDict, List
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    query: str
    vector_results: List[str]
    graph_context: str
    final_context: str
    response: str

class HybridRAGAgent:
    def __init__(self, vector_store, knowledge_graph):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        wf = StateGraph(AgentState)
        wf.add_node("vector_search", self.vector_search_node)
        wf.add_node("graph_search", self.graph_search_node)
        wf.add_node("combine_context", self.combine_context_node)
        wf.add_node("generate_response", self.generate_response_node)
        wf.add_edge("vector_search", "graph_search")
        wf.add_edge("graph_search", "combine_context")
        wf.add_edge("combine_context", "generate_response")
        wf.set_entry_point("vector_search")
        wf.set_finish_point("generate_response")
        return wf.compile()

    def vector_search_node(self, state: AgentState) -> AgentState:
        state["vector_results"] = self.vector_store.similarity_search(state["query"])
        return state

    def graph_search_node(self, state: AgentState) -> AgentState:
        state["graph_context"] = self.kg.get_graph_context(state["query"])
        return state

    def combine_context_node(self, state: AgentState) -> AgentState:
        state["final_context"] = f"Vector Results:\n{chr(10).join(state['vector_results'])}\nKG Context:\n{state['graph_context']}"
        return state

    def generate_response_node(self, state: AgentState) -> AgentState:
        msg = [
            {"role": "user", "content": f"Answer based on this context:\n{state['final_context']}\nQuestion: {state['query']}"}]
        try:
            resp = self.llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=msg,
                max_tokens=500
            )
            state["response"] = resp.choices[0].message.content
        except Exception as e:
            state["response"] = "Error: " + str(e)
        return state

    def query(self, user_query: str):
        initial = dict(query=user_query, vector_results=[], graph_context="", final_context="", response="")
        return self.workflow.invoke(initial)
