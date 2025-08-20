# src/hybrid_agent.py
class HybridRAGAgent:
    def __init__(self, vectorstore, knowledge_graph):
        self.vectorstore = vectorstore
        self.knowledge_graph = knowledge_graph

    def process_query_with_details(self, query):
        """Process query and return comprehensive response with source details"""
        try:
            # Vector search
            vector_docs = self.vectorstore.similarity_search(query, k=5)
            vector_results = [doc.page_content for doc in vector_docs]
            
            # Graph search  
            graph_results = self.knowledge_graph.query_graph(query, max_results=5)
            
            # Generate comprehensive response
            response = self._generate_comprehensive_response(query, vector_results, graph_results)
            
            # Determine route
            if graph_results and vector_results:
                route = "both"
            elif vector_results:
                route = "vector_only"
            elif graph_results:
                route = "graph_only"
            else:
                route = "none"
            
            source_details = {
                "vector_results": vector_results,
                "graph_results": graph_results, 
                "query": query,
                "route": route
            }
            
            return response, source_details
            
        except Exception as e:
            error_response = f"Error processing query: {str(e)}"
            empty_details = {
                "vector_results": [],
                "graph_results": [], 
                "query": query,
                "route": "error"
            }
            return error_response, empty_details
    
    def _generate_comprehensive_response(self, query, vector_results, graph_results):
        """Generate a well-formatted comprehensive medical response"""
        
        # Process graph results for structured info
        symptoms = []
        treatments = []
        causes = []
        related = []
        
        if graph_results:
            for relation in graph_results:
                relation_lower = relation.lower()
                if 'symptoms' in relation_lower:
                    part = relation.split(' symptoms ')[-1] if ' symptoms ' in relation else relation.split()[-1]
                    symptoms.append(part.strip())
                elif 'treatments' in relation_lower or 'treatment' in relation_lower:
                    part = relation.split(' treatments ')[-1] if ' treatments ' in relation else relation.split()[-1]
                    treatments.append(part.strip())
                elif 'causes' in relation_lower:
                    part = relation.split(' causes ')[-1] if ' causes ' in relation else relation.split()[-1]
                    causes.append(part.strip())
                else:
                    related.append(relation.strip())
        
        # Combine vector results
        combined_text = " ".join(vector_results[:2]) if vector_results else ""
        
        # Generate response
        response_parts = []
        query_lower = query.lower()
        
        # Header based on query type
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            if 'fever' in query_lower:
                response_parts.append("## 🌡️ What is Fever?")
                response_parts.append("**Fever** is a temporary increase in body temperature, often due to illness. It's a natural immune response that helps fight infections.\n")
                
                if combined_text:
                    clean_text = combined_text[:350].strip()
                    response_parts.append(f"### 📚 Medical Definition\n{clean_text}...\n")
                
        elif any(word in query_lower for word in ['symptoms', 'signs']):
            condition = query_lower.replace('symptoms of', '').replace('symptoms', '').strip().title()
            response_parts.append(f"## 🩺 Symptoms of {condition}")
            
        elif any(word in query_lower for word in ['treatment', 'cure', 'medicine']):
            condition = query_lower.replace('treatment for', '').replace('treatment', '').strip().title()
            response_parts.append(f"## 💊 Treatment for {condition}")
            
        else:
            response_parts.append(f"## 📋 Medical Information: {query.title()}")
            if combined_text:
                clean_text = combined_text[:400].strip()
                response_parts.append(f"{clean_text}...\n")
        
        # Add structured information from knowledge graph
        if symptoms or treatments or causes or related:
            response_parts.append("### 🔗 Key Medical Relationships")
            
            if symptoms:
                symptoms_text = ", ".join(list(set(symptoms))[:4])
                response_parts.append(f"**🩺 Symptoms:** {symptoms_text}")
                
            if treatments:
                treatments_text = ", ".join(list(set(treatments))[:4])
                response_parts.append(f"**💊 Treatments:** {treatments_text}")
                
            if causes:
                causes_text = ", ".join(list(set(causes))[:4])  
                response_parts.append(f"**⚠️ Causes:** {causes_text}")
                
            if related:
                related_text = ", ".join(list(set(related))[:3])
                response_parts.append(f"**🔗 Related:** {related_text}")
        
        # Add disclaimer
        response_parts.append("\n---")
        response_parts.append("*📋 This information is for educational purposes only. Always consult healthcare professionals for medical advice.*")
        
        return "\n\n".join(response_parts) if response_parts else f"I found some information about '{query}', but couldn't generate a comprehensive summary. Please check the detailed sources below."
