import os
import pickle
import networkx as nx
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from collections import defaultdict

class KnowledgeGraph:
    def __init__(self):
        # Initialize Medical NER pipeline (OpenMed model - best free option)
        print("🔬 Loading Medical NER model...")
        try:
            self.ner_pipeline = pipeline(
                "token-classification",
                model="blaze999/Medical-NER",
                aggregation_strategy="simple",
                device=-1  # CPU usage
            )
            print("✅ Medical NER model loaded successfully")
        except Exception as e:
            print(f"⚠️ NER model failed, using fallback: {e}")
            self.ner_pipeline = None
        
        self.graph = nx.MultiDiGraph()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
    
    def extract_medical_entities(self, text):
        """Extract medical entities using advanced NER"""
        if self.ner_pipeline:
            try:
                # Use transformers NER
                entities = self.ner_pipeline(text)
                
                # Group entities by type
                entity_dict = defaultdict(list)
                for entity in entities:
                    entity_type = entity['entity_group'].lower()
                    entity_text = entity['word'].strip()
                    
                    # Map to our categories
                    if 'disease' in entity_type or 'disorder' in entity_type:
                        entity_dict['diseases'].append(entity_text)
                    elif 'drug' in entity_type or 'medication' in entity_type:
                        entity_dict['treatments'].append(entity_text)
                    elif 'symptom' in entity_type:
                        entity_dict['symptoms'].append(entity_text)
                    elif 'anatomy' in entity_type or 'body' in entity_type:
                        entity_dict['anatomy'].append(entity_text)
                
                return dict(entity_dict)
            
            except Exception as e:
                print(f"⚠️ NER extraction failed: {e}")
        
        # Fallback to regex-based extraction
        return self._regex_entity_extraction(text)
    
    def _regex_entity_extraction(self, text):
        """Fallback regex-based entity extraction"""
        entities = {
            'diseases': [],
            'symptoms': [],
            'treatments': [],
            'anatomy': []
        }
        
        # Common medical patterns
        disease_patterns = [
            r'\b(diabetes|cancer|hypertension|asthma|arthritis|pneumonia)\b',
            r'\b(\w+itis|\w+osis|\w+emia)\b',  # inflammation, condition, blood condition
        ]
        
        treatment_patterns = [
            r'\b(insulin|aspirin|antibiotics|chemotherapy|surgery)\b',
            r'\b(\w+mycin|\w+cillin)\b',  # common antibiotic suffixes
        ]
        
        symptom_patterns = [
            r'\b(pain|fever|nausea|fatigue|headache|cough)\b',
            r'\b(high blood pressure|chest pain|shortness of breath)\b'
        ]
        
        for pattern in disease_patterns:
            entities['diseases'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        for pattern in treatment_patterns:
            entities['treatments'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        for pattern in symptom_patterns:
            entities['symptoms'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Clean and deduplicate
        for key in entities:
            entities[key] = list(set([e.lower().strip() for e in entities[key] if len(e.strip()) > 2]))
        
        return entities
    
    def extract_relationships(self, text, entities):
        """Extract relationships between entities"""
        relationships = []
        
        # Relationship patterns
        patterns = [
            (r'(\w+)\s+(causes?|leads? to|results? in)\s+(\w+)', 'causes'),
            (r'(\w+)\s+(is treated with|treated by|therapy|treatment)\s+(\w+)', 'treated_with'),
            (r'(\w+)\s+(symptoms?|signs?)\s+(include|are)\s+(\w+)', 'has_symptom'),
            (r'(\w+)\s+(affects?|impacts?)\s+(\w+)', 'affects'),
            (r'(\w+)\s+(side effects?|adverse effects?)\s+(\w+)', 'side_effect'),
        ]
        
        text_lower = text.lower()
        
        for pattern, relation_type in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if len(match.groups()) >= 2:
                    entity1 = match.group(1).strip()
                    entity2 = match.group(-1).strip() if len(match.groups()) > 2 else match.group(2).strip()
                    
                    # Validate entities exist in our extracted entities
                    all_entities = []
                    for ent_list in entities.values():
                        all_entities.extend(ent_list)
                    
                    if entity1 in all_entities or entity2 in all_entities:
                        relationships.append((entity1, relation_type, entity2))
        
        return relationships
    
    def build_graph_from_pdf(self, pdf_path, persist_file="knowledge_graph.pkl"):
        """Build knowledge graph from PDF using advanced NER"""
        if os.path.exists(persist_file):
            print(f"✅ Loading existing knowledge graph from {persist_file}")
            with open(persist_file, 'rb') as f:
                self.graph = pickle.load(f)
            return self.graph
        
        print(f"📄 Building knowledge graph from PDF: {pdf_path}")
        
        # Load and process PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"📋 Processing {len(chunks)} text chunks...")
        
        all_entities = defaultdict(set)
        all_relationships = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks[:50]):  # Limit for performance
            if i % 10 == 0:
                print(f"📊 Processed {i}/{min(50, len(chunks))} chunks...")
            
            text = chunk.page_content
            
            # Extract entities
            entities = self.extract_medical_entities(text)
            for entity_type, entity_list in entities.items():
                all_entities[entity_type].update(entity_list)
            
            # Extract relationships
            relationships = self.extract_relationships(text, entities)
            all_relationships.extend(relationships)
        
        # Build NetworkX graph
        print("🔗 Building graph structure...")
        
        # Add nodes
        for entity_type, entity_set in all_entities.items():
            for entity in entity_set:
                self.graph.add_node(entity, type=entity_type)
        
        # Add edges
        for entity1, relation, entity2 in all_relationships:
            if self.graph.has_node(entity1) and self.graph.has_node(entity2):
                self.graph.add_edge(entity1, entity2, relation=relation)
        
        # Save graph
        with open(persist_file, 'wb') as f:
            pickle.dump(self.graph, f)
        
        nodes = self.graph.number_of_nodes()
        edges = self.graph.number_of_edges()
        print(f"✅ Knowledge graph created: {nodes} nodes, {edges} edges")
        print(f"💾 Graph saved to {persist_file}")
        
        return self.graph
    
    def query_graph(self, query, max_results=5):
        """Query the knowledge graph"""
        if self.graph.number_of_nodes() == 0:
            return []
        
        query_lower = query.lower()
        results = []
        
        # Find relevant nodes
        relevant_nodes = []
        for node in self.graph.nodes():
            if any(word in node.lower() for word in query_lower.split()):
                relevant_nodes.append(node)
        
        # Get related information
        for node in relevant_nodes[:5]:  # Limit results
            # Direct neighbors
            neighbors = list(self.graph.neighbors(node))
            
            # Get edge information
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(node, neighbor)
                if edge_data:
                    for edge_info in edge_data.values():
                        relation = edge_info.get('relation', 'related_to')
                        result_text = f"{node} {relation} {neighbor}"
                        results.append(result_text)
        
        # Also search in reverse direction
        for node in relevant_nodes[:5]:
            predecessors = list(self.graph.predecessors(node))
            for pred in predecessors:
                edge_data = self.graph.get_edge_data(pred, node)
                if edge_data:
                    for edge_info in edge_data.values():
                        relation = edge_info.get('relation', 'related_to')
                        result_text = f"{pred} {relation} {node}"
                        results.append(result_text)
        
        # Remove duplicates and limit results
        unique_results = list(set(results))
        return unique_results[:max_results]
    
    def get_graph_stats(self):
        """Get basic statistics about the graph"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'node_types': len(set(nx.get_node_attributes(self.graph, 'type').values()))
        }
