import networkx as nx
import re
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def extract_entities(self, text):
        patterns = [
            r'\b(?:aspirin|paracetamol|ibuprofen|antibiotics|insulin|medication|drug|medicine)\b',
            r'\b(?:fever|headache|pain|diabetes|cancer|infection|disease|treatment|therapy)\b'
        ]
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend([match.lower() for match in matches])
        return list(set(entities))

    def extract_relations(self, text):
        relations = []
        relation_patterns = {
            'treats': [r'(\w+)\s+treats?\s+(\w+)', r'(\w+)\s+for\s+(\w+)'],
            'causes': [r'(\w+)\s+causes?\s+(\w+)'],
            'prevents': [r'(\w+)\s+prevents?\s+(\w+)']
        }
        
        for relation_type, patterns in relation_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        subject, obj = match
                        relations.append((subject.lower(), relation_type, obj.lower()))
        return relations

    def build_graph_from_pdf_cached(self, pdf_path, cache_file="./kg_cache.pkl"):
        """Build or load knowledge graph from cache"""
        
        # Check if KG cache exists
        if os.path.exists(cache_file):
            print(f"✅ Loading knowledge graph from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            print(f"🕸️ Building knowledge graph from PDF: {pdf_path}")
            # Load PDF and build graph
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            for chunk in chunks:
                text = chunk.page_content
                
                # Extract entities and relations
                entities = self.extract_entities(text)
                for entity in entities:
                    self.graph.add_node(entity)
                
                relations = self.extract_relations(text)
                for subject, relation, obj in relations:
                    self.graph.add_edge(subject, obj, relation=relation)
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph, f)
            print(f"💾 Knowledge graph saved to cache: {cache_file}")

    def get_graph_context(self, query):
        query_entities = self.extract_entities(query)
        context = []
        
        for entity in query_entities:
            if entity in self.graph:
                neighbors = list(self.graph.neighbors(entity))
                if neighbors:
                    context.append(f"Related to {entity}: {', '.join(neighbors[:5])}")
        
        return "\n".join(context) if context else "No graph relationships found."
