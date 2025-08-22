import os
import pickle
import networkx as nx
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import json
from collections import defaultdict
import streamlit as st

class KnowledgeGraph:
    def __init__(self):
        try:
            self.ner_pipeline = pipeline(
                "token-classification",
                "blaze999/Medical-NER",
                aggregation_strategy="simple",
                device=-1
            )
        except Exception:
            self.ner_pipeline = None

        self.graph = nx.MultiDiGraph()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    def extract_medical_entities(self, text):
        entities = {"diseases": [], "symptoms": [], "treatments": [], "anatomy": []}

        disease_patterns = [
            r'\b(diabetes|cancer|hypertension|asthma|arthritis|pneumonia|fever|malaria|tuberculosis)\b',
            r'\b(covid|corona|influenza|hepatitis|bronchitis|gastritis|dermatitis)\b',
            r'\b(\w+itis|\w+osis|\w+emia|\w+pathy)\b',
        ]

        treatment_patterns = [
            r'\b(insulin|aspirin|antibiotics|chemotherapy|surgery|ibuprofen|acetaminophen|paracetamol)\b',
            r'\b(treatment|therapy|medicine|medication|drug|remedy)\b',
            r'\b(\w+mycin|\w+cillin|\w+azole)\b',
        ]

        symptom_patterns = [
            r'\b(pain|fever|nausea|fatigue|headache|cough|chills|sweating|vomiting|dizziness)\b',
            r'\b(high blood pressure|chest pain|shortness of breath|stomach pain|back pain)\b',
        ]

        text_lower = text.lower()
        for pattern in disease_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities["diseases"].extend([m.strip() for m in matches if len(m.strip()) > 2])
        for pattern in treatment_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities["treatments"].extend([m.strip() for m in matches if len(m.strip()) > 2])
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities["symptoms"].extend([m.strip() for m in matches if len(m.strip()) > 2])
        for key in entities:
            entities[key] = list(set([e.lower().strip() for e in entities[key] if len(e.strip()) > 2]))
        return entities

    def load_external_relations(self, relations_file="medical_relations.json"):
        if not os.path.exists(relations_file):
            return
        
        with open(relations_file, "r", encoding="utf-8") as f:
            relations = json.load(f)

        for entity, entity_relations in relations.items():
            entity_clean = entity.lower().strip()
            if len(entity_clean) > 1:
                self.graph.add_node(entity_clean, type="medical_entity", source="biored")
                for rel_type, targets in entity_relations.items():
                    if isinstance(targets, list):
                        for target in targets:
                            if target and isinstance(target, str) and len(target.strip()) > 1:
                                target_clean = target.lower().strip()
                                self.graph.add_node(target_clean, type="related_entity", source="biored")
                                self.graph.add_edge(entity_clean, target_clean, relation=rel_type)
                                self.graph.add_edge(target_clean, entity_clean, relation=f"inverse_{rel_type}")

    def build_graph_from_pdf(self, pdf_path, persist_file="knowledge_graph.pkl"):
        if os.path.exists(persist_file):
            with open(persist_file, "rb") as f:
                self.graph = pickle.load(f)
            self.load_external_relations()
            with open(persist_file, "wb") as f:
                pickle.dump(self.graph, f)
            return self.graph

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)

        all_entities = defaultdict(set)
        for chunk in chunks[:50]:
            text = chunk.page_content
            entities = self.extract_medical_entities(text)
            for k, v in entities.items():
                all_entities[k].update(v)
        
        for k, node_set in all_entities.items():
            for node in node_set:
                if len(node) > 2:
                    self.graph.add_node(node, type=k, source="pdf")

        self.load_external_relations()
        
        with open(persist_file, "wb") as f:
            pickle.dump(self.graph, f)
        return self.graph

    def query_graph(self, query, max_results=5):  # ✅ Fixed indentation
        if self.graph.number_of_nodes() == 0:
            return []

        query_lower = query.lower()
        relevant_nodes = set()
        
        # Extract medical keywords from query - BETTER APPROACH
        medical_keywords = []
        for word in query_lower.split():
            word = word.strip('.,?!')  # Remove punctuation
            if len(word) > 2 and word not in ['what', 'is', 'are', 'the', 'for', 'about', 'tell', 'me']:
                medical_keywords.append(word)
        
        # Find nodes using keywords
        for node in self.graph.nodes():
            node_lower = str(node).lower()
            # Check if any keyword matches node
            if any(keyword in node_lower or node_lower in keyword for keyword in medical_keywords):
                relevant_nodes.add(node)
            # Original exact matching logic too
            if (query_lower == node_lower or 
                query_lower in node_lower or 
                node_lower in query_lower):
                relevant_nodes.add(node)

        results = []
        for node in list(relevant_nodes)[:15]:
            # Get neighbors and predecessors
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                if edge_data:
                    for edge_info in edge_data.values():
                        if edge_info:  # Check if edge_info is not None
                            relation = edge_info.get("relation", "related_to")
                            results.append(f"{node} {relation} {neighbor}")
            
            for pred in self.graph.predecessors(node):
                edge_data = self.graph.get_edge_data(pred, node)
                if edge_data:
                    for edge_info in edge_data.values():
                        if edge_info:  # Check if edge_info is not None
                            relation = edge_info.get("relation", "related_to")
                            results.append(f"{pred} {relation} {node}")

        return list(set(results))[:max_results]

    def get_graph_stats(self):
        node_types = nx.get_node_attributes(self.graph, "type")
        node_sources = nx.get_node_attributes(self.graph, "source")
        
        type_counts = defaultdict(int)
        source_counts = defaultdict(int)
        
        for node_type in node_types.values():
            type_counts[node_type] += 1
        for source in node_sources.values():
            source_counts[source] += 1

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": len(set(node_types.values())),
            "type_breakdown": dict(type_counts),
            "source_breakdown": dict(source_counts)
        }

@st.cache_resource
def load_or_create_knowledge_graph(pdf_path):
    kg = KnowledgeGraph()
    kg.build_graph_from_pdf(pdf_path)
    return kg
