import os
import json
import pickle
from typing import List, Dict, Any
import re
from datetime import datetime

class Utils:
    """Utility functions for the Hybrid RAG system"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            # Break if we've covered all words
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    @staticmethod
    def extract_medical_entities(text: str) -> List[str]:
        """Extract medical entities using simple patterns"""
        medical_patterns = [
            r'\b(?:aspirin|paracetamol|ibuprofen|antibiotics|insulin)\b',
            r'\b(?:fever|headache|pain|diabetes|cancer|infection)\b',
            r'\b(?:treatment|medicine|drug|medication|therapy)\b',
            r'\b(?:doctor|patient|hospital|clinic)\b'
        ]
        
        entities = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend([match.lower() for match in matches])
            
        return list(set(entities))
    
    @staticmethod
    def save_to_json(data: Dict[str, Any], filepath: str) -> bool:
        """Save data to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            return False
    
    @staticmethod
    def load_from_json(filepath: str) -> Dict[str, Any]:
        """Load data from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading from JSON: {e}")
            return {}
    
    @staticmethod
    def save_vectorstore(vectorstore, filepath: str) -> bool:
        """Save vectorstore using pickle"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(vectorstore, f)
            return True
        except Exception as e:
            print(f"Error saving vectorstore: {e}")
            return False
    
    @staticmethod
    def load_vectorstore(filepath: str):
        """Load vectorstore from pickle file"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            return None
    
    @staticmethod
    def format_response(response: str, max_length: int = 1000) -> str:
        """Format and truncate response"""
        if len(response) > max_length:
            response = response[:max_length] + "..."
        
        # Clean up formatting
        response = response.replace('\n\n\n', '\n\n')
        response = response.strip()
        
        return response
    
    @staticmethod
    def log_query(query: str, response: str, timestamp: str = None) -> None:
        """Log queries and responses for debugging"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            "timestamp": timestamp,
            "query": query,
            "response": response[:200] + "..." if len(response) > 200 else response
        }
        
        log_file = "logs/query_log.json"
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Load existing logs
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Add new log entry
        logs.append(log_entry)
        
        # Save updated logs
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def validate_groq_api_key(api_key: str) -> bool:
        """Validate Groq API key format"""
        if not api_key or len(api_key) < 10:
            return False
        return True
    
    @staticmethod
    def create_sample_medical_data() -> List[str]:
        """Create sample medical documents for testing"""
        return [
            "Aspirin is an effective medication for reducing fever and treating mild to moderate pain.",
            "Paracetamol (acetaminophen) is commonly used to treat headaches and reduce fever in both adults and children.",
            "Type 2 diabetes is a chronic condition that affects blood sugar regulation and can be managed with proper diet and exercise.",
            "Antibiotics are specifically designed to fight bacterial infections but are ineffective against viral infections.",
            "Regular physical exercise helps prevent cardiovascular disease and improves overall health outcomes.",
            "Vitamin D deficiency can lead to bone weakness and is often caused by insufficient sunlight exposure.",
            "Hypertension (high blood pressure) is a major risk factor for heart disease and stroke.",
            "Proper hydration is essential for kidney function and helps maintain optimal body temperature.",
            "Cancer screening programs help detect malignancies in their early stages when treatment is most effective.",
            "Mental health disorders require professional treatment and should not be ignored or self-treated."
        ]
