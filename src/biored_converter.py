import json
import os
from collections import defaultdict

def convert_biored_to_medical_relations():
    """Fixed BioRED dataset to medical relations converter"""
    medical_relations = defaultdict(lambda: {
        'symptoms': [],
        'treatments': [],
        'causes': [],
        'related_to': []
    })

    # Check BioRED files
    biored_files = []
    possible_files = [
        'BIORED/Train.BioC.JSON',
        'BIORED/Dev.BioC.JSON', 
        'BIORED/Test.BioC.JSON'
    ]

    for file_path in possible_files:
        if os.path.exists(file_path):
            biored_files.append(file_path)
            print(f"✅ Found: {file_path}")

    if not biored_files:
        print("❌ No BioRED files found!")
        return {}

    total_relations = 0
    total_documents = 0

    for file_path in biored_files:
        try:
            print(f"📊 Processing {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # BioC JSON structure: data should have 'documents' key
            documents = data.get('documents', [])
            if not documents:
                print(f"⚠️ No documents found in {file_path}")
                continue

            total_documents += len(documents)
            print(f"📄 Processing {len(documents)} documents...")

            for doc in documents:
                # Build entity ID to name mapping
                entity_map = {}
                
                # Check passages for annotations
                passages = doc.get('passages', [])
                for passage in passages:
                    annotations = passage.get('annotations', [])
                    for annotation in annotations:
                        entity_id = annotation.get('id')
                        entity_text = annotation.get('text', '').lower().strip()
                        if entity_id and entity_text and len(entity_text) > 2:
                            entity_map[entity_id] = entity_text

                # Process relations
                relations = doc.get('relations', [])
                print(f"  Document {doc.get('id', 'unknown')}: Found {len(relations)} relations")
                
                for relation in relations:
                    try:
                        # Get relation info
                        infons = relation.get('infons', {})
                        entity1_id = infons.get('entity1', '')
                        entity2_id = infons.get('entity2', '')
                        rel_type = infons.get('type', 'related_to').lower()

                        # Get entity names from mapping
                        entity1_name = entity_map.get(entity1_id, '').strip()
                        entity2_name = entity_map.get(entity2_id, '').strip()

                        if not entity1_name or not entity2_name or len(entity1_name) < 2 or len(entity2_name) < 2:
                            continue

                        # Map relation types to categories
                        if any(keyword in rel_type for keyword in ['treat', 'therapy', 'drug']):
                            medical_relations[entity1_name]['treatments'].append(entity2_name)
                        elif any(keyword in rel_type for keyword in ['cause', 'induce']):
                            medical_relations[entity2_name]['causes'].append(entity1_name)  
                        elif any(keyword in rel_type for keyword in ['symptom', 'sign']):
                            medical_relations[entity1_name]['symptoms'].append(entity2_name)
                        else:
                            medical_relations[entity1_name]['related_to'].append(entity2_name)

                        total_relations += 1

                    except Exception as e:
                        continue

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue

    print(f"📈 Total documents processed: {total_documents}")
    print(f"📈 Total relations processed: {total_relations}")

    # Clean and deduplicate
    final_relations = {}
    for entity, relations in medical_relations.items():
        if any(relations.values()) and len(entity) > 2:
            cleaned_relations = {}
            for rel_type, targets in relations.items():
                if targets:
                    cleaned_targets = list(set([t for t in targets if t and len(t) > 2]))
                    if cleaned_targets:
                        cleaned_relations[rel_type] = cleaned_targets
            if cleaned_relations:
                final_relations[entity] = cleaned_relations

    # Save to JSON
    output_file = 'medical_relations.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_relations, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted {len(final_relations)} entities with relations")
    print(f"💾 Saved to {output_file}")
    
    return final_relations

if __name__ == "__main__":
    relations = convert_biored_to_medical_relations()
    if relations:
        print("\n🎯 Sample extracted relations:")
        for entity, rels in list(relations.items())[:3]:
            print(f"  {entity}: {rels}")
