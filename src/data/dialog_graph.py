"""
Dialog Context Graph Construction Module

This module builds temporal graphs for dialog context:
- Turn-level connections for conversation flow
- Entity linking across turns
- Context propagation for better understanding
"""

import json
import re
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DialogTurn:
    """Represents a single dialog turn"""
    turn_id: int
    speaker: str
    utterance: str
    belief_state: Dict
    entities: List[str]
    intent: Optional[str] = None


@dataclass 
class DialogContext:
    """Represents dialog context with entity tracking"""
    dialogue_id: str
    turns: List[DialogTurn]
    entities: Dict[str, List[int]]  # entity -> turn_ids where it appears
    coreferences: List[Tuple[int, int, str]]  # (turn1, turn2, entity)


class DialogContextGraph:
    """Builds temporal context graphs for dialogs"""
    
    def __init__(self, ontology_path: str):
        """
        Initialize dialog context graph builder
        
        Args:
            ontology_path: Path to ontology.json for entity recognition
        """
        self.ontology_path = ontology_path
        self.ontology = self._load_ontology()
        
        # Build entity vocabularies for recognition
        self.all_entities = self._extract_all_entities()
        self.domain_keywords = self._extract_domain_keywords()
        
        # Coreference patterns
        self.pronoun_patterns = [
            r'\bit\b', r'\bthat\b', r'\bthis\b', r'\bthere\b',
            r'\bthem\b', r'\bthey\b', r'\bone\b'
        ]
        
        self.location_synonyms = {
            'centre': ['center', 'central', 'downtown'],
            'cheap': ['inexpensive', 'affordable', 'budget'],
            'expensive': ['costly', 'pricey', 'high-end']
        }
    
    def _load_ontology(self) -> Dict:
        """Load ontology from JSON file"""
        with open(self.ontology_path, 'r') as f:
            return json.load(f)
    
    def _extract_all_entities(self) -> Set[str]:
        """Extract all entities from ontology values"""
        entities = set()
        for slot, values in self.ontology.items():
            for value in values:
                if value not in ['none', 'dontcare']:
                    entities.add(value.lower())
        return entities
    
    def _extract_domain_keywords(self) -> Dict[str, Set[str]]:
        """Extract domain-specific keywords"""
        keywords = {
            'hotel': {'hotel', 'accommodation', 'stay', 'room', 'lodge', 'inn'},
            'restaurant': {'restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal'},
            'attraction': {'attraction', 'visit', 'see', 'tour', 'museum', 'park'},
            'train': {'train', 'railway', 'station', 'ticket', 'journey'},
            'taxi': {'taxi', 'cab', 'ride', 'transport', 'car'}
        }
        return keywords
    
    def extract_entities(self, utterance: str) -> List[str]:
        """
        Extract entities from utterance using ontology and patterns
        
        Args:
            utterance: Input utterance text
            
        Returns:
            List of extracted entities
        """
        utterance_lower = utterance.lower()
        extracted = []
        
        # 1. Direct ontology matches
        for entity in self.all_entities:
            if entity in utterance_lower:
                extracted.append(entity)
        
        # 2. Domain keyword matches
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in utterance_lower:
                    extracted.append(f"domain_{domain}")
        
        # 3. Numeric entities (times, numbers)
        time_pattern = r'\b(\d{1,2}):(\d{2})\b'
        times = re.findall(time_pattern, utterance)
        for hour, minute in times:
            extracted.append(f"time_{hour}:{minute}")
        
        number_pattern = r'\b(\d+)\s*(people|person|night|nights|day|days)?\b'
        numbers = re.findall(number_pattern, utterance_lower)
        for num, unit in numbers:
            if unit:
                extracted.append(f"number_{num}_{unit}")
            else:
                extracted.append(f"number_{num}")
        
        return list(set(extracted))  # Remove duplicates
    
    def detect_coreferences(self, turns: List[DialogTurn]) -> List[Tuple[int, int, str]]:
        """
        Detect coreference relationships between turns
        
        Args:
            turns: List of dialog turns
            
        Returns:
            List of (turn1_id, turn2_id, entity) coreference links
        """
        coreferences = []
        
        # Track entities across turns
        entity_history = {}  # entity -> most recent turn_id
        
        for turn in turns:
            if turn.speaker == 'user':  # Focus on user utterances
                utterance_lower = turn.utterance.lower()
                
                # Check for pronouns/references
                has_pronoun = any(re.search(pattern, utterance_lower) 
                                for pattern in self.pronoun_patterns)
                
                if has_pronoun and entity_history:
                    # Link to most recent entity mentions
                    for entity, prev_turn_id in entity_history.items():
                        if turn.turn_id - prev_turn_id <= 3:  # Within 3 turns
                            coreferences.append((prev_turn_id, turn.turn_id, entity))
                
                # Update entity history
                for entity in turn.entities:
                    entity_history[entity] = turn.turn_id
        
        return coreferences
    
    def build_dialog_context(self, dialog_data: Dict) -> DialogContext:
        """
        Build dialog context from MultiWOZ dialog data
        
        Args:
            dialog_data: Single dialog from MultiWOZ dataset
            
        Returns:
            DialogContext object with processed information
        """
        dialogue_id = dialog_data['dialogue_id']
        turns = []
        
        # Process each turn
        for turn_data in dialog_data['turns']:
            # Extract entities from utterance
            entities = self.extract_entities(turn_data['utterance'])
            
            # Determine intent (simplified)
            intent = self._classify_intent(turn_data['utterance'])
            
            turn = DialogTurn(
                turn_id=turn_data['turn_id'],
                speaker=turn_data['speaker'],
                utterance=turn_data['utterance'],
                belief_state=turn_data.get('belief_state', {}),
                entities=entities,
                intent=intent
            )
            turns.append(turn)
        
        # Detect coreferences
        coreferences = self.detect_coreferences(turns)
        
        # Build entity index
        entity_index = defaultdict(list)
        for turn in turns:
            for entity in turn.entities:
                entity_index[entity].append(turn.turn_id)
        
        return DialogContext(
            dialogue_id=dialogue_id,
            turns=turns,
            entities=dict(entity_index),
            coreferences=coreferences
        )
    
    def _classify_intent(self, utterance: str) -> str:
        """
        Classify user intent from utterance (simplified)
        
        Args:
            utterance: User utterance
            
        Returns:
            Predicted intent
        """
        utterance_lower = utterance.lower()
        
        # Intent patterns
        if any(word in utterance_lower for word in ['book', 'reserve', 'make a reservation']):
            return 'book'
        elif any(word in utterance_lower for word in ['find', 'look for', 'search', 'need']):
            return 'search'
        elif any(word in utterance_lower for word in ['thank', 'bye', 'goodbye', 'thats all']):
            return 'goodbye'
        elif any(word in utterance_lower for word in ['yes', 'yeah', 'ok', 'sure']):
            return 'affirm'
        elif any(word in utterance_lower for word in ['no', 'not', 'dont']):
            return 'deny'
        else:
            return 'inform'
    
    def build_turn_graph(self, context: DialogContext) -> Dict:
        """
        Build turn-level graph representation
        
        Args:
            context: Dialog context object
            
        Returns:
            Graph representation as adjacency information
        """
        num_turns = len(context.turns)
        
        # Node features: [turn_id, speaker_type, num_entities, intent_type]
        node_features = []
        turn_types = []  # 0=user, 1=system
        
        for turn in context.turns:
            speaker_type = 1 if turn.speaker == 'system' else 0
            intent_onehot = self._intent_to_onehot(turn.intent)
            
            features = [
                turn.turn_id,
                speaker_type, 
                len(turn.entities),
                len(turn.belief_state)
            ] + intent_onehot
            
            node_features.append(features)
            turn_types.append(speaker_type)
        
        # Edge construction
        edges = []
        edge_types = []
        
        # 1. Sequential edges (turn to turn)
        for i in range(num_turns - 1):
            edges.append([i, i + 1])
            edge_types.append('sequential')
        
        # 2. Coreference edges
        turn_id_to_idx = {turn.turn_id: i for i, turn in enumerate(context.turns)}
        
        for turn1_id, turn2_id, entity in context.coreferences:
            if turn1_id in turn_id_to_idx and turn2_id in turn_id_to_idx:
                idx1 = turn_id_to_idx[turn1_id]
                idx2 = turn_id_to_idx[turn2_id]
                edges.append([idx1, idx2])
                edge_types.append(f'coreference_{entity}')
        
        # 3. Entity sharing edges (turns that mention same entities)
        for i, turn1 in enumerate(context.turns):
            for j, turn2 in enumerate(context.turns[i+1:], i+1):
                shared_entities = set(turn1.entities).intersection(set(turn2.entities))
                if shared_entities and abs(i - j) <= 5:  # Within 5 turns
                    edges.append([i, j])
                    edge_types.append(f'entity_shared')
        
        return {
            'node_features': node_features,
            'turn_types': turn_types,
            'edges': edges,
            'edge_types': edge_types,
            'num_nodes': num_turns,
            'dialogue_id': context.dialogue_id
        }
    
    def _intent_to_onehot(self, intent: str) -> List[int]:
        """Convert intent to one-hot encoding"""
        intents = ['search', 'book', 'inform', 'affirm', 'deny', 'goodbye']
        onehot = [0] * len(intents)
        
        if intent in intents:
            onehot[intents.index(intent)] = 1
        
        return onehot
    
    def build_entity_graph(self, context: DialogContext) -> Dict:
        """
        Build entity-centric graph representation
        
        Args:
            context: Dialog context object
            
        Returns:
            Entity graph with entity nodes and turn connections
        """
        # Get unique entities
        all_entities = set()
        for turn in context.turns:
            all_entities.update(turn.entities)
        
        entity_list = sorted(list(all_entities))
        entity_to_idx = {entity: i for i, entity in enumerate(entity_list)}
        
        # Entity node features (simple embedding indices)
        entity_features = list(range(len(entity_list)))
        
        # Entity-Turn edges
        entity_turn_edges = []
        turn_entity_edges = []
        
        for turn_idx, turn in enumerate(context.turns):
            for entity in turn.entities:
                entity_idx = entity_to_idx[entity]
                entity_turn_edges.append([entity_idx, turn_idx])
                turn_entity_edges.append([turn_idx, entity_idx])
        
        # Entity-Entity edges (co-occurrence)
        entity_edges = []
        
        for i, entity1 in enumerate(entity_list):
            for j, entity2 in enumerate(entity_list[i+1:], i+1):
                # Check if entities co-occur in same turns
                turns1 = set(context.entities.get(entity1, []))
                turns2 = set(context.entities.get(entity2, []))
                
                if turns1.intersection(turns2):
                    entity_edges.append([i, j])
        
        return {
            'entity_features': entity_features,
            'entity_names': entity_list,
            'entity_turn_edges': entity_turn_edges,
            'turn_entity_edges': turn_entity_edges,
            'entity_edges': entity_edges,
            'num_entities': len(entity_list)
        }
    
    def process_dialog_batch(self, dialog_batch: List[Dict]) -> List[Dict]:
        """
        Process a batch of dialogs into graph representations
        
        Args:
            dialog_batch: List of dialog data from MultiWOZ
            
        Returns:
            List of processed graph representations
        """
        processed = []
        
        for dialog_data in dialog_batch:
            # Build context
            context = self.build_dialog_context(dialog_data)
            
            # Build graphs
            turn_graph = self.build_turn_graph(context)
            entity_graph = self.build_entity_graph(context)
            
            # Combine information
            graph_data = {
                'dialogue_id': context.dialogue_id,
                'turn_graph': turn_graph,
                'entity_graph': entity_graph,
                'context': context,
                'num_turns': len(context.turns)
            }
            
            processed.append(graph_data)
        
        return processed


def build_context_graphs_from_data(data_path: str, ontology_path: str, 
                                   output_path: str = None) -> List[Dict]:
    """
    Build context graphs from MultiWOZ data file
    
    Args:
        data_path: Path to MultiWOZ JSON file (train.json, val.json, etc.)
        ontology_path: Path to ontology.json
        output_path: Optional path to save processed graphs
        
    Returns:
        List of processed dialog graphs
    """
    # Load data
    with open(data_path, 'r') as f:
        dialog_data = json.load(f)
    
    # Initialize graph builder
    builder = DialogContextGraph(ontology_path)
    
    # Process dialogs
    print(f"Processing {len(dialog_data)} dialogs...")
    processed_graphs = builder.process_dialog_batch(dialog_data)
    
    # Save if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(processed_graphs, f, indent=2)
        print(f"Saved processed graphs to {output_path}")
    
    return processed_graphs


if __name__ == "__main__":
    # Example usage
    ontology_path = "../../data/ontology.json"
    train_path = "../../data/train.json"
    
    # Build context graphs
    builder = DialogContextGraph(ontology_path)
    
    # Load sample dialog
    with open(train_path, 'r') as f:
        sample_dialogs = json.load(f)[:5]  # First 5 dialogs
    
    # Process sample
    processed = builder.process_dialog_batch(sample_dialogs)
    
    print(f"Processed {len(processed)} dialogs")
    print(f"Sample dialog statistics:")
    
    for i, graph_data in enumerate(processed[:2]):
        print(f"\nDialog {i+1}:")
        print(f"  Turns: {graph_data['num_turns']}")
        print(f"  Entities: {len(graph_data['entity_graph']['entity_names'])}")
        print(f"  Turn edges: {len(graph_data['turn_graph']['edges'])}")
        print(f"  Entity edges: {len(graph_data['entity_graph']['entity_edges'])}")