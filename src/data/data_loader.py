"""
Data Processing Pipeline for GraphDST

This module handles:
- MultiWOZ data loading and preprocessing
- Graph construction and batching
- Text tokenization and formatting
- Label creation for multi-task learning
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import pickle


@dataclass
class DSTExample:
    """Single dialog state tracking example"""
    dialogue_id: str
    turn_id: int
    dialog_history: str
    current_utterance: str
    belief_state: Dict[str, str]
    domain_labels: List[int]  # Binary labels for each domain
    slot_labels: Dict[str, int]  # Binary labels for each slot  
    value_labels: Dict[str, Union[int, Tuple[int, int]]]  # Categorical indices or span positions
    
    # Graph information
    schema_graph: Dict
    dialog_graph: Dict
    turn_graph: Dict


class MultiWOZProcessor:
    """Processes MultiWOZ data into GraphDST format"""
    
    def __init__(self, data_dir: str, ontology_path: str, max_history_turns: int = 3):
        """
        Initialize MultiWOZ processor
        
        Args:
            data_dir: Directory containing MultiWOZ data files
            ontology_path: Path to ontology.json
            max_history_turns: Maximum number of history turns to include
        """
        self.data_dir = data_dir
        self.ontology_path = ontology_path
        self.max_history_turns = max_history_turns
        
        # Load ontology
        with open(ontology_path, 'r') as f:
            self.ontology = json.load(f)
            
        # Initialize graph builders
        self.schema_builder = None  # Will be set from schema_graph module
        self.dialog_builder = None  # Will be set from dialog_graph module
        
        # Domain and slot mappings
        self.domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi']
        self.domain2id = {domain: i for i, domain in enumerate(self.domains)}
        
        self.slots = list(self.ontology.keys())
        self.slot2id = {slot: i for i, slot in enumerate(self.slots)}
        
        # Categorize slots
        self.categorical_slots, self.span_slots = self._categorize_slots()
        
        # Build value mappings for categorical slots
        self.value_mappings = self._build_value_mappings()
        
    def _categorize_slots(self, threshold: int = 50) -> Tuple[List[str], List[str]]:
        """Categorize slots into categorical vs span based on vocabulary size"""
        categorical = []
        span = []
        
        for slot, values in self.ontology.items():
            if len(values) <= threshold:
                categorical.append(slot)
            else:
                span.append(slot)
                
        return categorical, span
    
    def _build_value_mappings(self) -> Dict[str, Dict[str, int]]:
        """Build value to index mappings for categorical slots"""
        mappings = {}
        
        for slot in self.categorical_slots:
            if slot in self.ontology:
                values = self.ontology[slot]
                mappings[slot] = {value: i for i, value in enumerate(values)}
        
        return mappings
    
    def format_dialog_history(self, turns: List[Dict], current_turn_idx: int) -> str:
        """
        Format dialog history for current turn
        
        Args:
            turns: List of dialog turns
            current_turn_idx: Index of current turn
            
        Returns:
            Formatted history string
        """
        if current_turn_idx == 0:
            return ""
        
        # Get history turns (up to max_history_turns)
        start_idx = max(0, current_turn_idx - self.max_history_turns)
        history_turns = turns[start_idx:current_turn_idx]
        
        # Format history
        history_parts = []
        for turn in history_turns:
            if turn['speaker'] == 'user':
                history_parts.append(f"[USR] {turn['utterance']}")
            else:
                # Truncate system response to avoid length issues
                response = turn.get('system_response', '')
                if len(response) > 150:
                    response = response[:147] + "..."
                history_parts.append(f"[SYS] {response}")
        
        return " ".join(history_parts)
    
    def create_domain_labels(self, belief_state: Dict[str, str]) -> List[int]:
        """
        Create binary domain labels from belief state
        
        Args:
            belief_state: Current belief state
            
        Returns:
            Binary labels for each domain
        """
        domain_labels = [0] * len(self.domains)
        
        for slot, value in belief_state.items():
            if value and value not in ['none', '']:
                domain = slot.split('-')[0]
                if domain in self.domain2id:
                    domain_labels[self.domain2id[domain]] = 1
        
        return domain_labels
    
    def create_slot_labels(self, belief_state: Dict[str, str]) -> Dict[str, int]:
        """
        Create binary slot activation labels
        
        Args:
            belief_state: Current belief state
            
        Returns:
            Binary labels for each slot
        """
        slot_labels = {}
        
        for slot in self.slots:
            # Check if slot is active (has non-empty, non-none value)
            value = belief_state.get(slot, '')
            is_active = value and value not in ['none', ''] and value != []
            slot_labels[slot] = 1 if is_active else 0
        
        return slot_labels
    
    def create_value_labels(self, belief_state: Dict[str, str], 
                          current_utterance: str) -> Dict[str, Union[int, Tuple[int, int]]]:
        """
        Create value labels for active slots
        
        Args:
            belief_state: Current belief state
            current_utterance: Current user utterance
            
        Returns:
            Dictionary of value labels (categorical indices or span positions)
        """
        value_labels = {}
        
        for slot, value in belief_state.items():
            if not value or value in ['none', ''] or value == []:
                continue
                
            if slot in self.categorical_slots:
                # Categorical slot - get class index
                if slot in self.value_mappings and value in self.value_mappings[slot]:
                    value_labels[slot] = self.value_mappings[slot][value]
                else:
                    # Unknown value - assign to special "unknown" class (index 0 usually for 'none')
                    value_labels[slot] = 0
                    
            elif slot in self.span_slots:
                # Span slot - find positions in utterance
                start_pos, end_pos = self._find_span_positions(value, current_utterance)
                if start_pos != -1 and end_pos != -1:
                    value_labels[slot] = (start_pos, end_pos)
        
        return value_labels
    
    def _find_span_positions(self, value: str, utterance: str) -> Tuple[int, int]:
        """
        Find character positions of value span in utterance
        
        Args:
            value: Target value to find
            utterance: Text to search in
            
        Returns:
            Tuple of (start_pos, end_pos) or (-1, -1) if not found
        """
        # Normalize strings
        value_norm = value.lower().strip()
        utterance_norm = utterance.lower()
        
        # Direct match
        start_pos = utterance_norm.find(value_norm)
        if start_pos != -1:
            return start_pos, start_pos + len(value_norm)
        
        # Try partial matches and variations
        # Handle common variations
        variations = [
            value_norm.replace(' ', ''),  # Remove spaces
            value_norm.replace('-', ' '),  # Replace hyphens with spaces
            value_norm.replace('hotel', '').strip(),  # Remove 'hotel' suffix
            value_norm.replace('restaurant', '').strip(),  # Remove 'restaurant' suffix
        ]
        
        for variation in variations:
            if variation and len(variation) > 2:  # Avoid very short matches
                start_pos = utterance_norm.find(variation)
                if start_pos != -1:
                    return start_pos, start_pos + len(variation)
        
        # If still not found, return -1, -1
        return -1, -1
    
    def process_dialog(self, dialog_data: Dict) -> List[DSTExample]:
        """
        Process single dialog into DST examples
        
        Args:
            dialog_data: Dialog data from MultiWOZ
            
        Returns:
            List of DST examples (one per turn)
        """
        dialogue_id = dialog_data['dialogue_id']
        turns = dialog_data['turns']
        examples = []
        
        # Build dialog context graph once for the entire dialog
        dialog_context = self.dialog_builder.build_dialog_context(dialog_data)
        dialog_graph = self.dialog_builder.build_turn_graph(dialog_context)
        
        for turn_idx, turn in enumerate(turns):
            if turn['speaker'] == 'user':  # Only process user turns
                # Format dialog history
                history = self.format_dialog_history(turns, turn_idx)
                
                # Create labels
                belief_state = turn.get('belief_state', {})
                domain_labels = self.create_domain_labels(belief_state)
                slot_labels = self.create_slot_labels(belief_state)
                value_labels = self.create_value_labels(belief_state, turn['utterance'])
                
                # Create example
                example = DSTExample(
                    dialogue_id=dialogue_id,
                    turn_id=turn['turn_id'],
                    dialog_history=history,
                    current_utterance=turn['utterance'],
                    belief_state=belief_state,
                    domain_labels=domain_labels,
                    slot_labels=slot_labels,
                    value_labels=value_labels,
                    schema_graph={},  # Will be filled by schema builder
                    dialog_graph=dialog_graph,
                    turn_graph={}  # Will be filled with turn-specific info
                )
                
                examples.append(example)
        
        return examples
    
    def process_dataset_split(self, split_name: str) -> List[DSTExample]:
        """
        Process entire dataset split (train/val/test)
        
        Args:
            split_name: Name of split ('train', 'val', 'test')
            
        Returns:
            List of all DST examples in the split
        """
        # Load data
        data_path = os.path.join(self.data_dir, f"{split_name}.json")
        with open(data_path, 'r') as f:
            dialogs = json.load(f)
        
        print(f"Processing {len(dialogs)} dialogs from {split_name} split...")
        
        all_examples = []
        for i, dialog in enumerate(dialogs):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(dialogs)} dialogs")
            
            examples = self.process_dialog(dialog)
            all_examples.extend(examples)
        
        print(f"Created {len(all_examples)} examples from {split_name} split")
        return all_examples
    
    def save_processed_data(self, examples: List[DSTExample], output_path: str):
        """Save processed examples to disk"""
        with open(output_path, 'wb') as f:
            pickle.dump(examples, f)
        print(f"Saved {len(examples)} examples to {output_path}")
    
    def load_processed_data(self, input_path: str) -> List[DSTExample]:
        """Load processed examples from disk"""
        with open(input_path, 'rb') as f:
            examples = pickle.load(f)
        print(f"Loaded {len(examples)} examples from {input_path}")
        return examples


class GraphDSTDataLoader:
    """Data loader for GraphDST training"""
    
    def __init__(self, examples: List[DSTExample], tokenizer, schema_graph,
                 batch_size: int = 16, max_length: int = 512, shuffle: bool = True):
        """
        Initialize data loader
        
        Args:
            examples: List of DST examples
            tokenizer: Text tokenizer (e.g., BERT tokenizer)
            schema_graph: Schema graph from schema builder
            batch_size: Batch size
            max_length: Maximum sequence length
            shuffle: Whether to shuffle data
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.schema_graph = schema_graph
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        
        self.current_idx = 0
        if shuffle:
            self._shuffle_examples()
    
    def _shuffle_examples(self):
        """Shuffle examples"""
        import random
        random.shuffle(self.examples)
    
    def __len__(self):
        """Number of batches"""
        return (len(self.examples) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterator interface"""
        self.current_idx = 0
        if self.shuffle:
            self._shuffle_examples()
        return self
    
    def __next__(self):
        """Get next batch"""
        if self.current_idx >= len(self.examples):
            raise StopIteration
        
        # Get batch examples
        batch_examples = self.examples[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        # Process batch
        batch_data = self._process_batch(batch_examples)
        return batch_data
    
    def _process_batch(self, examples: List[DSTExample]) -> Dict:
        """
        Process batch of examples into model inputs
        
        Args:
            examples: Batch of DST examples
            
        Returns:
            Dictionary of batched inputs
        """
        batch_size = len(examples)
        
        # Prepare text inputs
        input_texts = []
        dialogue_ids = []
        turn_ids = []
        
        for example in examples:
            # Format input as: [CLS] history [SEP] current_utterance [SEP]
            if example.dialog_history:
                text = f"{example.dialog_history} [SEP] {example.current_utterance}"
            else:
                text = example.current_utterance
                
            input_texts.append(text)
            dialogue_ids.append(example.dialogue_id)
            turn_ids.append(example.turn_id)
        
        # Tokenize texts
        tokenized = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        # Prepare labels
        labels = self._prepare_labels(examples, tokenized)
        
        # Prepare graph data
        graph_data = self._prepare_graph_data(examples)
        
        batch_data = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'offset_mapping': tokenized.get('offset_mapping'),
            'dialogue_ids': dialogue_ids,
            'turn_ids': turn_ids,
            'labels': labels,
            'graph_data': graph_data,
            'schema_graph': self.schema_graph
        }
        
        return batch_data
    
    def _prepare_labels(self, examples: List[DSTExample], tokenized) -> Dict:
        """Prepare labels for batch"""
        batch_size = len(examples)
        
        labels = {
            'domain_labels': [],
            'slot_labels': {},  
            'value_labels': {}
        }
        
        # Domain labels
        for example in examples:
            labels['domain_labels'].append(example.domain_labels)
        
        # Slot labels (binary activation)
        slot_names = examples[0].slot_labels.keys()  # Assume all examples have same slots
        
        for slot_name in slot_names:
            slot_labels = []
            for example in examples:
                slot_labels.append(example.slot_labels[slot_name])
            labels['slot_labels'][slot_name] = slot_labels
        
        # Value labels
        for example in examples:
            for slot_name, value_label in example.value_labels.items():
                if slot_name not in labels['value_labels']:
                    labels['value_labels'][slot_name] = []
                labels['value_labels'][slot_name].append(value_label)
        
        return labels
    
    def _prepare_graph_data(self, examples: List[DSTExample]) -> Dict:
        """Prepare graph data for batch"""
        dialog_graphs = []
        turn_graphs = []
        
        for example in examples:
            dialog_graphs.append(example.dialog_graph)
            turn_graphs.append(example.turn_graph)
        
        return {
            'dialog_graphs': dialog_graphs,
            'turn_graphs': turn_graphs
        }


def create_data_loaders(data_dir: str, ontology_path: str, tokenizer, 
                       batch_size: int = 16, max_length: int = 512) -> Dict:
    """
    Create data loaders for train/val/test splits
    
    Args:
        data_dir: Directory containing MultiWOZ data
        ontology_path: Path to ontology.json
        tokenizer: Text tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of data loaders for each split
    """
    # Initialize processor
    processor = MultiWOZProcessor(data_dir, ontology_path)
    
    # Initialize graph builders (would be actual imports in real implementation)
    # processor.schema_builder = SchemaGraphBuilder(ontology_path)
    # processor.dialog_builder = DialogContextGraph(ontology_path)
    
    # Build schema graph once
    schema_graph = {}  # Would be processor.schema_builder.build_hetero_graph()
    
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Check if processed data exists
        processed_path = os.path.join(data_dir, f"processed_{split}.pkl")
        
        if os.path.exists(processed_path):
            print(f"Loading processed data from {processed_path}")
            examples = processor.load_processed_data(processed_path)
        else:
            print(f"Processing raw data...")
            examples = processor.process_dataset_split(split)
            processor.save_processed_data(examples, processed_path)
        
        # Create data loader
        shuffle = (split == 'train')  # Only shuffle training data
        
        data_loader = GraphDSTDataLoader(
            examples=examples,
            tokenizer=tokenizer,
            schema_graph=schema_graph,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=shuffle
        )
        
        data_loaders[split] = data_loader
    
    return data_loaders


def get_dataset_statistics(data_loaders: Dict) -> Dict:
    """Get statistics about the processed dataset"""
    stats = {}
    
    for split_name, data_loader in data_loaders.items():
        examples = data_loader.examples
        
        # Basic statistics
        num_examples = len(examples)
        num_dialogs = len(set(ex.dialogue_id for ex in examples))
        
        # Domain statistics
        domain_counts = [0] * 5  # 5 domains
        for example in examples:
            for i, active in enumerate(example.domain_labels):
                domain_counts[i] += active
        
        # Slot statistics  
        slot_counts = {}
        for example in examples:
            for slot, active in example.slot_labels.items():
                slot_counts[slot] = slot_counts.get(slot, 0) + active
        
        # Value statistics
        value_counts = {}
        for example in examples:
            for slot, value in example.value_labels.items():
                value_counts[slot] = value_counts.get(slot, 0) + 1
        
        stats[split_name] = {
            'num_examples': num_examples,
            'num_dialogs': num_dialogs,
            'avg_turns_per_dialog': num_examples / num_dialogs if num_dialogs > 0 else 0,
            'domain_counts': domain_counts,
            'slot_counts': slot_counts,
            'value_counts': value_counts
        }
    
    return stats


if __name__ == "__main__":
    # Example usage
    data_dir = "../../data"
    ontology_path = "../../data/ontology.json"
    
    # Create processor
    processor = MultiWOZProcessor(data_dir, ontology_path)
    
    # Process small sample
    with open(f"{data_dir}/train.json", 'r') as f:
        sample_dialogs = json.load(f)[:5]
    
    print("Processing sample dialogs...")
    all_examples = []
    for dialog in sample_dialogs:
        examples = processor.process_dialog(dialog)
        all_examples.extend(examples)
    
    print(f"Created {len(all_examples)} examples from {len(sample_dialogs)} dialogs")
    
    # Print sample statistics
    if all_examples:
        sample = all_examples[0]
        print(f"\nSample example:")
        print(f"Dialog ID: {sample.dialogue_id}")
        print(f"Turn ID: {sample.turn_id}")
        print(f"History: {sample.dialog_history[:100]}...")
        print(f"Utterance: {sample.current_utterance}")
        print(f"Domain labels: {sample.domain_labels}")
        print(f"Active slots: {sum(sample.slot_labels.values())}")
        print(f"Value labels: {len(sample.value_labels)}")