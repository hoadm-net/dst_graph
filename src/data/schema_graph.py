"""
Schema Graph Construction Module for GraphDST

This module builds the multi-level schema graph from MultiWOZ ontology:
- Level 1: Domain Graph (domain relationships)
- Level 2: Schema Graph (domain-slot relationships)  
- Level 3: Value Graph (slot-value mappings)
"""

import json
import torch
import networkx as nx
from typing import Dict, List, Tuple, Set
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
import numpy as np


class SchemaGraphBuilder:
    """Constructs multi-level schema graphs from MultiWOZ ontology"""
    
    def __init__(self, ontology_path: str):
        """
        Initialize schema graph builder
        
        Args:
            ontology_path: Path to ontology.json file
        """
        self.ontology_path = ontology_path
        self.ontology = self._load_ontology()
        
        # Define domain mappings
        self.domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi']
        self.domain2id = {domain: i for i, domain in enumerate(self.domains)}
        self.id2domain = {i: domain for i, domain in enumerate(self.domains)}
        
        # Build slot mappings
        self.slots = list(self.ontology.keys())
        self.slot2id = {slot: i for i, slot in enumerate(self.slots)}
        self.id2slot = {i: slot for i, slot in enumerate(self.slots)}
        
        # Build value mappings (all unique values across ontology)
        self.values = self._extract_all_values()
        self.value2id = {value: i for i, value in enumerate(self.values)}
        self.id2value = {i: value for i, value in enumerate(self.values)}
        
        # Categorize slots by type
        self.categorical_slots, self.span_slots = self._categorize_slots()
        
    def _load_ontology(self) -> Dict:
        """Load ontology from JSON file"""
        with open(self.ontology_path, 'r') as f:
            return json.load(f)
    
    def _extract_all_values(self) -> List[str]:
        """Extract all unique values from ontology"""
        all_values = set()
        for slot, values in self.ontology.items():
            all_values.update(values)
        return sorted(list(all_values))
    
    def _categorize_slots(self, threshold: int = 50) -> Tuple[List[str], List[str]]:
        """
        Categorize slots into categorical vs span extraction based on vocabulary size
        
        Args:
            threshold: Max vocabulary size for categorical slots
            
        Returns:
            Tuple of (categorical_slots, span_slots)
        """
        categorical = []
        span = []
        
        for slot, values in self.ontology.items():
            if len(values) <= threshold:
                categorical.append(slot)
            else:
                span.append(slot)
                
        return categorical, span
    
    def build_domain_graph(self) -> nx.Graph:
        """
        Build Level 1: Domain Graph
        Domains are connected based on shared slot types (e.g., 'area' appears in multiple domains)
        """
        G = nx.Graph()
        
        # Add domain nodes
        for domain in self.domains:
            G.add_node(domain, node_type='domain')
        
        # Connect domains that share similar slots
        domain_slots = self._get_domain_slots()
        
        for i, domain1 in enumerate(self.domains):
            for j, domain2 in enumerate(self.domains[i+1:], i+1):
                # Count shared slot types (e.g., both have 'area', 'pricerange')
                slots1 = set([slot.split('-')[1] for slot in domain_slots[domain1]])
                slots2 = set([slot.split('-')[1] for slot in domain_slots[domain2]])
                
                shared_slot_types = slots1.intersection(slots2)
                
                if shared_slot_types:
                    # Edge weight = number of shared slot types
                    weight = len(shared_slot_types)
                    G.add_edge(domain1, domain2, weight=weight, shared_slots=list(shared_slot_types))
        
        return G
    
    def build_schema_graph(self) -> nx.DiGraph:
        """
        Build Level 2: Schema Graph  
        Directed graph: Domain -> Slot relationships
        """
        G = nx.DiGraph()
        
        # Add nodes
        for domain in self.domains:
            G.add_node(f"domain_{domain}", node_type='domain', domain=domain)
            
        for slot in self.slots:
            slot_type = 'categorical' if slot in self.categorical_slots else 'span'
            G.add_node(f"slot_{slot}", node_type='slot', slot=slot, slot_type=slot_type)
        
        # Add domain -> slot edges
        domain_slots = self._get_domain_slots()
        
        for domain, slots in domain_slots.items():
            for slot in slots:
                G.add_edge(f"domain_{domain}", f"slot_{slot}", edge_type='contains')
        
        # Add slot -> slot edges for related slots
        self._add_slot_relationships(G)
        
        return G
    
    def build_value_graph(self) -> nx.Graph:
        """
        Build Level 3: Value Graph
        Connect slots to their possible values
        """
        G = nx.Graph()
        
        # Add slot nodes
        for slot in self.slots:
            slot_type = 'categorical' if slot in self.categorical_slots else 'span'
            G.add_node(f"slot_{slot}", node_type='slot', slot=slot, slot_type=slot_type)
        
        # Add value nodes and slot-value edges
        for slot, values in self.ontology.items():
            for value in values:
                value_node = f"value_{value}"
                if not G.has_node(value_node):
                    G.add_node(value_node, node_type='value', value=value)
                
                # Connect slot to value
                G.add_edge(f"slot_{slot}", value_node, edge_type='accepts')
        
        # Add value -> value edges for similar values
        self._add_value_similarities(G)
        
        return G
    
    def _get_domain_slots(self) -> Dict[str, List[str]]:
        """Get slots for each domain"""
        domain_slots = {domain: [] for domain in self.domains}
        
        for slot in self.slots:
            domain = slot.split('-')[0]
            if domain in domain_slots:
                domain_slots[domain].append(slot)
        
        return domain_slots
    
    def _add_slot_relationships(self, G: nx.DiGraph):
        """Add relationships between related slots"""
        # Define slot relationships
        relationships = [
            # Booking related slots
            (['hotel-day', 'restaurant-day', 'train-day'], 'temporal'),
            (['hotel-people', 'restaurant-people', 'train-people'], 'quantity'),
            (['hotel-area', 'restaurant-area', 'attraction-area'], 'location'),
            (['hotel-pricerange', 'restaurant-pricerange'], 'price'),
            
            # Travel related  
            (['train-departure', 'taxi-departure'], 'departure'),
            (['train-destination', 'taxi-destination'], 'destination'),
            (['train-leaveat', 'taxi-leaveat'], 'time_leave'),
            (['train-arriveby', 'taxi-arriveby'], 'time_arrive'),
        ]
        
        for slot_group, relation_type in relationships:
            # Add edges between slots in the same group
            for i, slot1 in enumerate(slot_group):
                for slot2 in slot_group[i+1:]:
                    if f"slot_{slot1}" in G and f"slot_{slot2}" in G:
                        G.add_edge(f"slot_{slot1}", f"slot_{slot2}", 
                                 edge_type='similar', relation=relation_type)
    
    def _add_value_similarities(self, G: nx.Graph):
        """Add edges between similar values"""
        # Group similar values
        similar_groups = [
            ['cheap', 'moderate', 'expensive'],  # Price ranges
            ['centre', 'center', 'central'],      # Center variations
            ['east', 'west', 'north', 'south'],   # Directions
            ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],  # Days
            ['none', 'dontcare'],                 # Special values
        ]
        
        for group in similar_groups:
            for i, value1 in enumerate(group):
                for value2 in group[i+1:]:
                    value1_node = f"value_{value1}"
                    value2_node = f"value_{value2}"
                    
                    if G.has_node(value1_node) and G.has_node(value2_node):
                        G.add_edge(value1_node, value2_node, edge_type='similar')
    
    def build_hetero_graph(self) -> HeteroData:
        """
        Build PyTorch Geometric HeteroData graph combining all levels
        
        Returns:
            HeteroData object with all node types and edge types
        """
        data = HeteroData()
        
        # === Node Features ===
        
        # Domain nodes (5 domains)
        num_domains = len(self.domains)
        data['domain'].x = torch.eye(num_domains)  # One-hot encoding
        data['domain'].domain_names = self.domains
        
        # Slot nodes (37 slots)  
        num_slots = len(self.slots)
        slot_features = []
        slot_types = []
        
        for slot in self.slots:
            # Create slot features: [domain_onehot, slot_type, vocab_size]
            domain = slot.split('-')[0]
            domain_idx = self.domain2id.get(domain, 0)
            domain_onehot = torch.zeros(num_domains)
            domain_onehot[domain_idx] = 1
            
            slot_type = 1 if slot in self.categorical_slots else 0  # 1=categorical, 0=span
            vocab_size = len(self.ontology[slot]) if slot in self.ontology else 1
            
            features = torch.cat([
                domain_onehot, 
                torch.tensor([slot_type, np.log(vocab_size + 1)])
            ])
            slot_features.append(features)
            slot_types.append(slot_type)
        
        data['slot'].x = torch.stack(slot_features)
        data['slot'].slot_names = self.slots
        data['slot'].slot_types = torch.tensor(slot_types)
        
        # Value nodes (all possible values)
        num_values = len(self.values)
        # Use simple embedding indices for values (will be learned)
        data['value'].x = torch.arange(num_values).unsqueeze(1).float()
        data['value'].value_names = self.values
        
        # === Edge Indices ===
        
        # Domain-Domain edges (from domain graph)
        domain_graph = self.build_domain_graph()
        domain_edges = []
        domain_weights = []
        
        for edge in domain_graph.edges(data=True):
            src_idx = self.domain2id[edge[0]]
            dst_idx = self.domain2id[edge[1]]
            weight = edge[2].get('weight', 1.0)
            
            domain_edges.extend([[src_idx, dst_idx], [dst_idx, src_idx]])  # Undirected
            domain_weights.extend([weight, weight])
        
        if domain_edges:
            data['domain', 'connected', 'domain'].edge_index = torch.tensor(domain_edges).t()
            data['domain', 'connected', 'domain'].edge_attr = torch.tensor(domain_weights).float()
        
        # Domain-Slot edges
        domain_slot_edges = []
        domain_slots = self._get_domain_slots()
        
        for domain, slots in domain_slots.items():
            domain_idx = self.domain2id[domain]
            for slot in slots:
                if slot in self.slot2id:
                    slot_idx = self.slot2id[slot]
                    domain_slot_edges.append([domain_idx, slot_idx])
        
        if domain_slot_edges:
            data['domain', 'contains', 'slot'].edge_index = torch.tensor(domain_slot_edges).t()
        
        # Slot-Value edges
        slot_value_edges = []
        
        for slot, values in self.ontology.items():
            if slot in self.slot2id:
                slot_idx = self.slot2id[slot]
                for value in values:
                    if value in self.value2id:
                        value_idx = self.value2id[value]
                        slot_value_edges.append([slot_idx, value_idx])
        
        if slot_value_edges:
            data['slot', 'accepts', 'value'].edge_index = torch.tensor(slot_value_edges).t()
        
        # Slot-Slot edges (similar slots)
        slot_edges = []
        relationships = [
            (['hotel-day', 'restaurant-day', 'train-day'], 'temporal'),
            (['hotel-people', 'restaurant-people', 'train-people'], 'quantity'),
            (['hotel-area', 'restaurant-area', 'attraction-area'], 'location'),
            (['hotel-pricerange', 'restaurant-pricerange'], 'price'),
        ]
        
        for slot_group, _ in relationships:
            for i, slot1 in enumerate(slot_group):
                for slot2 in slot_group[i+1:]:
                    if slot1 in self.slot2id and slot2 in self.slot2id:
                        idx1, idx2 = self.slot2id[slot1], self.slot2id[slot2]
                        slot_edges.extend([[idx1, idx2], [idx2, idx1]])  # Undirected
        
        if slot_edges:
            data['slot', 'similar', 'slot'].edge_index = torch.tensor(slot_edges).t()
        
        return data
    
    def get_slot_info(self) -> Dict:
        """Get comprehensive slot information"""
        return {
            'total_slots': len(self.slots),
            'categorical_slots': len(self.categorical_slots),
            'span_slots': len(self.span_slots),
            'slot_names': self.slots,
            'categorical_slot_names': self.categorical_slots,
            'span_slot_names': self.span_slots,
            'slot2id': self.slot2id,
            'id2slot': self.id2slot
        }
    
    def get_domain_info(self) -> Dict:
        """Get comprehensive domain information"""
        domain_slots = self._get_domain_slots()
        
        return {
            'domains': self.domains,
            'domain2id': self.domain2id,
            'id2domain': self.id2domain,
            'domain_slots': domain_slots,
            'slots_per_domain': {domain: len(slots) for domain, slots in domain_slots.items()}
        }
    
    def visualize_schema(self, save_path: str = None):
        """Visualize the schema graph structure"""
        import matplotlib.pyplot as plt
        
        # Build and visualize domain graph
        domain_graph = self.build_domain_graph()
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(domain_graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(domain_graph, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.8)
        
        # Draw edges with weights
        edges = domain_graph.edges(data=True)
        weights = [edge[2]['weight'] for edge in edges]
        nx.draw_networkx_edges(domain_graph, pos, width=[w*2 for w in weights], 
                              alpha=0.6, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(domain_graph, pos, font_size=10, font_weight='bold')
        
        plt.title("MultiWOZ Domain Graph\n(Edge width = number of shared slot types)")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Example usage
    ontology_path = "../../data/ontology.json"
    builder = SchemaGraphBuilder(ontology_path)
    
    # Build graphs
    hetero_graph = builder.build_hetero_graph()
    print("Built heterogeneous graph:")
    print(hetero_graph)
    
    # Print statistics
    slot_info = builder.get_slot_info()
    domain_info = builder.get_domain_info()
    
    print(f"\nSlot Statistics:")
    print(f"Total slots: {slot_info['total_slots']}")
    print(f"Categorical slots: {slot_info['categorical_slots']}")
    print(f"Span slots: {slot_info['span_slots']}")
    
    print(f"\nDomain Statistics:")
    for domain, count in domain_info['slots_per_domain'].items():
        print(f"{domain}: {count} slots")