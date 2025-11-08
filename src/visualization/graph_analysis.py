"""
Graph Analysis and Metrics Module

This module provides analytical tools for understanding graph structures
and computing graph-based metrics for DST evaluation.
"""

from typing import Dict, List, Tuple, Optional, Set
import json
from collections import defaultdict, Counter
import math


class GraphAnalyzer:
    """Analyze graph structures and compute metrics"""
    
    def __init__(self):
        """Initialize graph analyzer"""
        self.metrics_cache = {}
    
    def analyze_schema_graph(self, schema_builder) -> Dict:
        """
        Analyze schema graph structure and properties
        
        Args:
            schema_builder: SchemaGraphBuilder instance
            
        Returns:
            Dictionary of analysis results
        """
        analysis = {}
        
        # Basic graph statistics
        hetero_graph = schema_builder.build_hetero_graph()
        
        analysis['node_counts'] = {
            'domains': len(schema_builder.domains),
            'slots': len(schema_builder.slots), 
            'values': len(schema_builder.values)
        }
        
        # Domain analysis
        domain_info = schema_builder.get_domain_info()
        analysis['domain_analysis'] = {
            'slots_per_domain': domain_info['slots_per_domain'],
            'avg_slots_per_domain': sum(domain_info['slots_per_domain'].values()) / len(domain_info['slots_per_domain']),
            'most_complex_domain': max(domain_info['slots_per_domain'], key=domain_info['slots_per_domain'].get),
            'least_complex_domain': min(domain_info['slots_per_domain'], key=domain_info['slots_per_domain'].get)
        }
        
        # Slot analysis
        slot_info = schema_builder.get_slot_info()
        analysis['slot_analysis'] = {
            'categorical_ratio': slot_info['categorical_slots'] / slot_info['total_slots'],
            'span_ratio': slot_info['span_slots'] / slot_info['total_slots'],
            'avg_vocab_size_categorical': self._compute_avg_vocab_size(
                schema_builder.ontology, schema_builder.categorical_slots
            ),
            'avg_vocab_size_span': self._compute_avg_vocab_size(
                schema_builder.ontology, schema_builder.span_slots
            )
        }
        
        # Cross-domain relationships
        analysis['cross_domain'] = self._analyze_cross_domain_relationships(schema_builder)
        
        # Graph connectivity
        analysis['connectivity'] = self._analyze_graph_connectivity(schema_builder)
        
        return analysis
    
    def analyze_dialog_graph(self, dialog_context) -> Dict:
        """
        Analyze dialog context graph
        
        Args:
            dialog_context: DialogContext object
            
        Returns:
            Dictionary of analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['basic_stats'] = {
            'num_turns': len(dialog_context.turns),
            'num_entities': len(dialog_context.entities),
            'num_coreferences': len(dialog_context.coreferences),
            'avg_entities_per_turn': sum(len(turn.entities) for turn in dialog_context.turns) / len(dialog_context.turns)
        }
        
        # Turn analysis
        user_turns = [t for t in dialog_context.turns if t.speaker == 'user']
        system_turns = [t for t in dialog_context.turns if t.speaker == 'system']
        
        analysis['turn_analysis'] = {
            'user_turns': len(user_turns),
            'system_turns': len(system_turns),
            'avg_user_entities': sum(len(t.entities) for t in user_turns) / len(user_turns) if user_turns else 0,
            'avg_system_entities': sum(len(t.entities) for t in system_turns) / len(system_turns) if system_turns else 0
        }
        
        # Entity analysis
        entity_frequencies = Counter()
        for turn in dialog_context.turns:
            entity_frequencies.update(turn.entities)
        
        analysis['entity_analysis'] = {
            'most_frequent_entities': entity_frequencies.most_common(5),
            'unique_entities': len(entity_frequencies),
            'avg_entity_frequency': sum(entity_frequencies.values()) / len(entity_frequencies) if entity_frequencies else 0
        }
        
        # Coreference analysis
        coreference_distances = []
        for turn1_id, turn2_id, entity in dialog_context.coreferences:
            distance = abs(turn2_id - turn1_id)
            coreference_distances.append(distance)
        
        if coreference_distances:
            analysis['coreference_analysis'] = {
                'avg_distance': sum(coreference_distances) / len(coreference_distances),
                'max_distance': max(coreference_distances),
                'min_distance': min(coreference_distances)
            }
        else:
            analysis['coreference_analysis'] = {'avg_distance': 0, 'max_distance': 0, 'min_distance': 0}
        
        return analysis
    
    def compute_graph_complexity_metrics(self, schema_builder) -> Dict:
        """
        Compute complexity metrics for graph structure
        
        Args:
            schema_builder: SchemaGraphBuilder instance
            
        Returns:
            Dictionary of complexity metrics
        """
        metrics = {}
        
        # Schema complexity
        domain_slots = schema_builder._get_domain_slots()
        
        # Gini coefficient for slot distribution across domains
        slot_counts = list(domain_slots.values())
        slot_counts = [len(slots) for slots in slot_counts]
        metrics['slot_distribution_gini'] = self._compute_gini_coefficient(slot_counts)
        
        # Value vocabulary complexity
        vocab_sizes = [len(values) for values in schema_builder.ontology.values()]
        metrics['vocab_size_stats'] = {
            'mean': sum(vocab_sizes) / len(vocab_sizes),
            'std': math.sqrt(sum((x - sum(vocab_sizes)/len(vocab_sizes))**2 for x in vocab_sizes) / len(vocab_sizes)),
            'min': min(vocab_sizes),
            'max': max(vocab_sizes)
        }
        
        # Graph density (theoretical)
        num_domains = len(schema_builder.domains)
        num_slots = len(schema_builder.slots)
        
        # Maximum possible edges in bipartite graph (domain-slot)
        max_edges = num_domains * num_slots
        actual_edges = sum(len(slots) for slots in domain_slots.values())
        metrics['schema_density'] = actual_edges / max_edges if max_edges > 0 else 0
        
        return metrics
    
    def _compute_avg_vocab_size(self, ontology: Dict, slots: List[str]) -> float:
        """Compute average vocabulary size for given slots"""
        if not slots:
            return 0.0
        
        total_size = sum(len(ontology.get(slot, [])) for slot in slots)
        return total_size / len(slots)
    
    def _analyze_cross_domain_relationships(self, schema_builder) -> Dict:
        """Analyze relationships between domains"""
        domain_slots = schema_builder._get_domain_slots()
        
        # Find shared slot types (e.g., 'area' appears in multiple domains)
        slot_types = defaultdict(list)
        for domain, slots in domain_slots.items():
            for slot in slots:
                slot_type = slot.split('-')[1] if '-' in slot else slot
                slot_types[slot_type].append(domain)
        
        # Count shared slot types
        shared_slot_types = {
            slot_type: domains for slot_type, domains in slot_types.items() 
            if len(domains) > 1
        }
        
        # Compute domain similarity based on shared slot types
        domain_similarities = {}
        domains = schema_builder.domains
        
        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                # Get slot types for each domain
                types1 = set(slot.split('-')[1] for slot in domain_slots[domain1] if '-' in slot)
                types2 = set(slot.split('-')[1] for slot in domain_slots[domain2] if '-' in slot)
                
                # Jaccard similarity
                intersection = len(types1.intersection(types2))
                union = len(types1.union(types2))
                similarity = intersection / union if union > 0 else 0
                
                domain_similarities[f"{domain1}-{domain2}"] = similarity
        
        return {
            'shared_slot_types': dict(shared_slot_types),
            'num_shared_types': len(shared_slot_types),
            'domain_similarities': domain_similarities,
            'avg_domain_similarity': sum(domain_similarities.values()) / len(domain_similarities) if domain_similarities else 0
        }
    
    def _analyze_graph_connectivity(self, schema_builder) -> Dict:
        """Analyze graph connectivity properties"""
        # Build domain graph for connectivity analysis
        domain_graph = schema_builder.build_domain_graph()
        
        # Basic connectivity metrics
        num_nodes = len(domain_graph.nodes())
        num_edges = len(domain_graph.edges())
        
        density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        # Degree statistics
        degrees = [degree for node, degree in domain_graph.degree()]
        
        connectivity = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0
        }
        
        return connectivity
    
    def _compute_gini_coefficient(self, values: List[float]) -> float:
        """Compute Gini coefficient for measuring inequality"""
        if not values or all(v == 0 for v in values):
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Compute Gini coefficient
        numerator = sum((2 * i - n - 1) * v for i, v in enumerate(sorted_values, 1))
        denominator = n * sum(sorted_values)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def generate_analysis_report(self, schema_builder, dialog_contexts: List = None) -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            schema_builder: SchemaGraphBuilder instance
            dialog_contexts: List of DialogContext objects (optional)
            
        Returns:
            Formatted analysis report
        """
        report_lines = []
        report_lines.append("GraphDST Analysis Report")
        report_lines.append("=" * 50)
        
        # Schema analysis
        schema_analysis = self.analyze_schema_graph(schema_builder)
        
        report_lines.append("\n📊 Schema Graph Analysis:")
        report_lines.append(f"  • Total domains: {schema_analysis['node_counts']['domains']}")
        report_lines.append(f"  • Total slots: {schema_analysis['node_counts']['slots']}")
        report_lines.append(f"  • Total values: {schema_analysis['node_counts']['values']}")
        
        report_lines.append(f"\n🏢 Domain Complexity:")
        domain_analysis = schema_analysis['domain_analysis']
        report_lines.append(f"  • Average slots per domain: {domain_analysis['avg_slots_per_domain']:.2f}")
        report_lines.append(f"  • Most complex domain: {domain_analysis['most_complex_domain']}")
        report_lines.append(f"  • Least complex domain: {domain_analysis['least_complex_domain']}")
        
        report_lines.append(f"\n🎯 Slot Distribution:")
        slot_analysis = schema_analysis['slot_analysis']
        report_lines.append(f"  • Categorical slots: {slot_analysis['categorical_ratio']:.1%}")
        report_lines.append(f"  • Span slots: {slot_analysis['span_ratio']:.1%}")
        report_lines.append(f"  • Avg categorical vocab size: {slot_analysis['avg_vocab_size_categorical']:.1f}")
        
        report_lines.append(f"\n🔗 Cross-Domain Relationships:")
        cross_domain = schema_analysis['cross_domain']
        report_lines.append(f"  • Shared slot types: {cross_domain['num_shared_types']}")
        report_lines.append(f"  • Average domain similarity: {cross_domain['avg_domain_similarity']:.3f}")
        
        # Dialog analysis (if provided)
        if dialog_contexts:
            report_lines.append(f"\n💬 Dialog Analysis ({len(dialog_contexts)} dialogs):")
            
            avg_turns = sum(len(ctx.turns) for ctx in dialog_contexts) / len(dialog_contexts)
            avg_entities = sum(len(ctx.entities) for ctx in dialog_contexts) / len(dialog_contexts)
            
            report_lines.append(f"  • Average turns per dialog: {avg_turns:.1f}")
            report_lines.append(f"  • Average entities per dialog: {avg_entities:.1f}")
        
        # Complexity metrics
        complexity = self.compute_graph_complexity_metrics(schema_builder)
        report_lines.append(f"\n📈 Complexity Metrics:")
        report_lines.append(f"  • Slot distribution Gini: {complexity['slot_distribution_gini']:.3f}")
        report_lines.append(f"  • Schema density: {complexity['schema_density']:.3f}")
        report_lines.append(f"  • Vocab size std: {complexity['vocab_size_stats']['std']:.1f}")
        
        return "\n".join(report_lines)


def analyze_multiwoz_dataset(data_dir: str, ontology_path: str) -> Dict:
    """
    Comprehensive analysis of MultiWOZ dataset structure
    
    Args:
        data_dir: Directory containing MultiWOZ data
        ontology_path: Path to ontology.json
        
    Returns:
        Complete analysis results
    """
    analyzer = GraphAnalyzer()
    
    # Initialize builders (would be actual imports in real implementation)
    # from src.data.schema_graph import SchemaGraphBuilder
    # from src.data.dialog_graph import DialogContextGraph
    
    # schema_builder = SchemaGraphBuilder(ontology_path)
    # dialog_builder = DialogContextGraph(ontology_path)
    
    results = {
        'dataset_path': data_dir,
        'ontology_path': ontology_path,
        'analysis_timestamp': None,  # Would use datetime.now()
        'schema_analysis': {},       # analyzer.analyze_schema_graph(schema_builder)
        'complexity_metrics': {},    # analyzer.compute_graph_complexity_metrics(schema_builder)
        'recommendations': []
    }
    
    # Add recommendations based on analysis
    # This would be populated based on actual analysis results
    results['recommendations'] = [
        "Consider balancing slot distribution across domains",
        "Optimize vocabulary sizes for categorical slots",
        "Leverage cross-domain relationships for transfer learning",
        "Use graph density metrics for architecture tuning"
    ]
    
    return results


if __name__ == "__main__":
    # Example usage
    analyzer = GraphAnalyzer()
    
    # This would work with actual data
    # results = analyze_multiwoz_dataset("data", "data/ontology.json")
    # print(json.dumps(results, indent=2))
    
    print("GraphAnalyzer initialized. Use with actual schema_builder for analysis.")