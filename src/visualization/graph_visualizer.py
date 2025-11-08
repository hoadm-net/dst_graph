"""
Interactive Graph Visualization for GraphDST using Streamlit and pyvis

This module provides interactive visualization tools for:
- Schema graphs (Domain-Slot-Value hierarchy)
- Dialog context graphs (Turn-level connections)
- Attention weights visualization
- Model interpretability analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from pyvis.network import Network
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
from pathlib import Path


class GraphVisualizer:
    """Interactive graph visualizer for GraphDST"""
    
    def __init__(self, width: str = "100%", height: str = "600px"):
        """
        Initialize graph visualizer
        
        Args:
            width: Network visualization width
            height: Network visualization height
        """
        self.width = width
        self.height = height
        self.color_schemes = {
            'domain': '#FF6B6B',
            'slot': '#4ECDC4', 
            'value': '#45B7D1',
            'utterance': '#96CEB4',
            'entity': '#FFEAA7',
            'turn': '#DDA0DD'
        }
    
    def create_schema_network(self, schema_graph: Dict, 
                            show_values: bool = True,
                            filter_domains: List[str] = None) -> Network:
        """
        Create interactive schema graph visualization
        
        Args:
            schema_graph: Schema graph data
            show_values: Whether to show value nodes
            filter_domains: List of domains to show (None for all)
            
        Returns:
            Pyvis Network object
        """
        net = Network(width=self.width, height=self.height, 
                     bgcolor="#ffffff", font_color="black")
        
        # Configure physics
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            }
          }
        }
        """)
        
        # Add domain nodes
        domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi']
        if filter_domains:
            domains = [d for d in domains if d in filter_domains]
            
        for domain in domains:
            net.add_node(f"domain_{domain}", 
                        label=domain.title(),
                        color=self.color_schemes['domain'],
                        size=30,
                        title=f"Domain: {domain}",
                        group="domain")
        
        # Add domain connections
        domain_connections = [
            ('hotel', 'restaurant', 'Both have area, pricerange'),
            ('train', 'taxi', 'Both have departure, destination'),
            ('hotel', 'attraction', 'Both have area'),
            ('restaurant', 'attraction', 'Both have area')
        ]
        
        for src, dst, reason in domain_connections:
            if (not filter_domains or 
                (src in filter_domains and dst in filter_domains)):
                net.add_edge(f"domain_{src}", f"domain_{dst}",
                           title=reason, width=2, color="#cccccc")
        
        # Add slot nodes and connections
        slot_domain_map = self._get_slot_domain_mapping()
        
        for slot, domain in slot_domain_map.items():
            if not filter_domains or domain in filter_domains:
                net.add_node(f"slot_{slot}",
                           label=slot.split('-')[1],
                           color=self.color_schemes['slot'],
                           size=20,
                           title=f"Slot: {slot}",
                           group="slot")
                
                # Connect domain to slot
                net.add_edge(f"domain_{domain}", f"slot_{slot}",
                           color="#888888", width=2)
        
        # Add value nodes if requested
        if show_values:
            sample_values = self._get_sample_values()
            for slot, values in sample_values.items():
                if not filter_domains or slot.split('-')[0] in filter_domains:
                    for i, value in enumerate(values[:5]):  # Show first 5 values
                        value_id = f"value_{slot}_{i}"
                        net.add_node(value_id,
                                   label=value[:15] + "..." if len(value) > 15 else value,
                                   color=self.color_schemes['value'],
                                   size=15,
                                   title=f"Value: {value}",
                                   group="value")
                        
                        # Connect slot to value
                        net.add_edge(f"slot_{slot}", value_id,
                                   color="#bbbbbb", width=1)
        
        return net
    
    def create_dialog_network(self, dialog_data: Dict,
                            show_entities: bool = True,
                            max_turns: int = 10) -> Network:
        """
        Create dialog context graph visualization
        
        Args:
            dialog_data: Dialog context data
            show_entities: Whether to show entity nodes
            max_turns: Maximum number of turns to show
            
        Returns:
            Pyvis Network object
        """
        net = Network(width=self.width, height=self.height,
                     bgcolor="#ffffff", font_color="black")
        
        # Configure for temporal layout
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "hierarchicalRepulsion": {
              "centralGravity": 0.0,
              "springLength": 100,
              "springConstant": 0.01,
              "nodeDistance": 120,
              "damping": 0.09
            },
            "maxVelocity": 50,
            "solver": "hierarchicalRepulsion"
          },
          "layout": {
            "hierarchical": {
              "enabled": true,
              "levelSeparation": 150,
              "nodeSpacing": 100,
              "treeSpacing": 200,
              "blockShifting": true,
              "edgeMinimization": true,
              "parentCentralization": true,
              "direction": "LR",
              "sortMethod": "directed"
            }
          }
        }
        """)
        
        turns = dialog_data.get('turns', [])[:max_turns]
        
        # Add turn nodes
        for i, turn in enumerate(turns):
            speaker = turn.get('speaker', 'unknown')
            utterance = turn.get('utterance', '')
            
            # Truncate long utterances
            display_text = utterance[:50] + "..." if len(utterance) > 50 else utterance
            
            color = self.color_schemes['turn'] if speaker == 'user' else '#FFB6C1'
            
            net.add_node(f"turn_{i}",
                       label=f"T{i}: {speaker}",
                       color=color,
                       size=25,
                       title=f"Turn {i} ({speaker}): {utterance}",
                       group=f"turn_{speaker}",
                       level=i)  # For hierarchical layout
        
        # Add sequential connections
        for i in range(len(turns) - 1):
            net.add_edge(f"turn_{i}", f"turn_{i+1}",
                       color="#666666", width=3,
                       title="Sequential flow")
        
        # Add entity nodes and connections
        if show_entities:
            entities = dialog_data.get('entities', {})
            entity_positions = {}
            
            for entity, turn_ids in entities.items():
                if len(turn_ids) > 1:  # Only show entities mentioned multiple times
                    entity_id = f"entity_{entity}"
                    net.add_node(entity_id,
                               label=entity,
                               color=self.color_schemes['entity'],
                               size=15,
                               title=f"Entity: {entity}",
                               group="entity")
                    
                    # Connect to turns where entity appears
                    for turn_id in turn_ids:
                        if turn_id < len(turns):
                            net.add_edge(entity_id, f"turn_{turn_id}",
                                       color="#FFA500", width=1,
                                       title=f"Entity mentioned in turn {turn_id}")
        
        # Add coreference connections
        coreferences = dialog_data.get('coreferences', [])
        for turn1, turn2, entity in coreferences:
            if turn1 < len(turns) and turn2 < len(turns):
                net.add_edge(f"turn_{turn1}", f"turn_{turn2}",
                           color="#FF4500", width=2, dashes=True,
                           title=f"Coreference: {entity}")
        
        return net
    
    def create_attention_heatmap(self, attention_weights: np.ndarray,
                               labels: List[str] = None) -> go.Figure:
        """
        Create attention weights heatmap
        
        Args:
            attention_weights: Attention weight matrix
            labels: Node labels
            
        Returns:
            Plotly figure
        """
        if labels is None:
            labels = [f"Node_{i}" for i in range(attention_weights.shape[0])]
        
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            x=labels,
            y=labels,
            colorscale='Viridis',
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Attention Weights Heatmap",
            xaxis_title="Target Nodes",
            yaxis_title="Source Nodes",
            width=600,
            height=600
        )
        
        return fig
    
    def create_metrics_dashboard(self, metrics_history: List[Dict]) -> go.Figure:
        """
        Create training metrics dashboard
        
        Args:
            metrics_history: List of metrics dictionaries
            
        Returns:
            Plotly figure with subplots
        """
        if not metrics_history:
            return go.Figure()
        
        # Extract metrics
        epochs = list(range(len(metrics_history)))
        jga = [m.get('joint_goal_accuracy', 0) for m in metrics_history]
        slot_f1 = [m.get('slot_avg_f1', 0) for m in metrics_history]
        domain_f1 = [m.get('domain_avg_f1', 0) for m in metrics_history]
        value_acc = [m.get('value_accuracy', 0) for m in metrics_history]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Joint Goal Accuracy', 'Slot F1', 'Domain F1', 'Value Accuracy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=epochs, y=jga, name='JGA', line=dict(color='red')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=epochs, y=slot_f1, name='Slot F1', line=dict(color='blue')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=epochs, y=domain_f1, name='Domain F1', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=epochs, y=value_acc, name='Value Acc', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Training Metrics Dashboard",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def _get_slot_domain_mapping(self) -> Dict[str, str]:
        """Get mapping of slots to domains"""
        return {
            'hotel-area': 'hotel', 'hotel-pricerange': 'hotel', 'hotel-type': 'hotel',
            'hotel-parking': 'hotel', 'hotel-internet': 'hotel', 'hotel-name': 'hotel',
            'hotel-stars': 'hotel', 'hotel-stay': 'hotel', 'hotel-day': 'hotel',
            'hotel-people': 'hotel', 'hotel-booked': 'hotel',
            
            'restaurant-area': 'restaurant', 'restaurant-pricerange': 'restaurant',
            'restaurant-food': 'restaurant', 'restaurant-name': 'restaurant',
            'restaurant-time': 'restaurant', 'restaurant-day': 'restaurant',
            'restaurant-people': 'restaurant', 'restaurant-booked': 'restaurant',
            
            'attraction-area': 'attraction', 'attraction-type': 'attraction',
            'attraction-name': 'attraction',
            
            'train-departure': 'train', 'train-destination': 'train',
            'train-day': 'train', 'train-leaveat': 'train', 'train-arriveby': 'train',
            'train-people': 'train', 'train-booked': 'train',
            
            'taxi-departure': 'taxi', 'taxi-destination': 'taxi',
            'taxi-leaveat': 'taxi', 'taxi-arriveby': 'taxi', 'taxi-booked': 'taxi'
        }
    
    def _get_sample_values(self) -> Dict[str, List[str]]:
        """Get sample values for slots"""
        return {
            'hotel-area': ['centre', 'east', 'west', 'north', 'south'],
            'hotel-pricerange': ['cheap', 'moderate', 'expensive'],
            'hotel-type': ['hotel', 'guest house', 'lodge'],
            'restaurant-food': ['british', 'chinese', 'italian', 'indian', 'french'],
            'restaurant-area': ['centre', 'east', 'west', 'north', 'south'],
            'attraction-type': ['museum', 'park', 'theatre', 'church', 'gallery'],
            'train-day': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
        }
    
    def save_network(self, net: Network, filename: str) -> str:
        """Save network to HTML file and return path"""
        temp_dir = tempfile.mkdtemp()
        filepath = os.path.join(temp_dir, filename)
        net.save_graph(filepath)
        return filepath


def load_sample_data() -> Dict:
    """Load sample data for visualization"""
    return {
        'schema_graph': {
            'domains': ['hotel', 'restaurant', 'attraction', 'train', 'taxi'],
            'slots': ['hotel-area', 'hotel-pricerange', 'restaurant-food'],
            'values': {'hotel-area': ['centre', 'east'], 'hotel-pricerange': ['cheap']}
        },
        'dialog_data': {
            'dialogue_id': 'sample_001',
            'turns': [
                {
                    'turn_id': 0,
                    'speaker': 'user',
                    'utterance': 'I need a cheap hotel in the centre',
                    'entities': ['cheap', 'hotel', 'centre']
                },
                {
                    'turn_id': 1, 
                    'speaker': 'system',
                    'utterance': 'I found several cheap hotels in the centre area',
                    'entities': ['cheap', 'hotels', 'centre']
                },
                {
                    'turn_id': 2,
                    'speaker': 'user', 
                    'utterance': 'I also need parking',
                    'entities': ['parking']
                }
            ],
            'entities': {
                'cheap': [0, 1],
                'hotel': [0, 1], 
                'centre': [0, 1],
                'parking': [2]
            },
            'coreferences': [(0, 1, 'hotel'), (0, 1, 'centre')]
        },
        'attention_weights': np.random.rand(8, 8),
        'metrics_history': [
            {'joint_goal_accuracy': 0.1, 'slot_avg_f1': 0.7, 'domain_avg_f1': 0.8, 'value_accuracy': 0.6},
            {'joint_goal_accuracy': 0.2, 'slot_avg_f1': 0.75, 'domain_avg_f1': 0.82, 'value_accuracy': 0.65},
            {'joint_goal_accuracy': 0.35, 'slot_avg_f1': 0.82, 'domain_avg_f1': 0.87, 'value_accuracy': 0.78},
            {'joint_goal_accuracy': 0.45, 'slot_avg_f1': 0.89, 'domain_avg_f1': 0.91, 'value_accuracy': 0.85},
            {'joint_goal_accuracy': 0.52, 'slot_avg_f1': 0.94, 'domain_avg_f1': 0.94, 'value_accuracy': 0.91}
        ]
    }


if __name__ == "__main__":
    # Test the visualizer
    visualizer = GraphVisualizer()
    sample_data = load_sample_data()
    
    # Create schema network
    schema_net = visualizer.create_schema_network(
        sample_data['schema_graph'], 
        show_values=True
    )
    
    # Save to file
    schema_path = visualizer.save_network(schema_net, "schema_graph.html")
    print(f"Schema graph saved to: {schema_path}")
    
    # Create dialog network
    dialog_net = visualizer.create_dialog_network(
        sample_data['dialog_data'],
        show_entities=True
    )
    
    dialog_path = visualizer.save_network(dialog_net, "dialog_graph.html")
    print(f"Dialog graph saved to: {dialog_path}")