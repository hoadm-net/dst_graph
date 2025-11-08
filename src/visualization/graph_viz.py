"""
Graph Visualization Module for GraphDST

This module provides interactive visualization for:
- Schema graphs (domain-slot-value relationships)
- Dialog context graphs (temporal connections)
- Attention weights and message passing
"""

import streamlit as st
import networkx as nx
from pyvis.network import Network
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
import tempfile
import os
from pathlib import Path


class SchemaGraphVisualizer:
    """Visualize schema graphs with interactive features"""
    
    def __init__(self, schema_builder):
        """
        Initialize schema graph visualizer
        
        Args:
            schema_builder: SchemaGraphBuilder instance
        """
        self.schema_builder = schema_builder
        self.colors = {
            'domain': '#FF6B6B',     # Red
            'slot': '#4ECDC4',       # Teal  
            'value': '#45B7D1',      # Blue
            'categorical': '#96CEB4', # Green
            'span': '#FFEAA7'        # Yellow
        }
        
    def create_schema_network(self, show_values: bool = False, 
                            filter_domains: List[str] = None) -> Network:
        """
        Create interactive schema network
        
        Args:
            show_values: Whether to include value nodes
            filter_domains: List of domains to include (None for all)
            
        Returns:
            Pyvis Network object
        """
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            }
          }
        }
        """)
        
        # Build networkx graph
        schema_graph = self.schema_builder.build_schema_graph()
        
        # Filter domains if specified
        if filter_domains:
            nodes_to_keep = set()
            for node in schema_graph.nodes():
                if node.startswith('domain_'):
                    domain = node.replace('domain_', '')
                    if domain in filter_domains:
                        nodes_to_keep.add(node)
                elif node.startswith('slot_'):
                    slot = node.replace('slot_', '')
                    domain = slot.split('-')[0]
                    if domain in filter_domains:
                        nodes_to_keep.add(node)
                        
            schema_graph = schema_graph.subgraph(nodes_to_keep)
        
        # Add nodes
        for node, data in schema_graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            
            if node_type == 'domain':
                color = self.colors['domain']
                size = 30
                shape = 'dot'
                label = data.get('domain', node)
                title = f"Domain: {label}"
                
            elif node_type == 'slot':
                slot_name = data.get('slot', node)
                slot_type = data.get('slot_type', 'unknown')
                
                color = self.colors.get(slot_type, self.colors['slot'])
                size = 20
                shape = 'diamond' if slot_type == 'categorical' else 'square'
                label = slot_name.replace('_', '-')
                title = f"Slot: {label} ({slot_type})"
                
            else:
                continue  # Skip other node types for now
                
            net.add_node(node, label=label, color=color, size=size, 
                        shape=shape, title=title)
        
        # Add edges
        for source, target, data in schema_graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            
            if edge_type == 'contains':
                color = '#888888'
                width = 2
                title = "contains"
            elif edge_type == 'similar':
                color = '#00FF00'
                width = 1
                title = f"similar ({data.get('relation', 'unknown')})"
            else:
                color = '#CCCCCC'
                width = 1
                title = edge_type
                
            net.add_edge(source, target, color=color, width=width, title=title)
        
        return net
    
    def create_value_network(self, selected_slots: List[str] = None) -> Network:
        """
        Create network showing slot-value relationships
        
        Args:
            selected_slots: List of slots to visualize (None for all)
            
        Returns:
            Pyvis Network object
        """
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
        
        value_graph = self.schema_builder.build_value_graph()
        
        # Filter by selected slots
        if selected_slots:
            nodes_to_keep = set()
            for slot in selected_slots:
                slot_node = f"slot_{slot}"
                if slot_node in value_graph:
                    nodes_to_keep.add(slot_node)
                    # Add connected values
                    for neighbor in value_graph.neighbors(slot_node):
                        nodes_to_keep.add(neighbor)
            
            value_graph = value_graph.subgraph(nodes_to_keep)
        
        # Add nodes
        for node, data in value_graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            
            if node_type == 'slot':
                slot_name = data.get('slot', node)
                slot_type = data.get('slot_type', 'unknown')
                
                color = self.colors.get(slot_type, self.colors['slot'])
                size = 25
                shape = 'diamond'
                label = slot_name.replace('_', '-')
                title = f"Slot: {label}"
                
            elif node_type == 'value':
                value = data.get('value', node.replace('value_', ''))
                color = self.colors['value']
                size = 15
                shape = 'dot'
                label = value[:20] + '...' if len(value) > 20 else value
                title = f"Value: {value}"
                
            else:
                continue
                
            net.add_node(node, label=label, color=color, size=size,
                        shape=shape, title=title)
        
        # Add edges
        for source, target, data in value_graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            
            if edge_type == 'accepts':
                color = '#4ECDC4'
                width = 2
            elif edge_type == 'similar':
                color = '#FFA500'
                width = 1
            else:
                color = '#CCCCCC'
                width = 1
                
            net.add_edge(source, target, color=color, width=width)
        
        return net


class DialogGraphVisualizer:
    """Visualize dialog context graphs"""
    
    def __init__(self, dialog_builder):
        """
        Initialize dialog graph visualizer
        
        Args:
            dialog_builder: DialogContextGraph instance
        """
        self.dialog_builder = dialog_builder
        self.colors = {
            'user_turn': '#FF6B6B',      # Red
            'system_turn': '#4ECDC4',    # Teal
            'entity': '#45B7D1',         # Blue
            'sequential': '#888888',     # Gray
            'coreference': '#00FF00',    # Green
            'entity_shared': '#FFA500'   # Orange
        }
    
    def create_dialog_network(self, dialog_context) -> Network:
        """
        Create interactive dialog network
        
        Args:
            dialog_context: DialogContext object
            
        Returns:
            Pyvis Network object
        """
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "hierarchicalRepulsion": {
              "centralGravity": 0.0,
              "springLength": 100,
              "springConstant": 0.01,
              "nodeDistance": 120,
              "damping": 0.09
            }
          },
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed"
            }
          }
        }
        """)
        
        # Add turn nodes
        for i, turn in enumerate(dialog_context.turns):
            node_id = f"turn_{turn.turn_id}"
            color = self.colors['user_turn'] if turn.speaker == 'user' else self.colors['system_turn']
            
            label = f"T{turn.turn_id}"
            title = f"{turn.speaker.upper()}: {turn.utterance[:50]}..."
            
            net.add_node(node_id, label=label, color=color, size=20,
                        title=title, level=turn.turn_id)
        
        # Add entity nodes
        entity_positions = {}
        for entity, turn_ids in dialog_context.entities.items():
            entity_id = f"entity_{entity}"
            color = self.colors['entity']
            
            # Position entity between related turns
            avg_level = sum(turn_ids) / len(turn_ids)
            entity_positions[entity_id] = avg_level
            
            label = entity[:15] + '...' if len(entity) > 15 else entity
            title = f"Entity: {entity} (appears in turns: {turn_ids})"
            
            net.add_node(entity_id, label=label, color=color, size=15,
                        title=title, level=avg_level)
        
        # Add sequential edges (turn to turn)
        for i in range(len(dialog_context.turns) - 1):
            source = f"turn_{dialog_context.turns[i].turn_id}"
            target = f"turn_{dialog_context.turns[i+1].turn_id}"
            
            net.add_edge(source, target, color=self.colors['sequential'],
                        width=2, title="sequential")
        
        # Add coreference edges
        for turn1_id, turn2_id, entity in dialog_context.coreferences:
            source = f"turn_{turn1_id}"
            target = f"turn_{turn2_id}"
            
            net.add_edge(source, target, color=self.colors['coreference'],
                        width=1, title=f"coreference: {entity}", dashes=True)
        
        # Add entity-turn edges
        for turn in dialog_context.turns:
            turn_node = f"turn_{turn.turn_id}"
            for entity in turn.entities:
                entity_node = f"entity_{entity}"
                if entity_node in [n['id'] for n in net.nodes]:
                    net.add_edge(turn_node, entity_node, 
                               color=self.colors['entity_shared'],
                               width=1, title="mentions")
        
        return net
    
    def create_attention_heatmap(self, attention_weights: Dict, turn_labels: List[str]) -> pd.DataFrame:
        """
        Create attention heatmap data
        
        Args:
            attention_weights: Dictionary of attention weights
            turn_labels: Labels for turns
            
        Returns:
            DataFrame for heatmap visualization
        """
        # Convert attention weights to DataFrame
        df = pd.DataFrame(attention_weights, index=turn_labels, columns=turn_labels)
        return df


class GraphDSTVisualizer:
    """Main visualization app for GraphDST"""
    
    def __init__(self, ontology_path: str):
        """
        Initialize GraphDST visualizer
        
        Args:
            ontology_path: Path to ontology.json file
        """
        self.ontology_path = ontology_path
        
        # Initialize builders (these would be imported in real implementation)
        # from src.data.schema_graph import SchemaGraphBuilder
        # from src.data.dialog_graph import DialogContextGraph
        
        # self.schema_builder = SchemaGraphBuilder(ontology_path)
        # self.dialog_builder = DialogContextGraph(ontology_path)
        
        # For now, create placeholder builders
        self.schema_builder = None
        self.dialog_builder = None
        
        # Initialize visualizers
        # self.schema_viz = SchemaGraphVisualizer(self.schema_builder)
        # self.dialog_viz = DialogGraphVisualizer(self.dialog_builder)
    
    def run_app(self):
        """Run Streamlit visualization app"""
        st.set_page_config(page_title="GraphDST Visualization", layout="wide")
        
        st.title("🧠 GraphDST Graph Visualization")
        st.sidebar.title("Navigation")
        
        # Sidebar navigation
        viz_type = st.sidebar.selectbox(
            "Choose Visualization Type",
            ["Schema Graph", "Dialog Context Graph", "Value Relationships", "Attention Analysis"]
        )
        
        if viz_type == "Schema Graph":
            self.render_schema_visualization()
        elif viz_type == "Dialog Context Graph":
            self.render_dialog_visualization()
        elif viz_type == "Value Relationships":
            self.render_value_visualization()
        elif viz_type == "Attention Analysis":
            self.render_attention_visualization()
    
    def render_schema_visualization(self):
        """Render schema graph visualization"""
        st.header("📊 Schema Graph Visualization")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Controls")
            
            # Domain filter
            domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi']
            selected_domains = st.multiselect(
                "Select Domains", 
                domains, 
                default=domains[:3]
            )
            
            # Display options
            show_values = st.checkbox("Show Value Nodes", value=False)
            layout_option = st.selectbox(
                "Layout Algorithm",
                ["Barnes-Hut", "Force Atlas", "Hierarchical"]
            )
            
            # Graph statistics
            if self.schema_builder:
                st.subheader("Graph Statistics")
                info = self.schema_builder.get_slot_info()
                st.metric("Total Slots", info['total_slots'])
                st.metric("Categorical Slots", info['categorical_slots'])
                st.metric("Span Slots", info['span_slots'])
        
        with col2:
            st.subheader("Interactive Schema Graph")
            
            if self.schema_builder:
                # Create and display network
                schema_viz = SchemaGraphVisualizer(self.schema_builder)
                net = schema_viz.create_schema_network(
                    show_values=show_values,
                    filter_domains=selected_domains
                )
                
                # Save to temporary file and display
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                    net.save_graph(tmp.name)
                    with open(tmp.name, 'r') as f:
                        html = f.read()
                    st.components.v1.html(html, height=600)
                    os.unlink(tmp.name)
            else:
                st.warning("Schema builder not initialized. Please check ontology path.")
    
    def render_dialog_visualization(self):
        """Render dialog context graph visualization"""
        st.header("💬 Dialog Context Graph")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Controls")
            
            # Dialog selection
            dialog_file = st.file_uploader(
                "Upload Dialog JSON",
                type=['json'],
                help="Upload a single dialog from MultiWOZ dataset"
            )
            
            if dialog_file:
                dialog_data = json.load(dialog_file)
                
                # Dialog information
                st.subheader("Dialog Information")
                st.text(f"Dialog ID: {dialog_data.get('dialogue_id', 'Unknown')}")
                st.text(f"Domains: {', '.join(dialog_data.get('domains', []))}")
                st.text(f"Turns: {len(dialog_data.get('turns', []))}")
                
                # Visualization options
                show_entities = st.checkbox("Show Entity Nodes", value=True)
                show_coreferences = st.checkbox("Show Coreferences", value=True)
        
        with col2:
            st.subheader("Interactive Dialog Graph")
            
            if dialog_file and self.dialog_builder:
                # Build dialog context
                dialog_context = self.dialog_builder.build_dialog_context(dialog_data)
                
                # Create and display network
                dialog_viz = DialogGraphVisualizer(self.dialog_builder)
                net = dialog_viz.create_dialog_network(dialog_context)
                
                # Display network
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                    net.save_graph(tmp.name)
                    with open(tmp.name, 'r') as f:
                        html = f.read()
                    st.components.v1.html(html, height=500)
                    os.unlink(tmp.name)
                
                # Show dialog turns
                st.subheader("Dialog Turns")
                for turn in dialog_context.turns:
                    with st.expander(f"Turn {turn.turn_id} ({turn.speaker})"):
                        st.text(turn.utterance)
                        if turn.entities:
                            st.text(f"Entities: {', '.join(turn.entities)}")
                        if turn.belief_state:
                            st.json(turn.belief_state)
            else:
                st.info("Upload a dialog JSON file to visualize the dialog graph.")
    
    def render_value_visualization(self):
        """Render value relationship visualization"""
        st.header("🔗 Value Relationships")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Controls")
            
            if self.schema_builder:
                # Slot selection  
                categorical_slots = self.schema_builder.categorical_slots
                selected_slots = st.multiselect(
                    "Select Slots to Visualize",
                    categorical_slots,
                    default=categorical_slots[:5]
                )
                
                # Display value counts
                st.subheader("Value Statistics")
                for slot in selected_slots:
                    if slot in self.schema_builder.ontology:
                        count = len(self.schema_builder.ontology[slot])
                        st.metric(f"{slot}", count)
        
        with col2:
            st.subheader("Slot-Value Network")
            
            if self.schema_builder and selected_slots:
                schema_viz = SchemaGraphVisualizer(self.schema_builder)
                net = schema_viz.create_value_network(selected_slots)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                    net.save_graph(tmp.name)
                    with open(tmp.name, 'r') as f:
                        html = f.read()
                    st.components.v1.html(html, height=500)
                    os.unlink(tmp.name)
            else:
                st.info("Select slots to visualize their value relationships.")
    
    def render_attention_visualization(self):
        """Render attention analysis"""
        st.header("🔍 Attention Analysis")
        
        st.info("This section would display attention weights from trained models.")
        st.text("Features:")
        st.text("- Attention heatmaps across dialog turns")
        st.text("- Graph attention weights visualization") 
        st.text("- Cross-domain attention patterns")
        st.text("- Temporal attention analysis")
        
        # Placeholder for attention visualization
        st.subheader("Attention Heatmap (Sample)")
        
        import numpy as np
        # Create sample attention data
        sample_data = np.random.rand(5, 5)
        sample_labels = [f"Turn {i+1}" for i in range(5)]
        
        df = pd.DataFrame(sample_data, index=sample_labels, columns=sample_labels)
        st.dataframe(df.style.background_gradient(cmap='Blues'))


def create_visualization_app(ontology_path: str = "data/ontology.json"):
    """
    Create and run GraphDST visualization app
    
    Args:
        ontology_path: Path to ontology file
    """
    visualizer = GraphDSTVisualizer(ontology_path)
    visualizer.run_app()


if __name__ == "__main__":
    # This would be run with: streamlit run src/visualization/graph_viz.py
    create_visualization_app()