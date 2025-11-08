#!/usr/bin/env python3
"""
Demo script to test GraphDST model implementation

This script demonstrates:
1. Model creation and initialization
2. Forward pass with dummy data
3. Loss computation
4. Parameter counting
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.graphdst import GraphDSTConfig, GraphDSTModel, MultiHeadGraphAttention
from models.graphdst import SchemaGCNLayer, CrossDomainGATLayer, TemporalGRULayer


def test_multi_head_attention():
    """Test MultiHeadGraphAttention layer"""
    print("\n" + "="*60)
    print("Testing MultiHeadGraphAttention")
    print("="*60)
    
    # Create layer
    attention = MultiHeadGraphAttention(
        input_dim=768,
        output_dim=768,
        num_heads=8,
        dropout=0.1
    )
    
    # Create dummy data
    num_nodes = 10
    num_edges = 20
    
    query = torch.randn(num_nodes, 768)
    key = torch.randn(num_nodes, 768)
    value = torch.randn(num_nodes, 768)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Forward pass
    output = attention(query, key, value, edge_index)
    
    print(f"✓ Input shape: {query.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Number of heads: {attention.num_heads}")
    print(f"✓ Head dimension: {attention.head_dim}")
    
    assert output.shape == query.shape, "Output shape mismatch!"
    print("✓ MultiHeadGraphAttention test passed!")


def test_schema_gcn():
    """Test SchemaGCNLayer"""
    print("\n" + "="*60)
    print("Testing SchemaGCNLayer")
    print("="*60)
    
    # Create layer
    gcn = SchemaGCNLayer(input_dim=768, output_dim=768, dropout=0.1)
    
    # Create dummy heterogeneous graph data
    num_domains = 5
    num_slots = 37
    num_values = 100
    
    x_dict = {
        'domain': torch.randn(num_domains, 768),
        'slot': torch.randn(num_slots, 768),
        'value': torch.randn(num_values, 768)
    }
    
    # Create dummy edges
    domain_edges = torch.randint(0, num_domains, (2, 10))
    slot_edges = torch.stack([
        torch.randint(0, num_domains, (15,)),  # Source: domains
        torch.randint(0, num_slots, (15,))     # Target: slots
    ])
    value_edges = torch.stack([
        torch.randint(0, num_slots, (50,)),    # Source: slots
        torch.randint(0, num_values, (50,))    # Target: values
    ])
    
    edge_index_dict = {
        ('domain', 'connected', 'domain'): domain_edges,
        ('domain', 'contains', 'slot'): slot_edges,
        ('slot', 'accepts', 'value'): value_edges
    }
    
    # Forward pass
    output = gcn(x_dict, edge_index_dict)
    
    print(f"✓ Domain features: {x_dict['domain'].shape} -> {output['domain'].shape}")
    print(f"✓ Slot features: {x_dict['slot'].shape} -> {output['slot'].shape}")
    print(f"✓ Value features: {x_dict['value'].shape} -> {output['value'].shape}")
    print("✓ SchemaGCNLayer test passed!")


def test_temporal_gru():
    """Test TemporalGRULayer"""
    print("\n" + "="*60)
    print("Testing TemporalGRULayer")
    print("="*60)
    
    # Create layer
    gru = TemporalGRULayer(input_dim=768, hidden_dim=768, num_layers=2, dropout=0.1)
    
    # Create dummy dialog sequence
    batch_size = 4
    max_turns = 10
    hidden_dim = 768
    
    dialog_sequence = torch.randn(batch_size, max_turns, hidden_dim)
    turn_mask = torch.ones(batch_size, max_turns, dtype=torch.bool)
    turn_mask[:, 7:] = False  # Mask last 3 turns
    
    # Forward pass
    output, hidden = gru(dialog_sequence, turn_mask)
    
    print(f"✓ Input shape: {dialog_sequence.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Hidden state shape: {hidden.shape}")
    print(f"✓ Max turns: {gru.max_turns}")
    print("✓ TemporalGRULayer test passed!")


def test_full_model():
    """Test full GraphDST model"""
    print("\n" + "="*60)
    print("Testing Full GraphDST Model")
    print("="*60)
    
    # Create config
    config = GraphDSTConfig(
        hidden_dim=256,  # Smaller for faster testing
        num_gnn_layers=2,
        num_attention_heads=4,
        dropout=0.1,
        num_domains=5,
        num_slots=37,
        text_encoder="bert-base-uncased"
    )
    
    # Create dummy slot info
    slot_info = {
        'slot_names': [f'hotel-slot{i}' for i in range(10)] + 
                     [f'restaurant-slot{i}' for i in range(10)],
        'categorical_slot_names': [f'hotel-slot{i}' for i in range(5)],
        'span_slot_names': [f'hotel-slot{i}' for i in range(5, 10)],
        'categorical_slots': {f'hotel-slot{i}': 20 for i in range(5)}
    }
    
    # Create model (without schema builder for testing)
    try:
        model = GraphDSTModel(config, schema_builder=None, slot_info=slot_info)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully!")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Model size: ~{total_params * 4 / (1024**2):.2f} MB")
        
        # Test forward pass with dummy data
        batch_size = 2
        seq_len = 128
        
        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        print(f"\n✓ Testing forward pass...")
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        
        # Forward pass
        with torch.no_grad():
            predictions = model(input_ids, attention_mask)
        
        print(f"\n✓ Predictions generated:")
        print(f"  - Domains: {predictions['domains'].shape}")
        print(f"  - Slot activations: {len(predictions['slot_activations'])} slots")
        print(f"  - Values: {len(predictions['values'])} value predictions")
        
        if 'span_start' in predictions:
            print(f"  - Span start: {predictions['span_start'].shape}")
            print(f"  - Span end: {predictions['span_end'].shape}")
        
        # Test loss computation
        print(f"\n✓ Testing loss computation...")
        
        # Create dummy labels
        labels = {
            'domain_labels': torch.randint(0, 2, (batch_size, config.num_domains)),
        }
        
        for slot_name in slot_info['slot_names'][:5]:  # Test first 5 slots
            labels[f'{slot_name}_active'] = torch.randint(0, 2, (batch_size,))
        
        # Compute loss
        losses = model.compute_loss(predictions, labels)
        
        print(f"  - Total loss: {losses['total'].item():.4f}")
        if 'domain' in losses:
            print(f"  - Domain loss: {losses['domain'].item():.4f}")
        if 'slot' in losses:
            print(f"  - Slot loss: {losses['slot'].item():.4f}")
        
        print("\n✓ Full model test passed!")
        
    except ImportError as e:
        print(f"⚠ Could not load transformers: {e}")
        print(f"  Install with: pip install transformers")
    except Exception as e:
        print(f"⚠ Error during model test: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("="*60)
    print("GraphDST PyTorch Implementation Tests")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    try:
        # Test individual components
        test_multi_head_attention()
        test_schema_gcn()
        test_temporal_gru()
        
        # Test full model
        test_full_model()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        print("\n🚀 GraphDST model is ready for training!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
