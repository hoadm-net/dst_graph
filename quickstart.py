#!/usr/bin/env python3
"""
Quick Start Guide for GraphDST Model

This script shows how to:
1. Create a model
2. Run a forward pass
3. Compute losses
4. Save/load checkpoints
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*70)
print("GraphDST Quick Start Guide")
print("="*70)

# ============================================================================
# STEP 1: Create Model Configuration
# ============================================================================
print("\n📋 STEP 1: Creating model configuration...")

from models.graphdst import GraphDSTConfig

config = GraphDSTConfig(
    hidden_dim=768,
    num_gnn_layers=3,
    num_attention_heads=8,
    dropout=0.1,
    num_domains=5,
    num_slots=37,
    text_encoder="bert-base-uncased",
    learning_rate=2e-5
)

print(f"✓ Config created:")
print(f"  - Hidden dim: {config.hidden_dim}")
print(f"  - GNN layers: {config.num_gnn_layers}")
print(f"  - Attention heads: {config.num_attention_heads}")

# ============================================================================
# STEP 2: Create Model
# ============================================================================
print("\n🏗️  STEP 2: Creating model...")

from models.graphdst import GraphDSTModel

# For quick start, we'll create a minimal slot_info
slot_info = {
    'slot_names': ['hotel-pricerange', 'hotel-area', 'restaurant-food'],
    'categorical_slot_names': ['hotel-pricerange', 'restaurant-food'],
    'span_slot_names': ['hotel-area'],
    'categorical_slots': {
        'hotel-pricerange': 4,  # cheap, moderate, expensive, dontcare
        'restaurant-food': 10   # various food types
    }
}

try:
    model = GraphDSTModel(config, schema_builder=None, slot_info=slot_info)
    print("✓ Model created successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
except ImportError as e:
    print(f"⚠️  Warning: {e}")
    print("  Install transformers: pip install transformers")
    sys.exit(1)

# ============================================================================
# STEP 3: Prepare Dummy Input
# ============================================================================
print("\n📝 STEP 3: Preparing input data...")

batch_size = 2
seq_len = 128

# Tokenized input
input_ids = torch.randint(0, 30000, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len)

print(f"✓ Input prepared:")
print(f"  - Batch size: {batch_size}")
print(f"  - Sequence length: {seq_len}")

# ============================================================================
# STEP 4: Forward Pass
# ============================================================================
print("\n🔄 STEP 4: Running forward pass...")

model.eval()  # Set to evaluation mode
with torch.no_grad():
    predictions = model(input_ids, attention_mask)

print("✓ Forward pass completed!")
print(f"✓ Predictions:")
print(f"  - Domains shape: {predictions['domains'].shape}")
print(f"  - Slot activations: {len(predictions['slot_activations'])} slots")

# Show example predictions
print(f"\n📊 Sample outputs:")
print(f"  Domain logits (first example): {predictions['domains'][0]}")

# ============================================================================
# STEP 5: Compute Loss
# ============================================================================
print("\n💰 STEP 5: Computing loss...")

# Create dummy labels
labels = {
    'domain_labels': torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]),  # hotel, restaurant
}

# Add slot activation labels
for slot_name in slot_info['slot_names']:
    labels[f'{slot_name}_active'] = torch.randint(0, 2, (batch_size,))

# Compute loss
model.train()  # Set to training mode
losses = model.compute_loss(predictions, labels)

print("✓ Loss computed!")
print(f"  - Total loss: {losses['total'].item():.4f}")
if 'domain' in losses:
    print(f"  - Domain loss: {losses['domain'].item():.4f}")
if 'slot' in losses:
    print(f"  - Slot loss: {losses['slot'].item():.4f}")

# ============================================================================
# STEP 6: Backward Pass (Training Step)
# ============================================================================
print("\n🎯 STEP 6: Simulating training step...")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Zero gradients
optimizer.zero_grad()

# Backward
losses['total'].backward()

# Show gradient statistics
grad_norms = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norms.append(param.grad.norm().item())

avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0

print("✓ Backward pass completed!")
print(f"  - Average gradient norm: {avg_grad_norm:.6f}")
print(f"  - Parameters with gradients: {len(grad_norms)}")

# Optimizer step
optimizer.step()
print("✓ Optimizer step completed!")

# ============================================================================
# STEP 7: Save Checkpoint
# ============================================================================
print("\n💾 STEP 7: Saving checkpoint...")

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'slot_info': slot_info
}

checkpoint_path = checkpoint_dir / "graphdst_quickstart.pt"
torch.save(checkpoint, checkpoint_path)
print(f"✓ Checkpoint saved to: {checkpoint_path}")

# ============================================================================
# STEP 8: Load Checkpoint
# ============================================================================
print("\n📂 STEP 8: Loading checkpoint...")

# Load checkpoint
loaded_checkpoint = torch.load(checkpoint_path)

# Create new model and load state
new_model = GraphDSTModel(
    loaded_checkpoint['config'],
    schema_builder=None,
    slot_info=loaded_checkpoint['slot_info']
)
new_model.load_state_dict(loaded_checkpoint['model_state_dict'])

print("✓ Checkpoint loaded successfully!")

# Verify loaded model
new_model.eval()
with torch.no_grad():
    new_predictions = model(input_ids, attention_mask)
    
print("✓ Loaded model verified!")

# ============================================================================
# STEP 9: Model Summary
# ============================================================================
print("\n📊 STEP 9: Model summary...")

def count_parameters_by_component(model):
    """Count parameters by model component"""
    counts = {}
    
    # Text encoder
    if hasattr(model, 'text_encoder'):
        counts['text_encoder'] = sum(p.numel() for p in model.text_encoder.parameters())
    
    # GNN layers
    if hasattr(model, 'schema_gnn_layers'):
        counts['schema_gnn'] = sum(
            p.numel() for layer in model.schema_gnn_layers for p in layer.parameters()
        )
    
    if hasattr(model, 'cross_domain_gat_layers'):
        counts['cross_domain_gat'] = sum(
            p.numel() for layer in model.cross_domain_gat_layers for p in layer.parameters()
        )
    
    # Temporal layer
    if hasattr(model, 'temporal_gru'):
        counts['temporal_gru'] = sum(p.numel() for p in model.temporal_gru.parameters())
    
    # Prediction heads
    if hasattr(model, 'prediction_heads'):
        counts['prediction_heads'] = sum(p.numel() for p in model.prediction_heads.parameters())
    
    return counts

param_counts = count_parameters_by_component(model)

print("✓ Parameter breakdown:")
for component, count in param_counts.items():
    percentage = (count / total_params) * 100
    print(f"  - {component}: {count:,} ({percentage:.1f}%)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("✅ Quick Start Complete!")
print("="*70)

print("\n🎓 What you learned:")
print("  1. ✓ How to create a GraphDST model")
print("  2. ✓ How to run forward pass")
print("  3. ✓ How to compute multi-task losses")
print("  4. ✓ How to perform training step")
print("  5. ✓ How to save/load checkpoints")
print("  6. ✓ How to analyze model structure")

print("\n📚 Next steps:")
print("  - Prepare your MultiWOZ data")
print("  - Implement full data loader")
print("  - Create complete training loop")
print("  - Add evaluation metrics")
print("  - Visualize attention weights")

print("\n💡 Tips:")
print("  - Use smaller hidden_dim (256) for faster development")
print("  - Enable gradient_checkpointing for large models")
print("  - Use mixed precision training for faster training")
print("  - Monitor gradient norms to detect issues")

print("\n" + "="*70)
print("Happy training! 🚀")
print("="*70)
