# 🚀 GraphDST PyTorch Implementation Guide

## ✅ Implementation Complete!

Tất cả các PyTorch operations cho GNN layers đã được implement đầy đủ!

## 📦 Installation

### 1. Install PyTorch (nếu chưa có)

```bash
# For CPU only
pip install torch torchvision torchaudio

# For CUDA 11.8 (recommended if you have GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install PyTorch Geometric

```bash
# Install PyTorch Geometric and dependencies
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**Note**: Thay `cu118` bằng version CUDA của bạn (hoặc `cpu` nếu không có GPU)

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

## 🏗️ Implemented Components

### ✓ Core GNN Layers

1. **MultiHeadGraphAttention** (`class MultiHeadGraphAttention(nn.Module)`)
   - Multi-head attention mechanism for graphs
   - Edge-based attention computation
   - Supports optional edge features
   - Softmax normalization per destination node
   - Full PyTorch implementation with autograd support

2. **SchemaGCNLayer** (`class SchemaGCNLayer(nn.Module)`)
   - Heterogeneous graph convolution
   - Separate processing for domain/slot/value nodes
   - Layer normalization and residual connections
   - Cross-type message passing

3. **HeteroGraphConv** (`class HeteroGraphConv(nn.Module)`)
   - General heterogeneous convolution operation
   - Degree normalization
   - Message aggregation with `index_add_`

4. **CrossDomainGATLayer** (`class CrossDomainGATLayer(nn.Module)`)
   - Cross-domain knowledge sharing
   - Slot similarity attention
   - Domain connection attention
   - Residual connections for stability

5. **TemporalGRULayer** (`class TemporalGRULayer(nn.Module)`)
   - GRU-based temporal modeling
   - Positional embeddings for dialog turns
   - Multi-head self-attention over turns
   - Turn masking support

### ✓ Prediction Heads

6. **MultiTaskHeads** (`class MultiTaskHeads(nn.Module)`)
   - Domain classification (multi-label)
   - Slot activation (binary per slot)
   - Categorical value prediction
   - Span extraction (start/end positions)
   - Dynamic feature combination

### ✓ Main Model

7. **GraphDSTModel** (`class GraphDSTModel(nn.Module)`)
   - Complete end-to-end model
   - BERT text encoder integration
   - Multi-layer GNN processing
   - Schema graph integration
   - Full forward pass implementation
   - Comprehensive loss computation

## 🎯 Key Features

### Multi-Task Learning
```python
losses = model.compute_loss(predictions, labels)
# Returns:
# {
#     'domain': domain_classification_loss,
#     'slot': slot_activation_loss,
#     'value': value_prediction_loss,
#     'total': combined_loss
# }
```

### Flexible Architecture
```python
config = GraphDSTConfig(
    hidden_dim=768,
    num_gnn_layers=3,
    num_attention_heads=8,
    dropout=0.1,
    text_encoder="bert-base-uncased"
)
```

### Heterogeneous Graph Support
```python
x_dict = {
    'domain': domain_features,  # (num_domains, hidden_dim)
    'slot': slot_features,      # (num_slots, hidden_dim)
    'value': value_features     # (num_values, hidden_dim)
}

edge_index_dict = {
    ('domain', 'connected', 'domain'): domain_edges,
    ('domain', 'contains', 'slot'): domain_slot_edges,
    ('slot', 'accepts', 'value'): slot_value_edges
}
```

## 🧪 Testing

### Run Component Tests

```bash
# Make script executable
chmod +x test_model.py

# Run tests
python test_model.py
```

Tests include:
- ✓ MultiHeadGraphAttention forward pass
- ✓ SchemaGCNLayer heterogeneous convolution
- ✓ TemporalGRULayer with masking
- ✓ Full model forward pass
- ✓ Loss computation
- ✓ Parameter counting

### Expected Output

```
==============================================================
GraphDST PyTorch Implementation Tests
==============================================================

==============================================================
Testing MultiHeadGraphAttention
==============================================================
✓ Input shape: torch.Size([10, 768])
✓ Output shape: torch.Size([10, 768])
✓ Number of heads: 8
✓ Head dimension: 96
✓ MultiHeadGraphAttention test passed!

... (more tests)

All tests passed! ✓
==============================================================
🚀 GraphDST model is ready for training!
```

## 📊 Model Usage

### Basic Usage

```python
from src.models.graphdst import create_graphdst_model, GraphDSTConfig

# Create model
config = GraphDSTConfig(hidden_dim=768, num_gnn_layers=3)
model = create_graphdst_model("data/ontology.json", config)

# Forward pass
predictions = model(input_ids, attention_mask)

# Compute loss
losses = model.compute_loss(predictions, labels)
```

### Training Loop Example

```python
import torch
from torch.optim import AdamW

# Setup
optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()

# Training
for batch in train_loader:
    optimizer.zero_grad()
    
    # Forward
    predictions = model(batch['input_ids'], batch['attention_mask'])
    
    # Loss
    losses = model.compute_loss(predictions, batch['labels'])
    
    # Backward
    losses['total'].backward()
    optimizer.step()
```

## 🔧 Technical Details

### Graph Attention Implementation

```python
# Compute attention scores along edges
src, dst = edge_index[0], edge_index[1]
Q_dst = Q[dst]  # Queries at destination nodes
K_src = K[src]  # Keys at source nodes

# Scaled dot-product attention
attn_scores = (Q_dst * K_src).sum(dim=-1) / sqrt(head_dim)

# Softmax per destination node
attn_weights = softmax(attn_scores, dst, num_nodes=num_nodes)

# Aggregate messages
V_src = V[src]
messages = V_src * attn_weights.unsqueeze(-1)
output.index_add_(0, dst, messages)
```

### Heterogeneous Message Passing

```python
# Different transformations for different node types
x_target_transformed = lin_target(x_target)
x_source_transformed = lin_source(x_source)

# Message passing from source to target
messages = x_source_transformed[edge_index[0]]
output.index_add_(0, edge_index[1], messages)

# Degree normalization
degree = compute_degree(edge_index[1])
output = output / degree
```

## 📈 Performance Optimization

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    predictions = model(input_ids, attention_mask)
    losses = model.compute_loss(predictions, labels)

scaler.scale(losses['total']).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Checkpointing

```python
model.text_encoder.gradient_checkpointing_enable()
```

## 🐛 Common Issues

### Issue 1: PyTorch Geometric Not Found

```bash
# Solution: Install correct version for your PyTorch
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Issue 2: CUDA Out of Memory

```python
# Solution: Reduce batch size or use gradient accumulation
config.batch_size = 4
config.gradient_accumulation_steps = 4
```

### Issue 3: Transformers Import Error

```bash
# Solution: Install transformers
pip install transformers
```

## 📚 Next Steps

1. **Data Preparation**: Implement full data loading pipeline
2. **Training Script**: Create complete training loop
3. **Evaluation**: Add comprehensive metrics
4. **Visualization**: Integrate attention visualization
5. **Optimization**: Profile and optimize bottlenecks

## 🎓 Key Implementation Details

### Memory-Efficient Operations
- ✓ Used `index_add_` for scatter operations
- ✓ Implemented proper masking to avoid unnecessary computation
- ✓ Reused buffers where possible

### Numerical Stability
- ✓ Scaled attention scores by sqrt(head_dim)
- ✓ Layer normalization after each GNN layer
- ✓ Residual connections throughout

### Gradient Flow
- ✓ All operations differentiable
- ✓ Proper backprop through heterogeneous graphs
- ✓ Gradient checkpointing support

## 🌟 Implementation Highlights

1. **Full PyTorch nn.Module**: All components are proper PyTorch modules
2. **Automatic Differentiation**: All operations support autograd
3. **Efficient Graph Operations**: Using PyTorch Geometric utilities
4. **Multi-GPU Support**: Ready for DataParallel/DistributedDataParallel
5. **Production Ready**: Includes all necessary safeguards and checks

---

**Status**: ✅ COMPLETE - Ready for training and experimentation!

**Contact**: For questions or issues, check the main README or open an issue.
