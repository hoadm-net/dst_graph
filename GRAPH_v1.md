# GraphDST v1.0 - Kiến Trúc và Xử Lý Dữ Liệu

## 1. Mô Tả Kiến Trúc Tổng Quan

GraphDST (Graph-based Dialogue State Tracking) là một mô hình sử dụng Graph Neural Networks (GNN) để theo dõi trạng thái hội thoại trong task-oriented dialogue systems. Model được thiết kế đặc biệt cho dataset MultiWOZ với khả năng mô hình hóa quan hệ phức tạp giữa các thành phần trong schema ontology.

### 1.1 Đặc Điểm Chính

- **Multi-level Graph Structure**: Xây dựng đồ thị đa tầng gồm Domain Graph, Schema Graph, và Value Graph
- **Heterogeneous Graph Neural Networks**: Sử dụng GNN cho các loại node khác nhau (domain, slot, value)
- **Temporal Modeling**: Mô hình hóa luồng hội thoại qua thời gian bằng GRU với attention mechanism
- **Multi-task Learning**: Dự đoán đồng thời domain activation, slot activation, và slot values
- **Incremental State Tracking**: Dự đoán belief_state_delta (chỉ những thay đổi ở turn hiện tại)

### 1.2 Thành Phần Chính

```
Input Text (Dialog History + Current Utterance)
    ↓
[Text Encoder] BERT-base-uncased
    ↓
[Schema Graph Processing]
    ├── Domain Graph Layer (5 domains)
    ├── Schema Graph Layer (37 slots)  
    └── Value Graph Layer (categorical values)
    ↓
[Graph Neural Network Layers]
    ├── Schema-aware GCN (3 layers)
    └── Cross-domain GAT (3 layers with 8 heads)
    ↓
[Temporal Modeling]
    └── Bidirectional GRU + Turn Attention
    ↓
[Multi-task Prediction Heads]
    ├── Domain Classification (5 domains)
    ├── Slot Activation (37 slots binary)
    └── Value Prediction
        ├── Categorical (30 slots with fixed vocab)
        └── Span Extraction (7 slots with span-based values)
    ↓
Output: belief_state_delta
```

### 1.3 Workflow Tổng Quan

1. **Input**: Dialog history (tối đa 3 turns trước) + Current user utterance
2. **Text Encoding**: BERT tokenization và encoding (max_length=512)
3. **Schema Graph Construction**: Xây dựng đồ thị từ ontology với 3 levels
4. **Graph Processing**: 3 layers GNN với Schema GCN + Cross-domain GAT
5. **Temporal Processing**: GRU modeling cho dialog flow + turn attention
6. **Multi-task Prediction**: Dự đoán domain → slot → value theo pipeline
7. **Output**: belief_state_delta (incremental changes only)

---

## 2. Kiến Trúc Mô Hình Chi Tiết

### 2.1 Text Encoder Module

**Transformer-based Encoder (BERT)**

```python
Text Encoder: BERT-base-uncased
- Vocabulary size: 30,522
- Hidden size: 768
- Number of layers: 12
- Attention heads: 12
- Total parameters: ~110M
```

**Input Format:**
```
[CLS] [USR] history_turn_1 [SYS] response_1 ... [SEP] current_utterance [SEP]
```

**Output:**
- `hidden_states`: (batch, seq_len, 768) - Token-level representations
- `cls_token`: (batch, 768) - Utterance-level representation

**Feature Projection:**
- Linear layer: 768 → hidden_dim (nếu cần match với GNN dimension)
- Layer normalization + Dropout (0.1)

### 2.2 Schema Graph Module

**Multi-level Heterogeneous Graph**

#### Level 1: Domain Graph (5 nodes)
```
Domains: [hotel, restaurant, attraction, train, taxi]

Edge Types:
- domain-domain: Related domains (e.g., hotel ↔ restaurant)
- domain-slot: Contains relationship (e.g., hotel → hotel-pricerange)
```

#### Level 2: Schema Graph (37 nodes)
```
Slots: 37 slots across 5 domains
- hotel-*: 10 slots (pricerange, type, parking, internet, area, stars, name, ...)
- restaurant-*: 7 slots (food, pricerange, area, name, ...)
- attraction-*: 4 slots (type, area, name, ...)
- train-*: 6 slots (departure, destination, day, time, ...)
- taxi-*: 4 slots (departure, destination, time, ...)

Edge Types:
- slot-slot: Co-occurrence relationship
- slot-value: Accepts relationship
```

#### Level 3: Value Graph (variable nodes)
```
Values: Categorical slot values
- Categorical slots (30): Fixed vocabulary (2-50 values per slot)
- Span slots (7): Extracted from text (open vocabulary)

Edge Types:
- value-value: Similarity/synonym relationship
```

**Graph Structure:**
```python
schema_graph = {
    'domain': {
        'x': domain_embeddings,  # (5, hidden_dim)
        'edge_index': domain_edges  # Domain connections
    },
    'slot': {
        'x': slot_embeddings,  # (37, hidden_dim)
        'edge_index': slot_edges  # Slot connections
    },
    'value': {
        'x': value_embeddings,  # (num_values, hidden_dim)
        'edge_index': value_edges  # Value connections
    },
    'edge_index_dict': {
        ('domain', 'connected', 'domain'): edge_index,
        ('domain', 'contains', 'slot'): edge_index,
        ('slot', 'accepts', 'value'): edge_index,
        ('slot', 'cooccurs', 'slot'): edge_index,
    }
}
```

### 2.3 Graph Neural Network Layers

#### 2.3.1 Schema-aware GCN Layer

**Heterogeneous Graph Convolution**

```python
class SchemaGCNLayer:
    Input: x_dict = {'domain': (5, D), 'slot': (37, D), 'value': (V, D)}
    
    # Domain processing
    domain_out = GCN(domain_features, domain_edges)
    domain_out = LayerNorm(domain_out + domain_features)  # Residual
    
    # Slot processing (aggregate from domains)
    slot_messages = Aggregate(domain_features, domain_slot_edges)
    slot_out = GCN(slot_features + slot_messages)
    slot_out = LayerNorm(slot_out + slot_features)  # Residual
    
    # Value processing (aggregate from slots)
    value_messages = Aggregate(slot_features, slot_value_edges)
    value_out = GCN(value_features + value_messages)
    value_out = LayerNorm(value_out + value_features)  # Residual
    
    Output: Updated x_dict
```

**GCN Message Passing:**
```
h_target = Linear_target(x_target)
h_source = Linear_source(x_source)

messages = Aggregate(h_source, edge_index)
output = h_target + messages
```

#### 2.3.2 Cross-Domain GAT Layer

**Multi-head Graph Attention**

```python
class CrossDomainGATLayer:
    Input: x_dict from SchemaGCNLayer
    Num_heads: 8
    
    # Multi-head attention for each node type
    for node_type in ['domain', 'slot', 'value']:
        Q = W_q(x[node_type])  # (N, 8, 96)
        K = W_k(x[node_type])  # (N, 8, 96)
        V = W_v(x[node_type])  # (N, 8, 96)
        
        # Compute attention along edges
        attn_scores = (Q[dst] * K[src]).sum(-1) / sqrt(96)  # (E, 8)
        attn_weights = softmax(attn_scores, dst)  # (E, 8)
        
        # Aggregate values
        messages = attn_weights * V[src]  # (E, 8, 96)
        output = Aggregate(messages, dst)  # (N, 768)
        output = W_o(output)
        
    Output: Updated x_dict with attention
```

**Stacking:**
```
x_dict → SchemaGCN_1 → CrossGAT_1 → 
         SchemaGCN_2 → CrossGAT_2 → 
         SchemaGCN_3 → CrossGAT_3 → 
         final_x_dict
```

### 2.4 Temporal Modeling Module

**GRU-based Dialog Context Processing**

```python
class TemporalGRULayer:
    Input: dialog_sequence (batch, max_turns, hidden_dim)
    
    # Add positional embeddings
    positions = [0, 1, 2, ..., max_turns-1]
    position_embeds = PositionEmbedding(positions)  # (max_turns, hidden_dim)
    dialog_sequence = dialog_sequence + position_embeds
    
    # GRU processing
    gru_output, hidden = GRU(dialog_sequence)  # (batch, max_turns, hidden_dim)
    gru_output = LayerNorm(gru_output)
    
    # Self-attention over turns
    contextualized, _ = MultiheadAttention(
        query=gru_output,
        key=gru_output, 
        value=gru_output,
        num_heads=8
    )
    
    # Residual connection
    output = gru_output + contextualized
    
    Output: (batch, max_turns, hidden_dim), hidden_state
```

### 2.5 Multi-task Prediction Heads

#### 2.5.1 Domain Classification Head

```python
Input: cls_token (batch, 768)

domain_logits = Linear(768 → 256) → ReLU → Dropout → Linear(256 → 5)

Output: (batch, 5) logits
Loss: Binary Cross-Entropy (multi-label)
```

#### 2.5.2 Slot Activation Heads (37 heads)

```python
For each slot:
    Input: [cls_token, slot_feature] → (batch, 768*2)
    
    slot_logits = Linear(1536 → 512) → ReLU → Dropout → Linear(512 → 2)
    
    Output: (batch, 2) - [inactive, active]
    Loss: Cross-Entropy
```

#### 2.5.3 Value Prediction Heads

**A. Categorical Slots (30 slots)**

```python
For each categorical slot (e.g., hotel-pricerange):
    Input: [cls_token, slot_feature] → (batch, 1536)
    
    value_logits = Linear(1536 → 512) → ReLU → Dropout → 
                   Linear(512 → vocab_size)
    
    Output: (batch, vocab_size) - probability over values
    Loss: Cross-Entropy (only for active slots)
    
Example:
    hotel-pricerange: vocab_size = 4 [cheap, moderate, expensive, dontcare]
    hotel-type: vocab_size = 5 [hotel, guest house, dontcare, ...]
```

**B. Span-based Slots (7 slots)**

```python
Input: token_features (batch, seq_len, 768)

# Expand cls_token to match sequence
utterance_expanded = cls_token.unsqueeze(1).expand(-1, seq_len, -1)
span_input = concat([token_features, utterance_expanded], dim=-1)

# Predict start/end positions
start_logits = Linear(1536 → 512) → ReLU → Linear(512 → 1)  # (batch, seq_len)
end_logits = Linear(1536 → 512) → ReLU → Linear(512 → 1)    # (batch, seq_len)

Output: start_logits, end_logits
Loss: Cross-Entropy for start + Cross-Entropy for end

Example span slots:
    - hotel-name, restaurant-name, attraction-name
    - train-departure, train-destination
    - taxi-departure, taxi-destination
```

### 2.6 Loss Function

**Multi-task Weighted Loss**

```python
total_loss = w1 * domain_loss + w2 * slot_loss + w3 * value_loss

1. Domain Loss (Multi-label BCE):
   L_domain = BCE(domain_logits, domain_labels)

2. Slot Loss (Average over 37 slots):
   L_slot = (1/37) * Σ CrossEntropy(slot_logits_i, slot_active_i)

3. Value Loss (Only for active slots):
   
   A. Categorical Loss:
      For each active categorical slot:
          L_cat_i = CrossEntropy(value_logits_i, value_label_i)
      L_categorical = Average(L_cat_i)
   
   B. Span Loss:
      L_span_start = CrossEntropy(start_logits, start_labels)
      L_span_end = CrossEntropy(end_logits, end_labels)
      L_span = (L_span_start + L_span_end) / 2
   
   L_value = (L_categorical + L_span) / 2

Default weights: w1=1.0, w2=1.0, w3=1.0
```

### 2.7 Model Statistics

```
Total Parameters: ~221M
- Text Encoder (BERT): ~110M
- Feature Projection: ~0.6M
- Schema GCN Layers (3x): ~7M
- Cross-domain GAT Layers (3x): ~9M
- Temporal GRU: ~4.7M
- Prediction Heads: ~90M
  - Domain head: ~0.2M
  - Slot activation heads (37x): ~40M
  - Categorical value heads (30x): ~45M
  - Span heads (2x): ~5M

Model Size: ~886 MB (float32)
Training Memory: ~12-16 GB (batch_size=64)
```

---

## 3. Xử Lý Dữ Liệu Đầu Vào

### 3.1 Data Pipeline Overview

```
Raw MultiWOZ JSON
    ↓
[Data Loading] Load dialogs from train/val/test.json
    ↓
[Turn Extraction] Extract individual turns
    ↓
[History Construction] Build dialog history (max 3 turns)
    ↓
[Text Formatting] Format input text with special tokens
    ↓
[Tokenization] BERT tokenizer (max_length=512)
    ↓
[Label Construction] Build multi-task labels
    ↓
[Batching] Collate into mini-batches
    ↓
Model Input
```

### 3.2 Input Data Format

**Raw MultiWOZ Dialog Structure:**

```json
{
    "dialogue_id": "MUL0001.json",
    "turns": [
        {
            "turn_id": 0,
            "speaker": "user",
            "utterance": "I'm looking for a cheap hotel with free wifi",
            "belief_state": {
                "hotel-pricerange": "cheap",
                "hotel-internet": "yes"
            },
            "belief_state_delta": {
                "hotel-pricerange": "cheap",
                "hotel-internet": "yes"
            }
        },
        {
            "turn_id": 1,
            "speaker": "system",
            "system_response": "There are 10 cheap hotels with wifi..."
        },
        {
            "turn_id": 2,
            "speaker": "user",
            "utterance": "I need it in the north area",
            "belief_state": {
                "hotel-pricerange": "cheap",
                "hotel-internet": "yes",
                "hotel-area": "north"
            },
            "belief_state_delta": {
                "hotel-area": "north"
            }
        }
    ]
}
```

### 3.3 Dialog History Construction

**History Window: 3 turns**

```python
def _format_history(turns: List[Dict], max_history: int = 3) -> str:
    """
    Build dialog history with special tokens
    """
    history_parts = []
    
    for turn in turns[-max_history:]:
        if turn['speaker'] == 'user':
            history_parts.append(f"[USR] {turn['utterance']}")
        else:
            # Truncate long system responses
            response = turn.get('system_response', '')
            if len(response) > 150:
                response = response[:147] + "..."
            history_parts.append(f"[SYS] {response}")
    
    return " ".join(history_parts)
```

**Example:**
```
History: "[USR] I need a hotel [SYS] What area? [USR] In the center"
Current: "I want something cheap"
Input Text: "[USR] I need a hotel [SYS] What area? [USR] In the center [SEP] I want something cheap"
```

### 3.4 Tokenization

**BERT Tokenizer Configuration:**

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize input
encoded = tokenizer(
    text=input_text,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

Output:
- input_ids: (batch, 512) - Token IDs
- attention_mask: (batch, 512) - 1 for real tokens, 0 for padding
- token_type_ids: (batch, 512) - Segment IDs (optional)
```

**Special Tokens:**
```
[CLS] - Start of sequence
[SEP] - Separator between history and current utterance
[USR] - User utterance marker (custom, mapped to [unused1])
[SYS] - System response marker (custom, mapped to [unused2])
[PAD] - Padding token
```

### 3.5 Label Construction

**Multi-task Labels:**

#### 3.5.1 Domain Labels (Multi-label)

```python
# Active domains in belief_state_delta
domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi']
domain_labels = torch.zeros(5)  # Binary vector

if 'hotel-pricerange' in belief_state_delta:
    domain_labels[0] = 1  # hotel active
if 'restaurant-food' in belief_state_delta:
    domain_labels[1] = 1  # restaurant active
...

Output: (5,) tensor [1, 0, 0, 1, 0]
```

#### 3.5.2 Slot Activation Labels (37 labels)

```python
# For each slot, check if it's in belief_state_delta
slots = [all 37 slots]
slot_labels = {}

for slot_name in slots:
    if slot_name in belief_state_delta:
        slot_labels[f'{slot_name}_active'] = 1  # Active
    else:
        slot_labels[f'{slot_name}_active'] = 0  # Inactive

Output: Dict with 37 entries, each (1,) tensor [0] or [1]
```

#### 3.5.3 Value Labels

**A. Categorical Slot Values**

```python
# For each categorical slot, get value index
categorical_slots = ['hotel-pricerange', 'hotel-type', ...]

for slot_name in categorical_slots:
    if slot_name in belief_state_delta:
        value = belief_state_delta[slot_name]
        value_vocab = ontology[slot_name]  # ['cheap', 'moderate', 'expensive']
        value_idx = value_vocab.index(value)
        
        value_labels[f'{slot_name}_value'] = value_idx
    else:
        value_labels[f'{slot_name}_value'] = -1  # Ignore

Example:
    belief_state_delta = {'hotel-pricerange': 'cheap'}
    value_vocab = ['cheap', 'moderate', 'expensive', 'dontcare']
    → value_labels['hotel_pricerange_value'] = 0
```

**B. Span-based Slot Values**

```python
# For span slots, find start/end token positions
span_slots = ['hotel-name', 'train-departure', ...]

for slot_name in span_slots:
    if slot_name in belief_state_delta:
        value = belief_state_delta[slot_name]
        
        # Find value in tokenized text
        start_pos, end_pos = find_span_in_tokens(value, input_ids)
        
        span_labels['span_start'] = start_pos
        span_labels['span_end'] = end_pos
    else:
        span_labels['span_start'] = -1  # Ignore
        span_labels['span_end'] = -1

Example:
    utterance = "I need the Train leaving from Cambridge"
    value = "cambridge"
    tokens = ['i', 'need', 'the', 'train', 'leaving', 'from', 'cambridge']
    → start_pos = 6, end_pos = 6
```

### 3.6 Batch Collation

**Custom Collate Function:**

```python
def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate batch with padding and stacking
    """
    # Stack input tensors
    input_ids = torch.stack([ex['input_ids'] for ex in batch])
    attention_mask = torch.stack([ex['attention_mask'] for ex in batch])
    
    # Stack domain labels
    domain_labels = torch.stack([ex['domain_labels'] for ex in batch])
    
    # Stack slot labels (37 tensors)
    slot_labels = {}
    for slot_name in all_slots:
        slot_labels[f'{slot_name}_active'] = torch.stack([
            ex[f'{slot_name}_active'] for ex in batch
        ])
    
    # Stack value labels
    value_labels = {}
    for slot_name in categorical_slots:
        value_labels[f'{slot_name}_value'] = torch.stack([
            ex[f'{slot_name}_value'] for ex in batch
        ])
    
    # Span labels
    span_labels = {
        'span_start': torch.stack([ex['span_start'] for ex in batch]),
        'span_end': torch.stack([ex['span_end'] for ex in batch])
    }
    
    # Raw data for debugging
    raw_data = [ex['raw_data'] for ex in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': {
            'domain_labels': domain_labels,
            **slot_labels,
            **value_labels,
            **span_labels
        },
        'raw_data': raw_data
    }
```

**Batch Shape:**
```
batch_size = 64

input_ids: (64, 512)
attention_mask: (64, 512)
domain_labels: (64, 5)
slot_labels: 37 tensors of (64, 1) each
value_labels: 30 tensors of (64, 1) each (categorical)
span_start: (64, 1)
span_end: (64, 1)
```

### 3.7 Data Statistics

**MultiWOZ 2.1 Dataset:**

```
Training Set:
- Dialogues: 8,437
- Turns: 56,778 user turns
- Average dialog length: 6.7 turns
- Average utterance length: 11.3 words

Validation Set:
- Dialogues: 1,000
- Turns: 7,374 user turns

Test Set:
- Dialogues: 999
- Turns: 7,368 user turns

Ontology:
- Domains: 5
- Slots: 37
- Total values: ~4,500 unique values
  - Categorical values: ~450
  - Span-based values: ~4,050

Slot Distribution:
- Categorical slots: 30 (81%)
- Span slots: 7 (19%)

Average values per categorical slot: 15
```

### 3.8 Belief State Delta vs Full State

**Key Concept: Incremental Prediction**

```python
# Wrong: Predict full cumulative state
belief_state_full = {
    'hotel-pricerange': 'cheap',
    'hotel-area': 'north',
    'hotel-internet': 'yes'
}

# Correct: Predict only changes in current turn
belief_state_delta = {
    'hotel-area': 'north'  # Only the new slot mentioned in current turn
}
```

**Training Labels:**
- Use `belief_state_delta` as target labels
- Model learns to predict incremental changes only
- Reduces output sparsity and improves convergence

**Evaluation Metrics:**
- **Delta Accuracy**: % of turns with perfect delta prediction
- **Joint Goal Accuracy**: % of dialogs with all deltas correct
- **Slot F1**: Per-slot precision/recall/F1
- **Domain F1**: Per-domain precision/recall/F1

### 3.9 Data Augmentation (Optional)

**Future Enhancements:**

1. **Paraphrasing**: Rephrase utterances while keeping belief state
2. **Slot Value Substitution**: Replace values with synonyms
3. **Turn Shuffling**: Reorder non-dependent turns
4. **Negative Sampling**: Add turns with no belief state changes

---

## Tổng Kết

GraphDST v1.0 là một kiến trúc phức tạp kết hợp nhiều thành phần deep learning hiện đại:

1. **Text Understanding**: BERT encoder cho semantic understanding
2. **Graph Reasoning**: Multi-level GNN cho schema reasoning
3. **Temporal Modeling**: GRU + Attention cho dialog flow
4. **Multi-task Learning**: Parallel prediction cho domain/slot/value

Model được tối ưu hóa cho task DST với:
- Input: Dialog history + Current utterance
- Output: belief_state_delta (incremental changes)
- Training: Multi-task loss với domain/slot/value objectives
- Inference: Sequential prediction pipeline

**Performance Target:**
- Delta Accuracy: >85%
- Joint Goal Accuracy: >55%
- Training time: ~2-3 hours (10 epochs, V100 GPU)
- Inference speed: ~100 examples/second

---

*Document Version: 1.0*  
*Last Updated: November 8, 2025*  
*Author: GraphDST Development Team*
