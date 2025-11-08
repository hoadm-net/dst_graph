# 🚀 Training and Validation Guide

## ✅ Hoàn Thành Implementation

Đã implement đầy đủ:
1. ✅ **Model PyTorch** - Tất cả GNN layers với PyTorch operations
2. ✅ **Training Script** - Complete training pipeline với checkpointing
3. ✅ **Validation Script** - Evaluation metrics và prediction analysis

---

## 📋 Prerequisites

### 1. Kiểm Tra Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Check dependencies
python3 check_dependencies.py
```

Nếu thiếu packages, cài đặt:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric transformers pyyaml tqdm
```

### 2. Kiểm Tra Data

Đảm bảo có các files:
```
data/
├── train.json          # Training data
├── val.json            # Validation data  
├── test.json           # Test data
└── ontology.json       # Slot-value ontology
```

---

## 🎯 Training

### Basic Training

```bash
# Train with default settings
python3 train.py \
    --data_dir data \
    --output_dir experiments/run_1 \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-5
```

### Training với Custom Config

```bash
python3 train.py \
    --data_dir data \
    --output_dir experiments/large_model \
    --num_epochs 20 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --device auto
```

### Training Options

- `--data_dir`: Thư mục chứa data (default: `data`)
- `--output_dir`: Thư mục output cho checkpoints (default: `experiments/train_run`)
- `--num_epochs`: Số epochs (default: `10`)
- `--batch_size`: Batch size (default: `16`)
- `--learning_rate`: Learning rate (default: `2e-5`)
- `--device`: Device để train (`auto`/`cpu`/`cuda`/`mps`, default: `auto`)

### Training Output

```
experiments/run_1/
├── logs/
│   └── train.log                # Training logs
├── checkpoint_epoch_1.pt        # Checkpoint mỗi epoch
├── checkpoint_epoch_2.pt
├── ...
└── best_model.pt                # Best model based on loss
```

---

## 📊 Validation

### Basic Validation

```bash
# Validate a trained model
python3 validate.py \
    --checkpoint experiments/run_1/best_model.pt \
    --data_dir data \
    --val_file val.json \
    --batch_size 16 \
    --output_dir experiments/run_1/val_results
```

### Validation Options

- `--checkpoint`: Path to model checkpoint (required)
- `--data_dir`: Data directory (default: `data`)
- `--val_file`: Validation file name (default: `val.json`)
- `--batch_size`: Batch size (default: `16`)
- `--output_dir`: Output directory (default: `experiments/val_results`)
- `--device`: Device (`auto`/`cpu`/`cuda`/`mps`, default: `auto`)

### Validation Output

```
experiments/run_1/val_results/
├── logs/
│   └── validation.log           # Validation logs
└── validation_results.json      # Metrics in JSON format
```

### Metrics Computed

1. **Loss Statistics:**
   - Total loss
   - Domain loss
   - Slot loss
   - Value loss

2. **Accuracy Metrics:**
   - Joint Goal Accuracy (all slots correct)
   - Average Domain F1
   - Average Slot F1

3. **Per-Domain Metrics:**
   - Precision, Recall, F1 for each domain

4. **Per-Slot Metrics:**
   - Precision, Recall, F1 for each slot

---

## 📈 Training Tips

### 1. Start Small

Để test pipeline, train với data nhỏ:
```bash
# Chỉnh sửa train.py để limit số examples
# Hoặc train với 1-2 epochs
python3 train.py --num_epochs 2 --batch_size 4
```

### 2. Monitor Training

```bash
# Watch training logs real-time
tail -f experiments/run_1/logs/train.log
```

### 3. Adjust Batch Size

Nếu out of memory:
```bash
# Giảm batch size
python3 train.py --batch_size 4

# Hoặc chỉnh config để giảm model size
```

### 4. Learning Rate

- Start với `2e-5` (BERT default)
- Nếu loss không giảm: thử `5e-5`
- Nếu unstable: thử `1e-5`

### 5. Device Selection

```bash
# Tự động chọn best device
python3 train.py --device auto

# Force CPU (for debugging)
python3 train.py --device cpu

# Use MPS on macOS (if supported)
python3 train.py --device mps
```

---

## 🔍 Quick Test

### Test Training Pipeline (1 epoch, small batch)

```bash
python3 train.py \
    --num_epochs 1 \
    --batch_size 2 \
    --output_dir experiments/test_run
```

### Test Validation Pipeline

```bash
python3 validate.py \
    --checkpoint experiments/test_run/checkpoint_epoch_1.pt \
    --batch_size 2 \
    --output_dir experiments/test_val
```

---

## 🐛 Troubleshooting

### Issue 1: Out of Memory

**Solution:**
```bash
# Giảm batch size
python3 train.py --batch_size 4

# Hoặc giảm model size trong config
# hidden_dim: 768 -> 256
# num_gnn_layers: 3 -> 2
```

### Issue 2: Slow Training

**Solution:**
```bash
# Sử dụng GPU/MPS nếu có
python3 train.py --device mps  # macOS

# Hoặc giảm sequence length
# Chỉnh max_length trong MultiWOZDataset
```

### Issue 3: Model Not Learning

**Possible causes:**
- Learning rate quá cao/thấp
- Batch size không phù hợp
- Data preprocessing issues

**Debug:**
```bash
# Check một batch của data
python3 -c "
from train import MultiWOZDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = MultiWOZDataset('data/train.json', 'data/ontology.json', tokenizer)
print('Dataset size:', len(dataset))
print('Sample:', dataset[0])
"
```

### Issue 4: Import Errors

**Solution:**
```bash
# Make sure running from project root
cd /Users/hoadinh/Desktop/DST/dst_graph

# Activate venv
source venv/bin/activate

# Check dependencies
python3 check_dependencies.py
```

---

## 📊 Expected Performance

### Training Time (rough estimates)

**Full MultiWOZ dataset (~57K training examples):**
- CPU: ~12-24 hours per epoch
- GPU (V100): ~2-4 hours per epoch
- MPS (M1 Mac): ~6-8 hours per epoch

**Batch size 16, 10 epochs:**
- CPU: ~5-10 days
- GPU: ~1-2 days

### Memory Usage

- **Model size:** ~440 MB (base config)
- **Training memory:**
  - Batch size 16: ~2-4 GB
  - Batch size 8: ~1-2 GB
  - Batch size 4: ~0.5-1 GB

### Target Metrics (after full training)

- Joint Goal Accuracy: 45-55%
- Domain F1: 90-95%
- Slot F1: 95-98%

---

## 💡 Advanced Usage

### 1. Resume Training from Checkpoint

```bash
# Modify train.py to add --resume argument
# Or manually load checkpoint in code
```

### 2. Custom Model Config

Create `my_config.yaml`:
```yaml
model:
  hidden_dim: 512
  num_gnn_layers: 4
  num_attention_heads: 8
  dropout: 0.2

training:
  num_epochs: 15
  batch_size: 8
  learning_rate: 1e-5
```

### 3. Distributed Training

```bash
# For multi-GPU training, modify train.py to use:
# torch.nn.DataParallel or DistributedDataParallel
```

---

## 📝 Example Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Check setup
python3 check_dependencies.py

# 3. Test model
python3 test_model.py

# 4. Quick training test (1 epoch)
python3 train.py --num_epochs 1 --batch_size 4 --output_dir experiments/quick_test

# 5. Validate test model
python3 validate.py --checkpoint experiments/quick_test/checkpoint_epoch_1.pt --output_dir experiments/quick_val

# 6. If all good, full training
python3 train.py --num_epochs 10 --batch_size 16 --output_dir experiments/full_run

# 7. Validate best model
python3 validate.py --checkpoint experiments/full_run/best_model.pt --output_dir experiments/full_val

# 8. Check results
cat experiments/full_val/validation_results.json
```

---

## 🎓 Next Steps

1. **Tune Hyperparameters:** Experiment với learning rate, batch size, model size
2. **Add Features:** Implement graph visualization, attention analysis
3. **Improve Data:** Add data augmentation, better preprocessing
4. **Optimize:** Profile code, add mixed precision training
5. **Deploy:** Create inference API, web demo

---

## 📚 Files Summary

- `train.py` - Training script với data loading và training loop
- `validate.py` - Validation script với metrics computation
- `test_model.py` - Unit tests cho model components
- `check_dependencies.py` - Dependency checker
- `src/models/graphdst.py` - Model implementation
- `src/data/` - Data processing modules (schema_graph, dialog_graph, data_loader)

---

**Happy Training! 🚀**

For questions or issues, check the logs and adjust parameters accordingly.
