#!/usr/bin/env python3
"""
Training Script for GraphDST Model

This script handles:
- Data loading and preprocessing
- Model initialization
- Training loop with validation
- Checkpointing and logging
- Mixed precision training support
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.graphdst import GraphDSTModel, GraphDSTConfig, create_graphdst_model
from data.schema_graph import SchemaGraphBuilder


# ============================================================================
# Setup Logging
# ============================================================================
def setup_logging(log_dir: str, log_level: str = "INFO"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# Dataset Class
# ============================================================================
class MultiWOZDataset(Dataset):
    """Dataset class for MultiWOZ data"""
    
    def __init__(self, data_path: str, ontology_path: str, tokenizer, 
                 max_length: int = 512, max_history: int = 3):
        """
        Initialize dataset
        
        Args:
            data_path: Path to train/val JSON file
            ontology_path: Path to ontology.json
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            max_history: Maximum number of history turns to include
        """
        self.data_path = data_path
        self.ontology_path = ontology_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_history = max_history
        
        # Load data
        with open(data_path, 'r') as f:
            self.dialogs = json.load(f)
        
        # Load ontology
        with open(ontology_path, 'r') as f:
            self.ontology = json.load(f)
        
        # Build examples (one per turn)
        self.examples = self._build_examples()
        
        # Domain and slot mappings
        self.domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi']
        self.domain2id = {d: i for i, d in enumerate(self.domains)}
        
        self.slots = list(self.ontology.keys())
        self.slot2id = {s: i for i, s in enumerate(self.slots)}
        
        # Categorize slots
        self.categorical_slots, self.span_slots = self._categorize_slots()
        
        # Build value vocabularies for categorical slots
        self.value_vocabs = self._build_value_vocabs()
    
    def _build_examples(self) -> List[Dict]:
        """Build examples from dialogs"""
        examples = []
        
        for dialog in self.dialogs:
            dialog_id = dialog['dialogue_id']
            turns = dialog['turns']
            
            # Process each user turn
            for i, turn in enumerate(turns):
                if turn['speaker'] != 'user':
                    continue
                
                # Build dialog history
                history_turns = turns[max(0, i - self.max_history):i]
                history_text = self._format_history(history_turns)
                
                # Current utterance
                current_utterance = turn['utterance']
                
                # Full input text
                input_text = f"{history_text} [SEP] {current_utterance}"
                
                # Use belief_state_delta for training (changes in this turn only)
                belief_state_delta = turn.get('belief_state_delta', {})
                belief_state_full = turn.get('belief_state', {})
                
                # Create example
                example = {
                    'dialogue_id': dialog_id,
                    'turn_id': turn['turn_id'],
                    'input_text': input_text,
                    'current_utterance': current_utterance,
                    'belief_state': belief_state_delta,  # Use delta for labels
                    'belief_state_full': belief_state_full  # Keep full for reference
                }
                
                examples.append(example)
        
        return examples
    
    def _format_history(self, turns: List[Dict]) -> str:
        """Format dialog history"""
        history_parts = []
        for turn in turns:
            if turn['speaker'] == 'user':
                history_parts.append(f"[USR] {turn['utterance']}")
            else:
                response = turn.get('system_response', '')
                if len(response) > 150:
                    response = response[:147] + "..."
                history_parts.append(f"[SYS] {response}")
        
        return " ".join(history_parts)
    
    def _categorize_slots(self, threshold: int = 50) -> Tuple[List[str], List[str]]:
        """Categorize slots into categorical vs span"""
        categorical = []
        span = []
        
        for slot, values in self.ontology.items():
            if len(values) <= threshold:
                categorical.append(slot)
            else:
                span.append(slot)
        
        return categorical, span
    
    def _build_value_vocabs(self) -> Dict[str, Dict[str, int]]:
        """Build value vocabularies for categorical slots"""
        vocabs = {}
        
        for slot in self.categorical_slots:
            if slot in self.ontology:
                values = self.ontology[slot]
                vocabs[slot] = {v: i for i, v in enumerate(values)}
        
        return vocabs
    
    def _create_labels(self, belief_state: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Create labels from belief state"""
        labels = {}
        
        # Domain labels (multi-label)
        domain_labels = [0] * len(self.domains)
        for slot, value in belief_state.items():
            if value and value not in ['none', '']:
                domain = slot.split('-')[0]
                if domain in self.domain2id:
                    domain_labels[self.domain2id[domain]] = 1
        
        labels['domain_labels'] = torch.tensor(domain_labels, dtype=torch.float)
        
        # Slot activation labels
        for slot in self.slots:
            is_active = 1 if slot in belief_state and belief_state[slot] not in ['none', ''] else 0
            labels[f'{slot}_active'] = torch.tensor(is_active, dtype=torch.long)
        
        # Value labels for categorical slots
        for slot in self.categorical_slots:
            if slot in belief_state and belief_state[slot] in self.value_vocabs.get(slot, {}):
                value_idx = self.value_vocabs[slot][belief_state[slot]]
                labels[f'{slot}_value'] = torch.tensor(value_idx, dtype=torch.long)
        
        return labels
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        encoding = self.tokenizer(
            example['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels
        labels = self._create_labels(example['belief_state'])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'dialogue_id': example['dialogue_id'],
            'turn_id': example['turn_id'],
            'current_utterance': example['current_utterance'],
            'input_text': example['input_text'],
            'belief_state_delta': example['belief_state'],  # This is delta
            'belief_state_full': example['belief_state_full']
        }


def collate_fn(batch):
    """Custom collate function to handle varying label keys"""
    # Collate input_ids and attention_mask normally
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Collect all possible label keys
    all_label_keys = set()
    for item in batch:
        all_label_keys.update(item['labels'].keys())
    
    # Collate labels - use None for missing keys
    labels = {}
    for key in all_label_keys:
        # Get all values for this key (None if missing)
        values = []
        for item in batch:
            if key in item['labels']:
                values.append(item['labels'][key])
            else:
                # Create a dummy tensor with -100 (ignore index)
                if 'value' in key:
                    values.append(torch.tensor(-100, dtype=torch.long))
                elif 'active' in key:
                    values.append(torch.tensor(-100, dtype=torch.long))
                else:  # domain_labels
                    values.append(torch.zeros_like(batch[0]['labels']['domain_labels']))
        
        if values:
            labels[key] = torch.stack(values)
    
    # Collect metadata and raw data for debugging
    dialogue_ids = [item['dialogue_id'] for item in batch]
    turn_ids = [item['turn_id'] for item in batch]
    
    # Store raw data for validation/debugging
    raw_data = []
    for item in batch:
        raw_data.append({
            'dialogue_id': item['dialogue_id'],
            'turn_id': item['turn_id'],
            'current_utterance': item.get('current_utterance', ''),
            'input_text': item.get('input_text', ''),
            'belief_state_delta': item.get('belief_state', {}),  # This is actually delta
            'belief_state_full': item.get('belief_state_full', {})
        })
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'dialogue_id': dialogue_ids,
        'turn_id': turn_ids,
        'raw_data': raw_data
    }


# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer, 
                scheduler, device, logger, epoch: int):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_domain_loss = 0
    total_slot_loss = 0
    total_value_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Move labels to device
        labels = {}
        for key, value in batch['labels'].items():
            if isinstance(value, torch.Tensor):
                labels[key] = value.to(device)
        
        # Forward pass
        predictions = model(input_ids, attention_mask)
        
        # Compute loss
        losses = model.compute_loss(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += losses['total'].item()
        if 'domain' in losses:
            total_domain_loss += losses['domain'].item()
        if 'slot' in losses:
            total_slot_loss += losses['slot'].item()
        if 'value' in losses:
            total_value_loss += losses['value'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Log every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(f"Epoch {epoch+1} Batch {batch_idx+1}: Loss = {avg_loss:.4f}")
    
    # Epoch statistics
    num_batches = len(train_loader)
    stats = {
        'loss': total_loss / num_batches,
        'domain_loss': total_domain_loss / num_batches,
        'slot_loss': total_slot_loss / num_batches,
        'value_loss': total_value_loss / num_batches
    }
    
    return stats


def save_checkpoint(model, optimizer, scheduler, epoch, stats, save_path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'stats': stats
    }
    
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to {save_path}")


# ============================================================================
# Main Training Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train GraphDST Model")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/train_run',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda/mps)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(str(output_dir / 'logs'))
    logger.info("="*70)
    logger.info("GraphDST Training")
    logger.info("="*70)
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load config
    config = GraphDSTConfig(
        hidden_dim=768,
        num_gnn_layers=3,
        num_attention_heads=8,
        dropout=0.1,
        learning_rate=args.learning_rate
    )
    
    logger.info(f"Model config: {config}")
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {config.text_encoder}")
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
    
    # Load data
    logger.info("Loading training data...")
    train_dataset = MultiWOZDataset(
        data_path=os.path.join(args.data_dir, 'train.json'),
        ontology_path=os.path.join(args.data_dir, 'ontology.json'),
        tokenizer=tokenizer,
        max_length=512,
        max_history=3
    )
    
    logger.info(f"Training examples: {len(train_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for macOS compatibility
        collate_fn=collate_fn
    )
    
    # Create model
    logger.info("Creating model...")
    slot_info = {
        'slot_names': train_dataset.slots,
        'categorical_slot_names': train_dataset.categorical_slots,
        'span_slot_names': train_dataset.span_slots,
        'categorical_slots': {slot: len(vocab) for slot, vocab in train_dataset.value_vocabs.items()}
    }
    
    model = GraphDSTModel(config, schema_builder=None, slot_info=slot_info)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    
    # Training loop
    logger.info("\n" + "="*70)
    logger.info("Starting Training")
    logger.info("="*70)
    
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_stats = train_epoch(
            model, train_loader, optimizer, scheduler, device, logger, epoch
        )
        
        # Log statistics
        logger.info(f"\nEpoch {epoch+1} Statistics:")
        logger.info(f"  Average Loss: {train_stats['loss']:.4f}")
        logger.info(f"  Domain Loss: {train_stats['domain_loss']:.4f}")
        logger.info(f"  Slot Loss: {train_stats['slot_loss']:.4f}")
        logger.info(f"  Value Loss: {train_stats['value_loss']:.4f}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
        save_checkpoint(model, optimizer, scheduler, epoch, train_stats, checkpoint_path)
        
        # Save best model
        if train_stats['loss'] < best_loss:
            best_loss = train_stats['loss']
            best_model_path = output_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, scheduler, epoch, train_stats, best_model_path)
            logger.info(f"New best model saved! Loss: {best_loss:.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("Training Complete!")
    logger.info("="*70)
    logger.info(f"Best Loss: {best_loss:.4f}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
