#!/usr/bin/env python3
"""
Validation Script for GraphDST Model

This script handles:
- Loading trained model checkpoint
- Validation on dev set
- Computing evaluation metrics
- Generating prediction outputs
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.graphdst import GraphDSTModel, GraphDSTConfig
from train import MultiWOZDataset, setup_logging


# ============================================================================
# Evaluation Metrics
# ============================================================================
class DSTMetrics:
    """Compute DST evaluation metrics"""
    
    def __init__(self, domains: List[str], slots: List[str]):
        self.domains = domains
        self.slots = slots
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_examples = 0
        self.correct_joint_goals = 0
        
        # Per-domain metrics
        self.domain_tp = defaultdict(int)
        self.domain_fp = defaultdict(int)
        self.domain_fn = defaultdict(int)
        
        # Per-slot metrics
        self.slot_tp = defaultdict(int)
        self.slot_fp = defaultdict(int)
        self.slot_fn = defaultdict(int)
        
        # Value accuracy
        self.slot_value_correct = defaultdict(int)
        self.slot_value_total = defaultdict(int)
    
    def update(self, predictions: Dict, labels: Dict, belief_states: List[Dict]):
        """Update metrics with batch predictions"""
        batch_size = predictions['domains'].size(0)
        self.total_examples += batch_size
        
        # Convert predictions to binary
        domain_preds = (torch.sigmoid(predictions['domains']) > 0.5).cpu().numpy()
        domain_labels = labels['domain_labels'].cpu().numpy()
        
        # Domain-level metrics
        for i in range(batch_size):
            for j, domain in enumerate(self.domains):
                pred = domain_preds[i, j]
                true = domain_labels[i, j]
                
                if pred == 1 and true == 1:
                    self.domain_tp[domain] += 1
                elif pred == 1 and true == 0:
                    self.domain_fp[domain] += 1
                elif pred == 0 and true == 1:
                    self.domain_fn[domain] += 1
        
        # Slot-level metrics
        for slot in self.slots:
            if f'{slot}_active' in labels:
                slot_preds = torch.argmax(predictions['slot_activations'][slot], dim=-1).cpu().numpy()
                slot_labels = labels[f'{slot}_active'].cpu().numpy()
                
                for i in range(batch_size):
                    pred = slot_preds[i]
                    true = slot_labels[i]
                    
                    if pred == 1 and true == 1:
                        self.slot_tp[slot] += 1
                    elif pred == 1 and true == 0:
                        self.slot_fp[slot] += 1
                    elif pred == 0 and true == 1:
                        self.slot_fn[slot] += 1
        
        # Joint goal accuracy (simplified - checks if all active slots are correct)
        for i in range(batch_size):
            all_correct = True
            
            for slot in self.slots:
                if f'{slot}_active' in labels:
                    slot_pred = torch.argmax(predictions['slot_activations'][slot][i])
                    slot_true = labels[f'{slot}_active'][i]
                    
                    if slot_pred != slot_true:
                        all_correct = False
                        break
            
            if all_correct:
                self.correct_joint_goals += 1
    
    def compute_metrics(self) -> Dict:
        """Compute final metrics"""
        metrics = {}
        
        # Joint goal accuracy
        metrics['joint_goal_accuracy'] = self.correct_joint_goals / max(self.total_examples, 1)
        
        # Domain F1 scores
        domain_metrics = {}
        for domain in self.domains:
            tp = self.domain_tp[domain]
            fp = self.domain_fp[domain]
            fn = self.domain_fn[domain]
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            
            domain_metrics[domain] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        metrics['domain_metrics'] = domain_metrics
        
        # Overall domain F1
        avg_domain_f1 = sum(m['f1'] for m in domain_metrics.values()) / len(domain_metrics)
        metrics['avg_domain_f1'] = avg_domain_f1
        
        # Slot F1 scores
        slot_metrics = {}
        for slot in self.slots:
            tp = self.slot_tp[slot]
            fp = self.slot_fp[slot]
            fn = self.slot_fn[slot]
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            
            slot_metrics[slot] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        metrics['slot_metrics'] = slot_metrics
        
        # Overall slot F1
        avg_slot_f1 = sum(m['f1'] for m in slot_metrics.values()) / len(slot_metrics)
        metrics['avg_slot_f1'] = avg_slot_f1
        
        return metrics


# ============================================================================
# Validation Function
# ============================================================================
def validate(model: nn.Module, val_loader: DataLoader, device, logger, 
            domains: List[str], slots: List[str]):
    """Run validation"""
    model.eval()
    
    total_loss = 0
    total_domain_loss = 0
    total_slot_loss = 0
    total_value_loss = 0
    
    metrics_calculator = DSTMetrics(domains, slots)
    
    progress_bar = tqdm(val_loader, desc="Validating")
    
    with torch.no_grad():
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
            
            # Update loss metrics
            total_loss += losses['total'].item()
            if 'domain' in losses:
                total_domain_loss += losses['domain'].item()
            if 'slot' in losses:
                total_slot_loss += losses['slot'].item()
            if 'value' in losses:
                total_value_loss += losses['value'].item()
            
            # Update evaluation metrics
            # For simplicity, we'll just pass empty belief_states for now
            belief_states = [{}] * input_ids.size(0)
            metrics_calculator.update(predictions, labels, belief_states)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}"
            })
    
    # Compute final metrics
    num_batches = len(val_loader)
    
    loss_stats = {
        'loss': total_loss / num_batches,
        'domain_loss': total_domain_loss / num_batches,
        'slot_loss': total_slot_loss / num_batches,
        'value_loss': total_value_loss / num_batches
    }
    
    eval_metrics = metrics_calculator.compute_metrics()
    
    # Combine stats
    all_stats = {**loss_stats, **eval_metrics}
    
    return all_stats


# ============================================================================
# Main Validation Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Validate GraphDST Model")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--val_file', type=str, default='val.json',
                       help='Validation file name')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, default='experiments/val_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda/mps)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(str(output_dir / 'logs'))
    logger.info("="*70)
    logger.info("GraphDST Validation")
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
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model config (for simplicity, use default)
    config = GraphDSTConfig(
        hidden_dim=768,
        num_gnn_layers=3,
        num_attention_heads=8,
        dropout=0.1
    )
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {config.text_encoder}")
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
    
    # Load validation data
    logger.info("Loading validation data...")
    val_dataset = MultiWOZDataset(
        data_path=os.path.join(args.data_dir, args.val_file),
        ontology_path=os.path.join(args.data_dir, 'ontology.json'),
        tokenizer=tokenizer,
        max_length=512,
        max_history=3
    )
    
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    logger.info("Creating model...")
    slot_info = {
        'slot_names': val_dataset.slots,
        'categorical_slot_names': val_dataset.categorical_slots,
        'span_slot_names': val_dataset.span_slots,
        'categorical_slots': {slot: len(vocab) for slot, vocab in val_dataset.value_vocabs.items()}
    }
    
    model = GraphDSTModel(config, schema_builder=None, slot_info=slot_info)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info("Model loaded successfully")
    
    # Run validation
    logger.info("\n" + "="*70)
    logger.info("Starting Validation")
    logger.info("="*70 + "\n")
    
    results = validate(
        model, val_loader, device, logger,
        val_dataset.domains, val_dataset.slots
    )
    
    # Log results
    logger.info("\n" + "="*70)
    logger.info("Validation Results")
    logger.info("="*70)
    logger.info(f"\nLoss Statistics:")
    logger.info(f"  Total Loss: {results['loss']:.4f}")
    logger.info(f"  Domain Loss: {results['domain_loss']:.4f}")
    logger.info(f"  Slot Loss: {results['slot_loss']:.4f}")
    logger.info(f"  Value Loss: {results['value_loss']:.4f}")
    
    logger.info(f"\nAccuracy Metrics:")
    logger.info(f"  Joint Goal Accuracy: {results['joint_goal_accuracy']:.4f}")
    logger.info(f"  Average Domain F1: {results['avg_domain_f1']:.4f}")
    logger.info(f"  Average Slot F1: {results['avg_slot_f1']:.4f}")
    
    logger.info(f"\nPer-Domain F1 Scores:")
    for domain, metrics in results['domain_metrics'].items():
        logger.info(f"  {domain:15s}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
    
    # Save results to file
    results_file = output_dir / 'validation_results.json'
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'loss': results['loss'],
            'domain_loss': results['domain_loss'],
            'slot_loss': results['slot_loss'],
            'value_loss': results['value_loss'],
            'joint_goal_accuracy': results['joint_goal_accuracy'],
            'avg_domain_f1': results['avg_domain_f1'],
            'avg_slot_f1': results['avg_slot_f1'],
            'domain_metrics': results['domain_metrics'],
            'slot_metrics': {k: v for k, v in list(results['slot_metrics'].items())[:10]}  # Top 10 slots
        }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
