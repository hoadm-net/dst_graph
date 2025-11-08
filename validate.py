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
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.graphdst import GraphDSTModel, GraphDSTConfig
from train import MultiWOZDataset, setup_logging, collate_fn


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
# Delta Accuracy Computation
# ============================================================================
def compute_delta_accuracy(predictions):
    """Compute delta-based accuracy metrics"""
    
    total_turns = len(predictions)
    perfect_deltas = 0
    
    # Slot-level metrics
    slot_correct = defaultdict(int)
    slot_total = defaultdict(int)
    slot_fp = defaultdict(int)
    slot_fn = defaultdict(int)
    
    for ex in predictions:
        true_delta = ex.get('true_belief_state_delta', {})
        pred_delta = ex.get('predicted_belief_state_delta', {})
        
        # Check if delta is perfect
        is_perfect = (set(true_delta.keys()) == set(pred_delta.keys()) and
                     all(true_delta.get(k) == pred_delta.get(k) for k in true_delta.keys()))
        
        if is_perfect:
            perfect_deltas += 1
        
        # All slots mentioned
        all_slots = set(list(true_delta.keys()) + list(pred_delta.keys()))
        
        for slot in all_slots:
            true_val = true_delta.get(slot)
            pred_val = pred_delta.get(slot)
            
            if true_val is not None:
                slot_total[slot] += 1
                
                if pred_val == true_val:
                    slot_correct[slot] += 1
                elif pred_val is None:
                    slot_fn[slot] += 1
            else:
                slot_fp[slot] += 1
    
    return {
        'total_turns': total_turns,
        'perfect_deltas': perfect_deltas,
        'delta_accuracy': perfect_deltas / max(total_turns, 1),
        'slot_correct': dict(slot_correct),
        'slot_total': dict(slot_total),
        'slot_fp': dict(slot_fp),
        'slot_fn': dict(slot_fn)
    }


# ============================================================================
# Prediction Extraction for Debugging
# ============================================================================
def extract_predictions(predictions: Dict, labels: Dict, batch: Dict, 
                       domains: List[str], slots: List[str], device,
                       tokenizer=None, val_dataset=None, batch_idx: int = 0) -> List[Dict]:
    """
    Extract predictions in human-readable format for debugging
    
    Returns:
        List of dictionaries containing predictions and labels for each example
    """
    batch_size = predictions['domains'].size(0)
    batch_predictions = []
    
    # Get input text
    input_ids = batch['input_ids'].cpu()
    
    for i in range(batch_size):
        # Decode input text
        input_text = ""
        if tokenizer is not None:
            tokens = input_ids[i]
            input_text = tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Get raw data if available
        raw_data = batch.get('raw_data', [{}] * batch_size)[i]
        
        example = {
            'example_id': batch_idx * batch_size + i,
            'dialogue_id': raw_data.get('dialogue_id', 'unknown'),
            'turn_id': raw_data.get('turn_id', -1),
            'input_text': input_text,
            'current_utterance': raw_data.get('current_utterance', ''),
            'belief_state_delta': raw_data.get('belief_state_delta', {}),  # Target: delta only
            'belief_state_full': raw_data.get('belief_state_full', {}),    # Full state for reference
            'predictions': {},
            'labels': {},
            'correct': {}
        }
        
        # True belief state DELTA from labels (what changed in this turn)
        true_belief_delta = {}
        predicted_belief_delta = {}
        
        # Domain predictions
        domain_probs = torch.sigmoid(predictions['domains'][i]).cpu().numpy()
        domain_preds = (domain_probs > 0.5).astype(int)
        domain_labels = labels['domain_labels'][i].cpu().numpy()
        
        example['predictions']['domains'] = {
            domain: {
                'predicted': bool(domain_preds[j]),
                'probability': float(domain_probs[j]),
                'label': bool(domain_labels[j]),
                'correct': bool(domain_preds[j] == domain_labels[j])
            }
            for j, domain in enumerate(domains)
        }
        
        # Slot predictions
        example['predictions']['slots'] = {}
        for slot in slots:
            if f'{slot}_active' in labels:
                slot_logits = predictions['slot_activations'][slot][i]
                slot_pred = torch.argmax(slot_logits).item()
                slot_prob = torch.softmax(slot_logits, dim=-1)[1].item()
                slot_label = labels[f'{slot}_active'][i].item()
                
                slot_info = {
                    'predicted_active': bool(slot_pred),
                    'probability': float(slot_prob),
                    'label_active': bool(slot_label),
                    'correct': bool(slot_pred == slot_label)
                }
                
                # Get value vocabulary if available
                value_vocab = None
                if val_dataset is not None:
                    value_vocab = val_dataset.value_vocabs.get(slot)
                
                # Value prediction (for categorical slots)
                if slot_pred == 1 and slot in predictions.get('values', {}):
                    value_logits = predictions['values'][slot][i]
                    value_pred_idx = torch.argmax(value_logits).item()
                    value_probs = torch.softmax(value_logits, dim=-1)
                    top_k = 3
                    top_values = torch.topk(value_probs, min(top_k, len(value_probs)))
                    
                    # Convert indices to actual values if vocab available
                    idx2value = None
                    if value_vocab is not None:
                        idx2value = {v: k for k, v in value_vocab.items()}
                    
                    slot_info['value_prediction'] = {
                        'predicted_idx': int(value_pred_idx),
                        'predicted_value': idx2value[value_pred_idx] if idx2value else None,
                        'top_predictions': [
                            {
                                'idx': int(idx),
                                'value': idx2value[int(idx)] if idx2value else None,
                                'probability': float(prob)
                            }
                            for idx, prob in zip(top_values.indices.tolist(), 
                                                top_values.values.tolist())
                        ]
                    }
                    
                    # Add to predicted belief state DELTA
                    if idx2value and value_pred_idx in idx2value:
                        predicted_belief_delta[slot] = idx2value[value_pred_idx]
                    
                    # Add label if available
                    value_label_key = f'{slot}_value'
                    if value_label_key in labels:
                        value_label_idx = labels[value_label_key][i].item()
                        slot_info['value_prediction']['label_idx'] = int(value_label_idx)
                        slot_info['value_prediction']['label_value'] = idx2value[value_label_idx] if idx2value and value_label_idx in idx2value else None
                        slot_info['value_prediction']['correct'] = bool(
                            value_pred_idx == value_label_idx
                        )
                        
                        # Add to true belief state DELTA
                        if idx2value and value_label_idx in idx2value:
                            true_belief_delta[slot] = idx2value[value_label_idx]
                
                # If label is active, add to true belief DELTA (even if we didn't predict it)
                elif slot_label == 1:
                    value_label_key = f'{slot}_value'
                    if value_label_key in labels and value_vocab is not None:
                        value_label_idx = labels[value_label_key][i].item()
                        idx2value = {v: k for k, v in value_vocab.items()}
                        if value_label_idx in idx2value:
                            true_belief_delta[slot] = idx2value[value_label_idx]
                
                example['predictions']['slots'][slot] = slot_info
        
        # Add belief state DELTAS for comparison
        example['true_belief_state_delta'] = true_belief_delta
        example['predicted_belief_state_delta'] = predicted_belief_delta
        
        # Check if prediction is perfect
        example['is_perfect'] = (
            all(info['correct'] for info in example['predictions']['domains'].values()) and
            all(info['correct'] for info in example['predictions']['slots'].values())
        )
        
        batch_predictions.append(example)
    
    return batch_predictions


# ============================================================================
# Validation Function
# ============================================================================
def validate(model: nn.Module, val_loader: DataLoader, device, logger, 
            domains: List[str], slots: List[str], save_predictions: bool = True,
            tokenizer=None, val_dataset=None):
    """Run validation"""
    model.eval()
    
    total_loss = 0
    total_domain_loss = 0
    total_slot_loss = 0
    total_value_loss = 0
    
    metrics_calculator = DSTMetrics(domains, slots)
    
    # Store predictions for debugging
    all_predictions = []
    
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
            
            # Save predictions for debugging
            if save_predictions:
                batch_predictions = extract_predictions(
                    predictions, labels, batch, domains, slots, device,
                    tokenizer=tokenizer, val_dataset=val_dataset, batch_idx=batch_idx
                )
                all_predictions.extend(batch_predictions)
            
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
    
    return all_stats, all_predictions


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
    parser.add_argument('--output_dir', type=str, default='experiments/evaluations',
                       help='Base output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda/mps)')
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_name = Path(args.checkpoint).parent.name
    output_dir = Path(args.output_dir) / f"{checkpoint_name}_eval_{timestamp}"
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
        num_workers=0,
        collate_fn=collate_fn
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
    
    results, predictions = validate(
        model, val_loader, device, logger,
        val_dataset.domains, val_dataset.slots,
        save_predictions=True,
        tokenizer=tokenizer,
        val_dataset=val_dataset
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
    
    # Compute delta accuracy
    logger.info(f"\nComputing Delta Accuracy...")
    delta_stats = compute_delta_accuracy(predictions)
    logger.info(f"  Delta Accuracy: {delta_stats['delta_accuracy']*100:.2f}%")
    
    # Prepare summary with all DST metrics
    summary = {
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': str(args.checkpoint),
        'validation_file': args.val_file,
        'num_examples': len(predictions),
        
        # Loss statistics
        'loss': {
            'total': results['loss'],
            'domain': results['domain_loss'],
            'slot': results['slot_loss'],
            'value': results['value_loss']
        },
        
        # Main accuracy metrics
        'accuracy': {
            'joint_goal_accuracy': results['joint_goal_accuracy'],
            'delta_accuracy': delta_stats['delta_accuracy'],
            'avg_domain_f1': results['avg_domain_f1'],
            'avg_slot_f1': results['avg_slot_f1']
        },
        
        # Per-domain metrics
        'domain_metrics': results['domain_metrics'],
        
        # Per-slot metrics (all slots)
        'slot_metrics': results['slot_metrics'],
        
        # Delta-based slot metrics
        'delta_slot_metrics': {
            slot: {
                'correct': delta_stats['slot_correct'].get(slot, 0),
                'total': delta_stats['slot_total'].get(slot, 0),
                'accuracy': delta_stats['slot_correct'].get(slot, 0) / max(delta_stats['slot_total'].get(slot, 1), 1),
                'false_positives': delta_stats['slot_fp'].get(slot, 0),
                'false_negatives': delta_stats['slot_fn'].get(slot, 0)
            }
            for slot in set(list(delta_stats['slot_correct'].keys()) + 
                          list(delta_stats['slot_fp'].keys()) + 
                          list(delta_stats['slot_fn'].keys()))
        }
    }
    
    # Save summary
    summary_file = output_dir / 'summary.json'
    logger.info(f"\nSaving summary to: {summary_file}")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Prepare readable predictions
    readable_predictions = []
    for ex in predictions:
        true_delta = ex.get('true_belief_state_delta', {})
        pred_delta = ex.get('predicted_belief_state_delta', {})
        
        # Skip if both are empty
        if not true_delta and not pred_delta:
            continue
        
        readable_ex = {
            'dialogue_id': ex.get('dialogue_id', 'unknown'),
            'turn_id': ex.get('turn_id', -1),
            'utterance': ex.get('current_utterance', ''),
            'predicted': pred_delta,
            'ground_truth': true_delta
        }
        readable_predictions.append(readable_ex)
    
    # Save predictions
    predictions_file = output_dir / 'predictions.json'
    logger.info(f"Saving {len(readable_predictions)} predictions to: {predictions_file}")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(readable_predictions, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "="*70)
    logger.info("Evaluation completed successfully!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("  - summary.json: All DST metrics")
    logger.info("  - predictions.json: Prediction history")
    logger.info("="*70)


if __name__ == '__main__':
    main()
