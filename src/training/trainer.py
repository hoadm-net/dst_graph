"""
Training Loop for GraphDST Model

This module implements:
- Multi-task training with domain, slot, and value prediction
- Learning rate scheduling and optimization
- Evaluation metrics and validation
- Model checkpointing and logging
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model parameters
    model_name: str = "graphdst"
    hidden_dim: int = 768
    num_gnn_layers: int = 3
    num_attention_heads: int = 8
    dropout: float = 0.1
    
    # Training parameters
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Data parameters
    max_length: int = 512
    max_history_turns: int = 3
    
    # Validation and saving
    eval_steps: int = 500
    save_steps: int = 1000
    patience: int = 5  # Early stopping patience
    
    # Paths
    data_dir: str = "data"
    output_dir: str = "experiments"
    log_dir: str = "logs"
    
    # Loss weights
    domain_loss_weight: float = 1.0
    slot_loss_weight: float = 1.0
    value_loss_weight: float = 1.0


class MetricsCalculator:
    """Calculate evaluation metrics for DST"""
    
    def __init__(self, domains: List[str], slots: List[str]):
        """
        Initialize metrics calculator
        
        Args:
            domains: List of domain names
            slots: List of slot names
        """
        self.domains = domains
        self.slots = slots
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_examples = 0
        self.correct_domains = 0
        self.correct_slots = 0
        self.correct_joint_goals = 0
        
        # Per-domain metrics
        self.domain_tp = [0] * len(self.domains)
        self.domain_fp = [0] * len(self.domains)
        self.domain_fn = [0] * len(self.domains)
        
        # Per-slot metrics  
        self.slot_tp = {slot: 0 for slot in self.slots}
        self.slot_fp = {slot: 0 for slot in self.slots}
        self.slot_fn = {slot: 0 for slot in self.slots}
        
        # Value prediction metrics
        self.correct_values = {slot: 0 for slot in self.slots}
        self.total_values = {slot: 0 for slot in self.slots}
    
    def update(self, predictions: Dict, labels: Dict, batch_size: int):
        """
        Update metrics with batch predictions
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            batch_size: Batch size
        """
        self.total_examples += batch_size
        
        # Domain metrics
        if 'domains' in predictions and 'domain_labels' in labels:
            self._update_domain_metrics(predictions['domains'], labels['domain_labels'])
        
        # Slot metrics
        if 'slot_activations' in predictions and 'slot_labels' in labels:
            self._update_slot_metrics(predictions['slot_activations'], labels['slot_labels'])
        
        # Value metrics
        if 'values' in predictions and 'value_labels' in labels:
            self._update_value_metrics(predictions['values'], labels['value_labels'])
        
        # Joint goal accuracy
        self._update_joint_goal_accuracy(predictions, labels)
    
    def _update_domain_metrics(self, pred_domains, true_domains):
        """Update domain classification metrics"""
        # Convert predictions to binary (assuming logits)
        # pred_binary = (pred_domains > 0.5).int()  # Would use torch.sigmoid in real implementation
        
        for i, (pred, true) in enumerate(zip(pred_domains, true_domains)):
            # Simplified binary conversion (placeholder)
            pred_binary = [1 if p > 0.5 else 0 for p in pred] if isinstance(pred, list) else pred
            
            for domain_idx in range(len(self.domains)):
                pred_val = pred_binary[domain_idx] if isinstance(pred_binary, list) else pred_binary
                true_val = true[domain_idx] if isinstance(true, list) else true
                
                if pred_val == 1 and true_val == 1:
                    self.domain_tp[domain_idx] += 1
                elif pred_val == 1 and true_val == 0:
                    self.domain_fp[domain_idx] += 1
                elif pred_val == 0 and true_val == 1:
                    self.domain_fn[domain_idx] += 1
    
    def _update_slot_metrics(self, pred_slots, true_slots):
        """Update slot activation metrics"""
        for slot_name in self.slots:
            if slot_name in pred_slots and slot_name in true_slots:
                pred_activations = pred_slots[slot_name]
                true_activations = true_slots[slot_name]
                
                for pred, true in zip(pred_activations, true_activations):
                    pred_binary = 1 if pred > 0.5 else 0
                    
                    if pred_binary == 1 and true == 1:
                        self.slot_tp[slot_name] += 1
                    elif pred_binary == 1 and true == 0:
                        self.slot_fp[slot_name] += 1
                    elif pred_binary == 0 and true == 1:
                        self.slot_fn[slot_name] += 1
    
    def _update_value_metrics(self, pred_values, true_values):
        """Update value prediction metrics"""
        for slot_name in self.slots:
            if slot_name in pred_values and slot_name in true_values:
                pred_vals = pred_values[slot_name]
                true_vals = true_values[slot_name]
                
                # Handle both categorical and span predictions
                for pred, true in zip(pred_vals, true_vals):
                    self.total_values[slot_name] += 1
                    
                    if isinstance(true, tuple):  # Span prediction
                        # Check if predicted span matches true span
                        if isinstance(pred, dict) and 'start_pos' in pred and 'end_pos' in pred:
                            if pred['start_pos'] == true[0] and pred['end_pos'] == true[1]:
                                self.correct_values[slot_name] += 1
                    else:  # Categorical prediction
                        pred_class = pred if isinstance(pred, int) else int(pred)
                        if pred_class == true:
                            self.correct_values[slot_name] += 1
    
    def _update_joint_goal_accuracy(self, predictions, labels):
        """Update joint goal accuracy (exact match of entire belief state)"""
        # Simplified JGA calculation (placeholder)
        # In practice, would need to reconstruct full belief state and compare
        batch_size = len(labels.get('domain_labels', []))
        
        for i in range(batch_size):
            # Placeholder for exact belief state matching
            # Would need to combine domain, slot, and value predictions
            # and compare with ground truth belief state
            is_exact_match = False  # Placeholder
            
            if is_exact_match:
                self.correct_joint_goals += 1
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute final metrics"""
        if self.total_examples == 0:
            return {}
        
        metrics = {}
        
        # Overall accuracy metrics
        metrics['joint_goal_accuracy'] = self.correct_joint_goals / self.total_examples
        
        # Domain metrics
        domain_precisions = []
        domain_recalls = []
        domain_f1s = []
        
        for i, domain in enumerate(self.domains):
            tp, fp, fn = self.domain_tp[i], self.domain_fp[i], self.domain_fn[i]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            domain_precisions.append(precision)
            domain_recalls.append(recall)
            domain_f1s.append(f1)
            
            metrics[f'domain_{domain}_precision'] = precision
            metrics[f'domain_{domain}_recall'] = recall
            metrics[f'domain_{domain}_f1'] = f1
        
        metrics['domain_avg_precision'] = sum(domain_precisions) / len(domain_precisions)
        metrics['domain_avg_recall'] = sum(domain_recalls) / len(domain_recalls)
        metrics['domain_avg_f1'] = sum(domain_f1s) / len(domain_f1s)
        
        # Slot metrics
        slot_precisions = []
        slot_recalls = []
        slot_f1s = []
        
        for slot in self.slots:
            tp = self.slot_tp[slot]
            fp = self.slot_fp[slot] 
            fn = self.slot_fn[slot]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            slot_precisions.append(precision)
            slot_recalls.append(recall)
            slot_f1s.append(f1)
        
        metrics['slot_avg_precision'] = sum(slot_precisions) / len(slot_precisions)
        metrics['slot_avg_recall'] = sum(slot_recalls) / len(slot_recalls)
        metrics['slot_avg_f1'] = sum(slot_f1s) / len(slot_f1s)
        
        # Value accuracy
        total_correct_values = sum(self.correct_values.values())
        total_value_predictions = sum(self.total_values.values())
        
        metrics['value_accuracy'] = total_correct_values / total_value_predictions if total_value_predictions > 0 else 0.0
        
        return metrics


class GraphDSTTrainer:
    """Main trainer for GraphDST model"""
    
    def __init__(self, config: TrainingConfig, model, data_loaders: Dict, tokenizer):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
            model: GraphDST model
            data_loaders: Dictionary of data loaders
            tokenizer: Text tokenizer
        """
        self.config = config
        self.model = model
        self.data_loaders = data_loaders
        self.tokenizer = tokenizer
        
        # Setup directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize optimizer and scheduler (placeholders)
        self.optimizer = None  # Would be AdamW in real implementation
        self.scheduler = None  # Would be linear scheduler with warmup
        
        # Metrics calculator
        domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi']
        slots = list(model.slot_info['slot_names'])
        self.metrics_calculator = MetricsCalculator(domains, slots)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_jga = 0.0
        self.patience_counter = 0
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.config.log_dir, f"training_{int(time.time())}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting training with config: {asdict(self.config)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train() if hasattr(self.model, 'train') else None  # Set training mode
        
        epoch_losses = defaultdict(list)
        data_loader = self.data_loaders['train']
        
        self.logger.info(f"Starting epoch {self.epoch + 1}")
        
        for batch_idx, batch_data in enumerate(data_loader):
            # Forward pass
            predictions = self.model.forward(batch_data)
            
            # Compute loss
            losses = self.model.compute_loss(predictions, batch_data['labels'])
            total_loss = losses['total']
            
            # Backward pass (placeholder)
            # total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            # self.optimizer.step()
            # self.scheduler.step()
            # self.optimizer.zero_grad()
            
            # Track losses
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name].append(float(loss_value))
            
            self.global_step += 1
            
            # Logging
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Epoch {self.epoch + 1}, Batch {batch_idx}, "
                    f"Loss: {total_loss:.4f}, "
                    f"Step: {self.global_step}"
                )
            
            # Validation
            if self.global_step % self.config.eval_steps == 0:
                val_metrics = self.validate()
                self.val_metrics.append(val_metrics)
                
                # Check for improvement
                current_jga = val_metrics.get('joint_goal_accuracy', 0.0)
                if current_jga > self.best_val_jga:
                    self.best_val_jga = current_jga
                    self.patience_counter = 0
                    self.save_model(f"best_model_step_{self.global_step}")
                    self.logger.info(f"New best JGA: {current_jga:.4f}")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    self.logger.info("Early stopping triggered")
                    return {k: sum(v) / len(v) for k, v in epoch_losses.items()}
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}")
        
        # Compute average losses for epoch
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        self.train_losses.append(avg_losses)
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval() if hasattr(self.model, 'eval') else None  # Set evaluation mode
        
        self.metrics_calculator.reset()
        val_losses = defaultdict(list)
        
        data_loader = self.data_loaders['val']
        
        self.logger.info("Running validation...")
        
        for batch_data in data_loader:
            with torch.no_grad() if 'torch' in globals() else nullcontext():
                # Forward pass
                predictions = self.model.forward(batch_data)
                
                # Compute loss
                losses = self.model.compute_loss(predictions, batch_data['labels'])
                
                # Track losses
                for loss_name, loss_value in losses.items():
                    val_losses[loss_name].append(float(loss_value))
                
                # Update metrics
                batch_size = len(batch_data['dialogue_ids'])
                self.metrics_calculator.update(predictions, batch_data['labels'], batch_size)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_metrics()
        
        # Add loss information
        avg_val_losses = {f"val_{k}": sum(v) / len(v) for k, v in val_losses.items()}
        metrics.update(avg_val_losses)
        
        # Log metrics
        self.logger.info(f"Validation Results:")
        self.logger.info(f"  Joint Goal Accuracy: {metrics.get('joint_goal_accuracy', 0.0):.4f}")
        self.logger.info(f"  Domain F1: {metrics.get('domain_avg_f1', 0.0):.4f}")
        self.logger.info(f"  Slot F1: {metrics.get('slot_avg_f1', 0.0):.4f}")
        self.logger.info(f"  Value Accuracy: {metrics.get('value_accuracy', 0.0):.4f}")
        
        self.model.train() if hasattr(self.model, 'train') else None  # Back to training mode
        
        return metrics
    
    def test(self) -> Dict[str, float]:
        """Run final test evaluation"""
        self.logger.info("Running final test evaluation...")
        
        # Load best model
        best_model_path = os.path.join(self.config.output_dir, "best_model")
        if os.path.exists(best_model_path):
            self.load_model(best_model_path)
            self.logger.info("Loaded best model for testing")
        
        self.model.eval() if hasattr(self.model, 'eval') else None
        
        self.metrics_calculator.reset()
        test_losses = defaultdict(list)
        
        data_loader = self.data_loaders['test']
        
        for batch_data in data_loader:
            with torch.no_grad() if 'torch' in globals() else nullcontext():
                predictions = self.model.forward(batch_data)
                losses = self.model.compute_loss(predictions, batch_data['labels'])
                
                for loss_name, loss_value in losses.items():
                    test_losses[loss_name].append(float(loss_value))
                
                batch_size = len(batch_data['dialogue_ids'])
                self.metrics_calculator.update(predictions, batch_data['labels'], batch_size)
        
        # Compute final metrics
        test_metrics = self.metrics_calculator.compute_metrics()
        avg_test_losses = {f"test_{k}": sum(v) / len(v) for k, v in test_losses.items()}
        test_metrics.update(avg_test_losses)
        
        # Log final results
        self.logger.info("="*50)
        self.logger.info("FINAL TEST RESULTS:")
        self.logger.info(f"Joint Goal Accuracy: {test_metrics.get('joint_goal_accuracy', 0.0):.4f}")
        self.logger.info(f"Domain Avg F1: {test_metrics.get('domain_avg_f1', 0.0):.4f}")
        self.logger.info(f"Slot Avg F1: {test_metrics.get('slot_avg_f1', 0.0):.4f}")
        self.logger.info(f"Value Accuracy: {test_metrics.get('value_accuracy', 0.0):.4f}")
        self.logger.info("="*50)
        
        # Save test results
        results_path = os.path.join(self.config.output_dir, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        return test_metrics
    
    def train(self) -> Dict[str, float]:
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            self.logger.info(f"Epoch {epoch + 1} completed:")
            self.logger.info(f"  Train Loss: {train_losses.get('total', 0.0):.4f}")
            
            # Early stopping check
            if self.patience_counter >= self.config.patience:
                self.logger.info("Training stopped early due to no improvement")
                break
        
        # Final test evaluation
        test_results = self.test()
        
        # Save training history
        self.save_training_history()
        
        return test_results
    
    def save_model(self, name: str):
        """Save model checkpoint"""
        save_path = os.path.join(self.config.output_dir, name)
        
        # In real implementation, would save model state dict
        # torch.save({
        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'scheduler_state_dict': self.scheduler.state_dict(),
        #     'epoch': self.epoch,
        #     'global_step': self.global_step,
        #     'best_val_jga': self.best_val_jga,
        #     'config': asdict(self.config)
        # }, save_path)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        # In real implementation, would load model state dict
        # checkpoint = torch.load(path)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # self.epoch = checkpoint['epoch']
        # self.global_step = checkpoint['global_step']
        # self.best_val_jga = checkpoint['best_val_jga']
        
        self.logger.info(f"Model loaded from {path}")
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint"""
        self.save_model(name)
        
        # Save additional training state
        checkpoint_info = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_jga': self.best_val_jga,
            'patience_counter': self.patience_counter,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        
        info_path = os.path.join(self.config.output_dir, f"{name}_info.json")
        with open(info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
    
    def save_training_history(self):
        """Save complete training history"""
        history = {
            'config': asdict(self.config),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_val_jga': self.best_val_jga,
            'total_steps': self.global_step,
            'total_epochs': self.epoch + 1
        }
        
        history_path = os.path.join(self.config.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")


# Context manager placeholder for no torch
class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def main():
    """Main training function"""
    # Configuration
    config = TrainingConfig(
        num_epochs=10,
        batch_size=16,
        learning_rate=2e-5,
        data_dir="../../data",
        output_dir="../../experiments/graphdst_run1"
    )
    
    # In real implementation, would:
    # 1. Load data loaders
    # 2. Initialize model
    # 3. Initialize tokenizer
    # 4. Create trainer
    # 5. Start training
    
    print("Training configuration:")
    print(json.dumps(asdict(config), indent=2))
    
    # Placeholder for actual training
    print("Training would start here with actual PyTorch implementation")


if __name__ == "__main__":
    main()