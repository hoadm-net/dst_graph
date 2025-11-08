"""
Configuration loader and manager for GraphDST

This module handles loading and merging configuration files,
environment variable overrides, and command line arguments.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, fields
import argparse


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = "graphdst"
    type: str = "graph_neural_network"
    hidden_dim: int = 768
    num_gnn_layers: int = 3
    num_attention_heads: int = 8
    dropout: float = 0.1
    text_encoder: str = "bert-base-uncased"
    max_sequence_length: int = 512
    num_domains: int = 5
    num_slots: int = 37
    categorical_threshold: int = 50


@dataclass  
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 10
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    eval_steps: int = 500
    save_steps: int = 1000
    patience: int = 5
    metric_for_best_model: str = "joint_goal_accuracy"
    greater_is_better: bool = True


@dataclass
class DataConfig:
    """Data processing configuration"""
    data_dir: str = "data"
    ontology_path: str = "data/ontology.json"
    train_file: str = "train.json"
    val_file: str = "val.json"
    test_file: str = "test.json"
    max_history_turns: int = 3
    max_utterance_length: int = 100
    max_system_response_length: int = 150
    tokenizer: str = "bert-base-uncased"


@dataclass
class OutputConfig:
    """Output and logging configuration"""
    output_dir: str = "experiments"
    run_name: str = "graphdst_base"
    log_dir: str = "logs"
    log_level: str = "INFO"
    save_model: bool = True
    save_best_model: bool = True
    use_wandb: bool = False
    use_tensorboard: bool = True


@dataclass
class GraphDSTConfig:
    """Complete GraphDST configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42
    deterministic: bool = True


class ConfigManager:
    """Configuration manager for GraphDST"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.available_configs = self._discover_configs()
    
    def _discover_configs(self) -> Dict[str, Path]:
        """Discover available configuration files"""
        configs = {}
        
        if self.config_dir.exists():
            for config_file in self.config_dir.glob("*.yaml"):
                config_name = config_file.stem.replace("_config", "")
                configs[config_name] = config_file
        
        return configs
    
    def load_config(self, config_name: str = "base", 
                   config_overrides: Optional[Dict[str, Any]] = None) -> GraphDSTConfig:
        """
        Load configuration from file with optional overrides
        
        Args:
            config_name: Name of configuration to load
            config_overrides: Dictionary of configuration overrides
            
        Returns:
            Complete GraphDST configuration
        """
        # Load base configuration
        if config_name in self.available_configs:
            config_path = self.available_configs[config_name]
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            print(f"Configuration {config_name} not found. Using default configuration.")
            config_dict = {}
        
        # Apply overrides
        if config_overrides:
            config_dict = self._deep_merge(config_dict, config_overrides)
        
        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)
        
        # Create configuration object
        config = self._dict_to_config(config_dict)
        
        return config
    
    def _deep_merge(self, base_dict: Dict, override_dict: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base_dict.copy()
        
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config_dict: Dict) -> Dict:
        """Apply environment variable overrides"""
        # Define mapping of environment variables to config paths
        env_mappings = {
            'GRAPHDST_DATA_DIR': ['data', 'data_dir'],
            'GRAPHDST_OUTPUT_DIR': ['output', 'output_dir'],
            'GRAPHDST_BATCH_SIZE': ['training', 'batch_size'],
            'GRAPHDST_LEARNING_RATE': ['training', 'learning_rate'],
            'GRAPHDST_NUM_EPOCHS': ['training', 'num_epochs'],
            'GRAPHDST_HIDDEN_DIM': ['model', 'hidden_dim'],
            'GRAPHDST_USE_WANDB': ['output', 'use_wandb']
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the correct nested dictionary
                current_dict = config_dict
                for key in config_path[:-1]:
                    if key not in current_dict:
                        current_dict[key] = {}
                    current_dict = current_dict[key]
                
                # Set the value with appropriate type conversion
                final_key = config_path[-1]
                current_dict[final_key] = self._convert_env_value(env_value)
        
        return config_dict
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _dict_to_config(self, config_dict: Dict) -> GraphDSTConfig:
        """Convert dictionary to GraphDSTConfig object"""
        # Extract sub-configurations
        model_dict = config_dict.get('model', {})
        training_dict = config_dict.get('training', {})
        data_dict = config_dict.get('data', {})
        output_dict = {**config_dict.get('output', {}), **config_dict.get('logging', {})}
        
        # Create sub-config objects
        model_config = self._create_dataclass_from_dict(ModelConfig, model_dict)
        training_config = self._create_dataclass_from_dict(TrainingConfig, training_dict)
        data_config = self._create_dataclass_from_dict(DataConfig, data_dict)
        output_config = self._create_dataclass_from_dict(OutputConfig, output_dict)
        
        # Create main config
        main_config = GraphDSTConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            output=output_config,
            seed=config_dict.get('seed', 42),
            deterministic=config_dict.get('deterministic', True)
        )
        
        return main_config
    
    def _create_dataclass_from_dict(self, dataclass_type, data_dict: Dict):
        """Create dataclass instance from dictionary"""
        # Get field names and types
        field_names = {f.name for f in fields(dataclass_type)}
        
        # Filter dictionary to only include valid fields
        filtered_dict = {k: v for k, v in data_dict.items() if k in field_names}
        
        # Create instance
        return dataclass_type(**filtered_dict)
    
    def save_config(self, config: GraphDSTConfig, save_path: str):
        """Save configuration to file"""
        config_dict = self._config_to_dict(config)
        
        save_path = Path(save_path)
        
        if save_path.suffix == '.json':
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:  # Default to YAML
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _config_to_dict(self, config: GraphDSTConfig) -> Dict:
        """Convert GraphDSTConfig to dictionary"""
        return {
            'model': self._dataclass_to_dict(config.model),
            'training': self._dataclass_to_dict(config.training),
            'data': self._dataclass_to_dict(config.data),
            'output': self._dataclass_to_dict(config.output),
            'seed': config.seed,
            'deterministic': config.deterministic
        }
    
    def _dataclass_to_dict(self, dataclass_instance) -> Dict:
        """Convert dataclass instance to dictionary"""
        result = {}
        for field in fields(dataclass_instance):
            value = getattr(dataclass_instance, field.name)
            result[field.name] = value
        return result
    
    def list_configs(self) -> List[str]:
        """List available configuration names"""
        return list(self.available_configs.keys())
    
    def validate_config(self, config: GraphDSTConfig) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate paths
        if not Path(config.data.data_dir).exists():
            issues.append(f"Data directory does not exist: {config.data.data_dir}")
        
        if not Path(config.data.ontology_path).exists():
            issues.append(f"Ontology file does not exist: {config.data.ontology_path}")
        
        # Validate model parameters
        if config.model.hidden_dim <= 0:
            issues.append("Model hidden_dim must be positive")
        
        if config.model.num_gnn_layers <= 0:
            issues.append("Number of GNN layers must be positive")
        
        # Validate training parameters
        if config.training.batch_size <= 0:
            issues.append("Batch size must be positive")
        
        if config.training.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        
        return issues


def create_arg_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description="GraphDST Training")
    
    # Configuration
    parser.add_argument('--config', type=str, default='base',
                       help='Configuration name to use')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to custom configuration file')
    
    # Common overrides
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory path')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory path')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Run name for experiment')
    
    # Training overrides
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of training epochs')
    
    # Model overrides
    parser.add_argument('--hidden-dim', type=int, default=None,
                       help='Model hidden dimension')
    parser.add_argument('--num-gnn-layers', type=int, default=None,
                       help='Number of GNN layers')
    
    # Other options
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    return parser


def load_config_from_args(args: argparse.Namespace) -> GraphDSTConfig:
    """Load configuration from command line arguments"""
    config_manager = ConfigManager()
    
    # Build override dictionary from args
    overrides = {}
    
    if args.data_dir:
        overrides['data'] = {'data_dir': args.data_dir}
    if args.output_dir:
        overrides['output'] = {'output_dir': args.output_dir}
    if args.run_name:
        overrides.setdefault('output', {})['run_name'] = args.run_name
    
    if args.batch_size:
        overrides['training'] = {'batch_size': args.batch_size}
    if args.learning_rate:
        overrides.setdefault('training', {})['learning_rate'] = args.learning_rate
    if args.num_epochs:
        overrides.setdefault('training', {})['num_epochs'] = args.num_epochs
    
    if args.hidden_dim:
        overrides['model'] = {'hidden_dim': args.hidden_dim}
    if args.num_gnn_layers:
        overrides.setdefault('model', {})['num_gnn_layers'] = args.num_gnn_layers
    
    if args.seed:
        overrides['seed'] = args.seed
    
    # Load configuration
    config_name = args.config
    if args.config_file:
        # Load custom config file
        with open(args.config_file, 'r') as f:
            custom_config = yaml.safe_load(f)
        config = config_manager._dict_to_config(custom_config)
    else:
        config = config_manager.load_config(config_name, overrides)
    
    return config


if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager()
    
    print("Available configurations:")
    for config_name in config_manager.list_configs():
        print(f"  - {config_name}")
    
    # Load base configuration
    config = config_manager.load_config("base")
    
    print(f"\nLoaded configuration:")
    print(f"Model: {config.model.name}")
    print(f"Hidden dim: {config.model.hidden_dim}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Data dir: {config.data.data_dir}")
    print(f"Output dir: {config.output.output_dir}")
    
    # Validate configuration
    issues = config_manager.validate_config(config)
    if issues:
        print(f"\nConfiguration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration is valid!")
    
    # Save configuration example
    save_path = "configs/loaded_config.yaml"
    config_manager.save_config(config, save_path)
    print(f"\nSaved configuration to {save_path}")