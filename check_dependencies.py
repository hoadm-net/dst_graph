#!/usr/bin/env python3
"""
Dependency Checker for GraphDST

This script checks if all required dependencies are installed
and provides installation instructions for missing ones.
"""

import sys
from importlib import import_module
from typing import List, Tuple

REQUIRED_PACKAGES = [
    ("torch", "PyTorch", "pip install torch torchvision torchaudio"),
    ("torch_geometric", "PyTorch Geometric", "pip install torch-geometric"),
    ("transformers", "Transformers", "pip install transformers"),
    ("numpy", "NumPy", "pip install numpy"),
    ("yaml", "PyYAML", "pip install pyyaml"),
]

OPTIONAL_PACKAGES = [
    ("torch_scatter", "torch-scatter", "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html"),
    ("torch_sparse", "torch-sparse", "pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html"),
    ("wandb", "Weights & Biases", "pip install wandb"),
    ("tensorboard", "TensorBoard", "pip install tensorboard"),
    ("networkx", "NetworkX", "pip install networkx"),
    ("matplotlib", "Matplotlib", "pip install matplotlib"),
]


def check_package(package_name: str, display_name: str) -> Tuple[bool, str]:
    """
    Check if a package is installed and get its version
    
    Returns:
        Tuple of (is_installed, version_or_error)
    """
    try:
        module = import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def main():
    print("=" * 70)
    print("GraphDST Dependency Checker")
    print("=" * 70)
    
    # Check required packages
    print("\n📦 Checking Required Packages:")
    print("-" * 70)
    
    missing_required = []
    for package, display_name, install_cmd in REQUIRED_PACKAGES:
        is_installed, version_or_error = check_package(package, display_name)
        
        if is_installed:
            print(f"✓ {display_name:20s} : {version_or_error}")
        else:
            print(f"✗ {display_name:20s} : NOT INSTALLED")
            missing_required.append((display_name, install_cmd))
    
    # Check optional packages
    print("\n📦 Checking Optional Packages:")
    print("-" * 70)
    
    missing_optional = []
    for package, display_name, install_cmd in OPTIONAL_PACKAGES:
        is_installed, version_or_error = check_package(package, display_name)
        
        if is_installed:
            print(f"✓ {display_name:20s} : {version_or_error}")
        else:
            print(f"- {display_name:20s} : not installed (optional)")
            missing_optional.append((display_name, install_cmd))
    
    # Check PyTorch CUDA availability
    try:
        import torch
        print("\n🔧 PyTorch Configuration:")
        print("-" * 70)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("Note: Running on CPU (GPU acceleration not available)")
    except ImportError:
        pass
    
    # Check Python version
    print("\n🐍 Python Environment:")
    print("-" * 70)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    
    # Summary
    print("\n" + "=" * 70)
    
    if not missing_required:
        print("✅ All required packages are installed!")
    else:
        print("⚠️  Missing Required Packages:")
        print("-" * 70)
        for name, cmd in missing_required:
            print(f"\n{name}:")
            print(f"  {cmd}")
        
        print("\n" + "=" * 70)
        print("Install all required packages:")
        print("-" * 70)
        print("pip install torch torchvision torchaudio")
        print("pip install torch-geometric")
        print("pip install transformers numpy pyyaml")
        print("\nOr use requirements.txt:")
        print("pip install -r requirements.txt")
    
    if missing_optional:
        print("\n💡 Optional Packages (for enhanced features):")
        print("-" * 70)
        for name, cmd in missing_optional:
            print(f"{name}: {cmd}")
    
    print("\n" + "=" * 70)
    
    # Check if model can be imported
    print("\n🧪 Testing Model Import:")
    print("-" * 70)
    
    try:
        sys.path.insert(0, 'src')
        from models.graphdst import GraphDSTConfig, GraphDSTModel
        print("✓ GraphDST model can be imported successfully!")
        
        # Try to create a minimal config
        config = GraphDSTConfig(hidden_dim=256)
        print(f"✓ Config created: hidden_dim={config.hidden_dim}")
        
    except ImportError as e:
        print(f"✗ Cannot import GraphDST model: {e}")
        print("  Make sure you're running from the project root directory")
    except Exception as e:
        print(f"⚠️  Model import warning: {e}")
    
    print("\n" + "=" * 70)
    
    if missing_required:
        print("❌ Setup incomplete. Install missing packages first.")
        print("=" * 70)
        return 1
    else:
        print("✅ Setup complete! You're ready to use GraphDST.")
        print("=" * 70)
        print("\n🚀 Next steps:")
        print("  1. Run tests: python test_model.py")
        print("  2. Quick start: python quickstart.py")
        print("  3. Read docs: cat IMPLEMENTATION.md")
        return 0


if __name__ == "__main__":
    exit(main())
