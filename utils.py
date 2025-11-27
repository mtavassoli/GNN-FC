# Training settings
import argparse
import yaml
import numpy as np
import torch
import random
from examples.models.GCN import GCN

def load_config(yaml_file: str) -> dict:
    """Load YAML configuration file."""
    try:
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error reading config file: {e}")
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found: {yaml_file}")

def create_arg_parser(config: dict) -> argparse.ArgumentParser:
    """Create argument parser using YAML config defaults."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=config.get('seed', 0), help='Random seed.')
    parser.add_argument('--epochs', type=int, default=config.get('epochs', 100), help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=config.get('lr', 0.01), help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=config.get('weight_decay', 1e-5), help='Weight decay (L2).')
    parser.add_argument('--dropout', type=float, default=config.get('dropout', 0.2), help='Dropout rate.')
    parser.add_argument('--dataset', type=str, default=config.get('dataset', 'german'))
    parser.add_argument('--model', type=str, default=config.get('model', 'GCN'))
    parser.add_argument('--enable_eosp', type=bool, default=False, help='Enable EOSP (default: False)')
    parser.add_argument('--num_hidden', type=int, default=config.get('num_hidden', 32), help='Number of hidden units.')
    parser.add_argument('--num_layers', type=int, default=config.get('num_layers', 1), help='Number of hidden layers.')
    parser.add_argument('--eo_coeff', type=float, default=config.get('eo_coeff', 0.2), help='Equality Opportunity coefficient.')
    parser.add_argument('--sp_coeff', type=float, default=config.get('sp_coeff', 0.2), help='Statistical Parity coefficient.')
  
    return parser

def get_configs(config_file: str, args_list: list = None) -> argparse.Namespace:
    """Load config and parse arguments."""
    config = load_config(config_file)
    parser = create_arg_parser(config)
    return parser.parse_args(args_list)

def set_seed(seed: int):
    """Ensure reproducible results across Python, NumPy, and PyTorch (CPU & GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(model_name: str, input_dim: int, args, **kwargs):
    """Return an initialized model instance by name."""
    model_registry = {
        'GCN': GCN,    
    }

    if model_name not in model_registry:
        raise ValueError(f"Unknown model type: {model_name}")

    return model_registry[model_name](nfeat=input_dim, args=args, **kwargs)

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1
