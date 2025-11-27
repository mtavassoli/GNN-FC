import torch.nn.functional as F
import optuna
import argparse
from examples.datasets import GermanDataset
from evaluation import summarize_results
from objective import objective
from utils import set_seed, get_configs

# -------------------------------
# Experiment Runner
# -------------------------------
def run_experiments(enable_eosp, model_name, dataset_name):

    seed_list = [773, 429, 231, 258, 1002]
    best_results = {}

    # Load configuration
    config_path = f"./examples/configs/{dataset_name}/{model_name}.yml"
    args = get_configs(config_path)
    set_seed(args.seed)

    for split_seed in seed_list:
        print(f"\n🔹 Running {dataset_name.upper()} | Split seed: {split_seed} | Model: {model_name} | EOSP: {enable_eosp}")

        # Load dataset based on name
        if dataset_name == 'german':
            dataset = GermanDataset(dataset_name, split_seed)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        # Create Optuna study
        optuna.logging.disable_default_handler()
        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(direction='maximize', sampler=sampler)
           

        # Run optimization
        n_trials = 15 if enable_eosp else 1
        study.optimize(lambda trial: objective(trial, dataset, args, model_name, enable_eosp),
                        n_trials=n_trials)

        best_results[split_seed] = study.best_trial.user_attrs["metrics"]

    # Summarize results
    summarize_results(seed_list, best_results, dataset_name)


# -------------------------------
# Main Entry
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run graph neural network experiments')
    parser.add_argument('--enable_eosp', type=bool, default=True,
                       help='Enable EOSP (default: False)')
    parser.add_argument('--model_name', type=str, default='GCN',
                       help='Model name (default: GCN)')
    parser.add_argument('--dataset_name', type=str, default='german',
                       choices=['german', 'nba'],
                       help='Dataset name (default: german)')
    
    args = parser.parse_args()
    
    run_experiments(args.enable_eosp, args.model_name, args.dataset_name)