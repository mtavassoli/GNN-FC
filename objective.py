from torch_geometric.utils import from_scipy_sparse_matrix
import dgl
from utils import set_seed, create_model


# -------------------------------
# Optuna Objective Function
# -------------------------------
def objective(trial, dataset, args, model_name: str, enable_eosp: bool):
    """Objective function for Optuna hyperparameter optimization."""
    set_seed(args.seed)

    # Fairness coefficients search space
    coeff_space = [0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09,
                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]

    if enable_eosp:

        args.eo_coeff = trial.suggest_categorical('eo_coeff', coeff_space)
        args.sp_coeff = trial.suggest_categorical('sp_coeff', coeff_space)
    else:
        args.eo_coeff = args.sp_coeff = 0.0

    # Graph preprocessing
    edge_index, _ = from_scipy_sparse_matrix(dataset.adj)
    graph_dgl = dgl.from_scipy(dataset.adj)

    # Model initialization
    model = create_model(model_name, dataset.features.shape[1], args)

    # Fit depending on model type
    if model_name == 'FairGNN':
        metrics, score = model.fit(
            dataset.features, dataset.labels,
            dataset.idx_train, dataset.idx_val, dataset.idx_test,
            dataset.sens, graph_dgl
        )
    elif model_name == 'FairVGNN':
        metrics, score = model.fit(
            dataset.adj, dataset.features, dataset.labels,
            dataset.idx_train, dataset.idx_val, dataset.idx_test,
            dataset.sens, 0
        )
    else:
        metrics, score = model.fit(
            dataset.features, dataset.labels,
            dataset.idx_train, dataset.idx_val, dataset.idx_test,
            dataset.sens, edge_index
        )

    trial.set_user_attr("metrics", metrics)
    return score
