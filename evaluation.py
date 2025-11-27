import numpy as np

def summarize_results(seed_list, best_result, dataset_name):
    f1_scores, bal_acc_scores, roc_scores, parity_scores, equality_scores = [], [], [], [], []

    for seed in seed_list:
        if seed in best_result:
            result = best_result[seed]
            f1_scores.append(result["f1"])
            roc_scores.append(result["roc"])
            parity_scores.append(result["parity"])
            equality_scores.append(result["equality"])
            bal_acc_scores.append(result["balanced_acc"])

    # Compute mean & std
    f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
    roc_mean, roc_std = np.mean(roc_scores), np.std(roc_scores)
    parity_mean, parity_std = np.mean(parity_scores), np.std(parity_scores)
    equality_mean, equality_std = np.mean(equality_scores), np.std(equality_scores)
    bal_mean, bal_std = np.mean(bal_acc_scores), np.std(bal_acc_scores)

    print(f"Dataset {dataset_name}")
    print("#" * 90)
    print(
        f"Balanced ACC: {bal_mean * 100:.2f} ± {bal_std * 100:.2f}, "
        f"AUC: {roc_mean * 100:.2f} ± {roc_std * 100:.2f}, "
        f"F1: {f1_mean * 100:.2f} ± {f1_std * 100:.2f}, "
        f"Parity: {parity_mean * 100:.2f} ± {parity_std * 100:.2f}, "
        f"Equality: {equality_mean * 100:.2f} ± {equality_std * 100:.2f}"
    )
    print("#" * 90)
