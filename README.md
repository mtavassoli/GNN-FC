# GNN-FC

A PyTorch implementation of:

**Model-Agnostic Fairness Regularization for GNNs with Incomplete Sensitive Information**

This framework introduces a model-agnostic fairness enhancement method for Graph Neural Networks (GNNs) under incomplete sensitive attribute labels, using semi-supervised fairness regularization based on statistical parity and equal opportunity (EOSP).

---

# Installation

**Clone the repository:**


```cmd
git clone https://github.com/mtavassoli/GNN-FC

cd GNN-FC
```



---

# Usage

**Integration:**

The EOSP fairness constraint can be easily integrated into existing GNN training loops:

**During training**
```python
from gnn_fc.losses import FairnessLoss

fairness_loss_fn = FairnessLoss(alpha, beta)

for _ in range(epochs):
    model.train()
    
    # Forward pass
    predictions = model(features, G)
    
    # Compute task-specific loss (e.g., BCE for classification)
    task_loss = F.binary_cross_entropy_with_logits(predictions[training_index], labels[training_index].float())
    
    # Compute fairness loss
    fair_loss = fairness_loss_fn(labels, sensitive_attributes, training_index, torch.sigmoid(predictions))
    
    # Total loss
    total_loss = task_loss + fair_loss
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

**Example:**

Run the main script:

```python
python main.py --enable_eosp True --model GCN --dataset german
```

Example Output (Model: GCN, Dataset: German)
```cmd
🔹 Running GERMAN | Split seed: 773 | Model: GCN | EOSP: True

🔹 Running GERMAN | Split seed: 429 | Model: GCN | EOSP: True

🔹 Running GERMAN | Split seed: 231 | Model: GCN | EOSP: True

🔹 Running GERMAN | Split seed: 258 | Model: GCN | EOSP: True

🔹 Running GERMAN | Split seed: 1002 | Model: GCN | EOSP: True
Dataset german
##########################################################################################
Balanced ACC: 60.02 ± 5.25, AUC: 63.21 ± 5.32, F1: 63.89 ± 8.75, Parity: 1.98 ± 1.81, Equality: 1.86 ± 1.04
##########################################################################################
```

# Project Structure
``` cmd
GNN-FC/
├── examples/
├── gnn_fc/
├── evaluation.py
├── main.py
├── metrics.py
├── objective.py
├── utils.py
├── LICENSE
└── README.txt
```

# License

Distributed under the MIT License.
See the LICENSE file for full details.

---

# Contact

Mahdi Tavassoli Kejani

Email: mtk.tavassoli [at] gmail.com

Project link: https://github.com/mtavassoli/GNN-FC

---

# Citation

If you use this repository or find our research helpful, please cite:

@misc{kejani2025modelagnosticfairnessregularizationgnns,
      title={Model-Agnostic Fairness Regularization for GNNs with Incomplete Sensitive Information}, 
      author={Mahdi Tavassoli Kejani and Fadi Dornaika and Jean-Michel Loubes},
      year={2025},
      eprint={2512.03074},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.03074}, 
}

---

Thank you for your interest in our project!
We welcome contributions, discussions, and feedback.
