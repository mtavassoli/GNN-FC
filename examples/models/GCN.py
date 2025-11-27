import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import balanced_accuracy_score
from gnn_fc.losses import FairnessLoss
from metrics import evaluate_group_fairness

from torchmetrics.functional.classification import (
    binary_auroc,
    binary_f1_score,
)
import numpy as np
class GCN(nn.Module):
    def __init__(self, nfeat, args):
        super(GCN, self).__init__()
        self.args = args
        self.convs = nn.ModuleList([GCNConv(nfeat if i == 0 else self.args.num_hidden, self.args.num_hidden) for i in range(self.args.num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.args.num_hidden) for _ in range(self.args.num_layers - 1)])
        self.fc = nn.Linear(int(self.args.num_hidden), 1)
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        for layer in range(self.args.num_layers-1):
            x = self.convs[layer](x, edge_index)
            x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = F.relu(self.convs[-1](x, edge_index))
        preds = self.fc(x)
        return preds.squeeze()
   
    def fit(self, features, labels, idx_train, idx_val, idx_test, sens, G):
        best_result = {}
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        best_metric = 0
        fairnessLoss = FairnessLoss(self.args.eo_coeff, self.args.sp_coeff)

        for _ in range(100):
            self.train()
            preds = self(features, G)
            loss_prediction = F.binary_cross_entropy_with_logits(preds[idx_train], labels[idx_train].float())
            fair_loss = fairnessLoss(labels, sens, idx_train, torch.sigmoid(preds))
            loss = loss_prediction + fair_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.eval()
            preds = self(features, G)
            if preds.dim() == 1:
                # ----- Validation -----
                parity_val, equality_val = evaluate_group_fairness(preds, idx_val, labels, sens)
                labels_val = labels[idx_val].cpu().numpy()
                preds_val = (preds[idx_val].squeeze() > 0).type_as(labels).cpu().numpy()
                balanced_acc_val = balanced_accuracy_score(labels_val, preds_val)

                # ----- Test -----
                auc_test = binary_auroc(preds[idx_test], labels[idx_test])
                f1_test = binary_f1_score(preds[idx_test], labels[idx_test])
                parity_test, equality_test = evaluate_group_fairness(preds, idx_test, labels, sens)
                labels_test = labels[idx_test].cpu().numpy()
                preds_test = (preds[idx_test].squeeze() > 0).type_as(labels).cpu().numpy()
                balanced_acc_test = balanced_accuracy_score(labels_test, preds_test)

            # ----- Validation Metric -----
            fairness_penalty = 0.5 * (
                (100 - parity_val * 100) + (100 - equality_val * 100)
            )
            current_metric = (balanced_acc_val * 100) + fairness_penalty

            # ----- Track Best Result -----
            if current_metric > best_metric:
                best_metric = current_metric
                best_result["f1"] = f1_test
                best_result["roc"] = auc_test
                best_result["parity"] = parity_test
                best_result["equality"] = equality_test
                best_result["balanced_acc"] = balanced_acc_test

        return best_result, best_metric