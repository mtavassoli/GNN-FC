from examples.datasets.base_dataset import BaseDataset
import pandas as pd
import random
import scipy.sparse as sp
import torch
import numpy as np
import os
import requests
from torch_geometric.utils import from_scipy_sparse_matrix
from utils import feature_norm


def download(url: str, file: str):
        r = requests.get(url)
        assert r.status_code == 200
        open(file, "wb").write(r.content)

class GermanDataset(BaseDataset):
    def __init__(self, dataset_name, seed):
        self.dataset_name = dataset_name
        self.seed = seed
        super().__init__(dataset_name)

    def _on_init_get_data_specification(self):
        sens_attr = "Gender"
        predict_attr = "GoodCustomer"
        label_number = 100
        path = "./examples/data/german/"
        sens_idx = 0    
        return sens_attr, predict_attr, path, label_number, sens_idx

    def load_data(self, dataset, sens_attr, predict_attr, path, label_number):
        if not os.path.exists(os.path.join(path)):
            os.makedirs(path)

        if not os.path.exists(os.path.join(path, "german.csv")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/german/german.csv"
            file = os.path.join(path, "german.csv")
            download(url, file)

        if not os.path.exists(os.path.join(path, "german_edges.txt")):
            url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/german/german_edges.txt"
            file = os.path.join(path, "german_edges.txt")
            download(url, file)

        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))

        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove("OtherLoansAtStore")
        header.remove("PurposeOfLoan")

        # Sensitive Attribute
        idx_features_labels.loc[idx_features_labels["Gender"] == "Female", "Gender"] = 1
        idx_features_labels.loc[idx_features_labels["Gender"] == "Male", "Gender"] = 0


        edges_unordered = np.genfromtxt(os.path.join(path, f"{dataset}_edges.txt")).astype("int")

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = adj + sp.eye(adj.shape[0])

        edge_index, _ = from_scipy_sparse_matrix(adj)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        import random
        random.seed(self.seed)
        label_idx_0 = np.where(labels==0)[0]
        label_idx_1 = np.where(labels==1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
    

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)
        idx_sens_train = list(set(idx_train))
        idx_sens_train = torch.LongTensor(idx_sens_train)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        if sens_attr:
            sens[sens>0]=1
        
        labels[labels>1]=1
       
        features = feature_norm(features)
        return features, adj, edge_index, labels, idx_train, idx_val, idx_test, sens