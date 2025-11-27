from abc import ABCMeta, abstractmethod
from torch_geometric.data import Data

class BaseDataset(metaclass=ABCMeta):
    def __init__(self, dataname):
        sens_attr, predict_attr, path, label_number, sens_idx = self._on_init_get_data_specification()
        features, adj, edge_index, labels, idx_train, idx_val, idx_test, sens = self.load_data(dataset=dataname,
                                                                                              sens_attr=sens_attr,
                                                                                              predict_attr=predict_attr,                                                                                         path=path,
                                                                                              label_number=label_number)
        self.features = features
        self.adj = adj
        self.edge_index = edge_index
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.sens_idx = sens_idx

      
        self.data = Data(x=self.features,
                         edge_index=self.edge_index,
                         y=self.labels.float(),
                         idx_train=self.idx_train,
                         idx_val=self.idx_val,
                         idx_test=self.idx_test,
                         sens=self.sens)

    
    @abstractmethod
    def _on_init_get_data_specification(self):
        raise NotImplementedError

    @abstractmethod
    def load_data(self, dataset, sens_attr, predict_attr, path, label_number):
        raise NotImplementedError