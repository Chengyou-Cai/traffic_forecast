import numpy as np
from torch.utils.data import Dataset

class TrafficDataset(Dataset):

    def __init__(self,data_file_paths,category="train",scaler=None) -> None:
        super(TrafficDataset,self).__init__()
        assert (category == 'train' or category == 'valid' or category=='test')
        
        self.data_file_paths = data_file_paths

        self.data = dict()
        self.get_data(category=category)
        
        if scaler:
            scaler.fit(self.train_npz['x'][...,0]) 
            self.data['x'][...,0] = scaler.transform(self.data['x'][...,0]) # [...,0]: (bs,12,207,2)->(bs,12,207)
    
    def get_data(self,category):
        if category == "train":
            self.data['x'],self.data['y'] = self.train_npz['x'], self.train_npz['y']
        elif category == "valid":
            self.data['x'],self.data['y'] = self.valid_npz['x'], self.valid_npz['y']
        else:
            self.data['x'],self.data['y'] = self.test_npz['x'], self.test_npz['y']
    
    @property
    def train_npz(self):
        return np.load(self.data_file_paths["train"])

    @property
    def valid_npz(self):
        return np.load(self.data_file_paths["valid"])

    @property
    def test_npz(self):
        return np.load(self.data_file_paths["test"])

    def __getitem__(self, index):
        return self.data['x'][index],self.data['y'][index]

    def __len__(self):
        return self.data['x'].shape[0]