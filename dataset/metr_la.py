import numpy as np
from torch.utils.data import Dataset

class METR_LA(Dataset):

    def __init__(self,category="train",scaler=None) -> None:
        super(METR_LA,self).__init__()
        assert (category == 'train' or category == 'valid' or category=='test')
        
        self.data = dict()
        self.get_data(category=category)
        
        if scaler:
            scaler.fit(self.train_npz['x'][...,0]) 
            self.data['x'][...,0] = scaler.transform(self.data['x'][...,0])
    
    def get_data(self,category):
        if category == "train":
            self.data['x'],self.data['y'] = self.train_npz['x'], self.train_npz['y']
        elif category == "valid":
            self.data['x'],self.data['y'] = self.valid_npz['x'], self.valid_npz['y']
        else:
            self.data['x'],self.data['y'] = self.test_npz['x'], self.test_npz['y']

    @property
    def train_npz(self):
        return np.load('_metr_la/train.npz')

    @property
    def valid_npz(self):
        return np.load('_metr_la/valid.npz')

    @property
    def test_npz(self):
        return np.load('_metr_la/test.npz')
    
    def __getitem__(self, index):
        return self.data['x'][index],self.data['y'][index]

    def __len__(self):
        return self.data['x'].shape[0]