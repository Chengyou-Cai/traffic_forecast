import numpy as np
from torch.utils.data import Dataset

class StandardScaler():
    
    def __init__(self,mean=None,std=None):
        super(StandardScaler,self).__init__()
        self.mean = mean
        self.std = std

    def fit(self,data):
        self.mean = data.mean()
        self.std = data.std()

    def transform(self,data):
        return (data - self.mean)/self.std

    def inverse_transform(self,data):
        return (data * self.std) + self.mean


class METR_LA(Dataset):

    def __init__(self,category="train",scaler=None) -> None:
        super(METR_LA,self).__init__()
        assert (category == 'train' or category == 'valid' or category=='test')

        self.npz = np.load('/workspace/traffic_v2/_metr_la/train.npz')
        if scaler:
            scaler.fit(self.npz['x'][...,0]) 

        if category == 'valid':
            self.npz = np.load('/workspace/traffic_v2/_metr_la/valid.npz')
        elif category == 'test':
            self.npz = np.load('/workspace/traffic_v2/_metr_la/test.npz')

        self.data = {'x':self.npz['x'], 'y':self.npz['y']}
        
        if scaler:
            self.data['x'][...,0] = scaler.transform(self.npz['x'][...,0])

    def __getitem__(self, index):
        return self.data['x'][index],self.data['y'][index]

    def __len__(self):
        return self.data['x'].shape[0]