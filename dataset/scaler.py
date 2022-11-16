
class StandardScaler():
    
    def __init__(self,mean=None,std=None):
        super(StandardScaler,self).__init__()
        self.mean = mean
        self.std = std

    def fit(self,data):
        self.mean = data.mean()
        self.std = data.std()

    def transform(self,data,fill_zeros=True):
        if fill_zeros:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean)/self.std

    def fit_transform(self,data,fill_zeros=True):
        self.mean = data.mean()
        self.std = data.std()
        if fill_zeros:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean)/self.std

    def inverse_transform(self,data):
        return (data * self.std) + self.mean