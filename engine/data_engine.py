import pytorch_lightning as pl

from torch.utils.data import DataLoader
from dataset.traffic_dataset import TrafficDataset

class DataEngine(pl.LightningDataModule):
    
    def __init__(self, cfg, scaler, data_file_paths) -> None:
        super(DataEngine, self).__init__()
        self.cfg = cfg
        self.scaler = scaler
        self.data_file_paths = data_file_paths

        self.num_workers = cfg.num_workers
        self.batch_size = cfg.batch_size

    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage='fit') -> None:
        assert (stage == 'fit' or stage == 'test')
        if stage == 'fit':
            self.train_set = TrafficDataset(data_file_paths=self.data_file_paths, category='train', scaler=self.scaler)
            self.valid_set = TrafficDataset(data_file_paths=self.data_file_paths, category='valid', scaler=self.scaler)
        elif stage == 'test':
            self.test_set = TrafficDataset(data_file_paths=self.data_file_paths, category='train', scaler=self.scaler)

    def train_dataloader(self):
        print(self.train_set)
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )