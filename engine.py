import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from dataset.metr_la import METR_LA

class DataEngine(pl.LightningDataModule):
    
    def __init__(self, cfg, scaler=None) -> None:
        super(DataEngine, self).__init__()
        self.cfg = cfg
        self.scaler = scaler

        self.num_workers = cfg.num_workers
        self.batch_size = cfg.batch_size

    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage='fit') -> None:
        assert (stage == 'fit' or stage == 'test')
        if stage == 'fit':
            self.train_set = METR_LA(category='train',scaler = self.scaler)
            self.valid_set = METR_LA(category='valid',scaler = self.scaler)
        elif stage == 'test':
            self.test_set = METR_LA(category='test',scaler = self.scaler)

    def train_dataloader(self):
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

import torch.optim as optim

from torch.nn import functional as F
from models.gwavenet import GWNet
from common.tool import make_graph_inputs,calc_metrics

class SystemV1(pl.LightningModule):

    def __init__(self,cfg,scaler=None) -> None:
        super(SystemV1,self).__init__()
        self.cfg = cfg
        self.scaler = scaler

        aptinit, supports = make_graph_inputs(args=cfg,device=self.device)
        self.model = GWNet.from_args(
            args=cfg,
            device=self.device,
            supports=supports,
            aptinit=aptinit
            )

        self.loss_train = None
        self.loss_valid = None
        self.loss_test = None

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),lr=self.cfg.lr,weight_decay=self.cfg.wd)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:self.cfg.lrd**epoch)
        return {'optimizer':optimizer,'lr_scheduler': scheduler}

    def training_step(self,batch,batch_idx):
        
        x, y = batch
        x, y = x.transpose(1,3), y.transpose(1,3) # (bs,2,207,12)

        y_speed = y[:,0,:,:] # (bs,207,12)
        if y_speed.max() == 0 : 
            return None
        ###
        x = F.pad(x,(1,0,0,0)) # (bs,2,207,13)
        outputs = self.model(x) # (bs,12,207,?)
        outputs = outputs.transpose(1,3) # (bs,?,207,12) ?=1
        preds = self.scaler.inverse_transform(outputs)
        assert  preds.shape[1] == 1
        preds = preds.squeeze(1) # (bs,207,12)
        mae, mape, rmse = calc_metrics(preds=preds,labels=y_speed)
        ###
        return {"loss":mae,"mape":mape,"rmse":rmse}

    def training_epoch_end(self, outputs) -> None:
        pass



