import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torch.utils.data import DataLoader

class DataEngine(pl.LightningDataModule):
    
    def __init__(self, cfg) -> None:
        super(DataEngine, self).__init__()
        self.cfg = cfg

        self.num_workers = cfg.num_workers
        self.batch_size = cfg.batch_size

    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage='fit',train_set=None,valid_set=None,test_set=None) -> None:
        assert (stage == 'fit' or stage == 'test')
        if stage == 'fit':
            self.train_set = train_set
            self.valid_set = valid_set
        elif stage == 'test':
            self.test_set = test_set

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
from models.gwavenet_mine import GWNet
from common.tool import calc_metrics

class SystemV1(pl.LightningModule):

    def __init__(self,cfg,scaler) -> None:
        super(SystemV1,self).__init__()
        self.cfg = cfg
        self.scaler = scaler

        self.model = GWNet.from_args(args=cfg,device=self.device)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),lr=self.cfg.lr,weight_decay=self.cfg.wd)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:self.cfg.lrd**epoch)
        return {'optimizer':optimizer,'lr_scheduler': scheduler}

    def training_step(self,batch,batch_idx):
        
        x, y = batch # metr-la (bs,12,207,2)
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

    def on_after_backward(self) -> None:
        
        if self.cfg.clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip)

    def training_epoch_end(self, outputs) -> None:
        
        avg_loss = torch.stack([out['loss'] for out in outputs]).mean()
        avg_mape = torch.stack([out['mape'] for out in outputs]).mean()
        avg_rmse = torch.stack([out['rmse'] for out in outputs]).mean()
        logd = {
            "train_epoch_avg_loss":avg_loss,
            "train_epoch_avg_mape":avg_mape,
            "train_epoch_avg_rmse":avg_rmse,
        }
        self.log_dict(logd)
    
    @torch.no_grad()
    def _shared_eval_step(self,batch):
        
        x ,y = batch
        x, y = x.transpose(1,3), y.transpose(1,3)

        y_speed = y[:,0,:,:] # (bs,207,12)
        ###
        x = F.pad(x,(1,0,0,0))
        outputs = self.model(x)
        outputs = outputs.transpose(1,3)
        preds = self.scaler.inverse_transform(outputs)
        preds = torch.clamp(preds, min=0., max=70.)
        preds = preds.squeeze(1)
        mae, mape, rmse = calc_metrics(preds=preds,labels=y_speed)
        ###
        return mae,mape,rmse
    

    def validation_step(self, batch, batch_idx):

        loss,mape,rmse = self._shared_eval_step(batch)
        
        logd = {
            "valid_step_loss":loss,
            "valid_step_mape":mape,
            "valid_step_rmse":rmse,
        }
        self.log_dict(logd)

    def test_step(self, batch, batch_idx):
        
        loss,mape,rmse = self._shared_eval_step(batch)
        
        logd = {
            "test_step_loss":loss,
            "test_step_mape":mape,
            "test_step_rmse":rmse,
        }
        self.log_dict(logd)

