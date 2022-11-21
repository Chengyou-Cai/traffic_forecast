import torch
import torch.nn as nn
import torch.optim as optim
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

import torch.optim as optim

from torch.nn import functional as F
from models.gwavenet_mine import GWNet
from common.tool import calc_metrics

class SystemGWNet(pl.LightningModule):

    def __init__(self,cfg,scaler) -> None:
        super(SystemGWNet,self).__init__()
        self.cfg = cfg
        self.scaler = scaler

        print("load graph wavenet")
        self.model = GWNet.from_args(args=cfg,device="cuda:0")

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
        outputs = self.model(x.float()) # (bs,12,207,?)
        outputs = outputs.transpose(1,3) # (bs,?,207,12) ?=1  ###
        preds = self.scaler.inverse_transform(outputs)        ### wrong inverse_transform?
        assert  preds.shape[1] == 1
        preds = preds.squeeze(1) # (bs,207,12)
        mae, rmse, mape = calc_metrics(preds=preds,labels=y_speed)
        ###

        logd = {
            "gwnet_train_loss":mae,
            "gwnet_train_rmse":rmse,
            "gwnet_train_mape":mape,
        }

        self.log_dict(logd,on_step=True,on_epoch=True,prog_bar=True)

        return {"loss":mae,"rmse":rmse,"mape":mape}

    def on_after_backward(self) -> None:
        
        if self.cfg.clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip)
    
    @torch.no_grad()
    def _shared_eval_step(self,batch):
        
        x ,y = batch
        x, y = x.transpose(1,3), y.transpose(1,3)

        y_speed = y[:,0,:,:] # (bs,207,12)
        ###
        x = F.pad(x,(1,0,0,0))
        outputs = self.model(x.float())
        outputs = outputs.transpose(1,3)
        preds = self.scaler.inverse_transform(outputs)
        preds = torch.clamp(preds, min=0., max=70.)
        preds = preds.squeeze(1)
        mae, rmse, mape = calc_metrics(preds=preds,labels=y_speed)
        ###
        return mae,rmse,mape
    

    def validation_step(self, batch, batch_idx):

        mae,rmse,mape = self._shared_eval_step(batch)
        
        logd = {
            "gwnet_valid_loss":mae,
            "gwnet_valid_rmse":rmse,
            "gwnet_valid_mape":mape,
        }
        self.log_dict(logd,on_epoch=True,prog_bar=True) # default on_step=False,on_epoch=True

    def test_step(self, batch, batch_idx):
        
        mae,rmse,mape = self._shared_eval_step(batch)
        
        logd = {
            "gwnet_test_loss":mae,
            "gwnet_test_rmse":rmse,
            "gwnet_test_mape":mape,
        }
        self.log_dict(logd,on_epoch=True,prog_bar=True) # default on_step=False,on_epoch=True

from models.traffic_transformer import TrafficTransformer

class SystemTTNet(pl.LightningModule):

    def __init__(self,config,scaler):
        super(SystemTTNet,self).__init__()
        self.config = config
        self.scaler = scaler
        print("load traffic transformer")
        self.model = TrafficTransformer(
            config=config,
            device='cuda:0',
            d_model=128
        )
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),lr=self.config.lr,weight_decay=self.config.wd)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:self.config.lrd**epoch)
        return {'optimizer':optimizer,'lr_scheduler': scheduler}

    def training_step(self,batch,batch_idx):
        
        x, y = batch # metr-la (bs,12,207,2)

        ### !!! 放在外面 
        x, y = x.permute(0,3,2,1),y.permute(0,3,2,1) # (bs,2,207,12)
        y_speed = y[:,0,:,:] # (bs,207,12)
        if y_speed.max() == 0 : 
            return None

        ### training
        src = x # (bs,2,207,12)
        tgt = torch.concat((x[...,[-1]],y[...,:-1]),dim=3) # (bs,2,207,12)
        ###
        out = self.model(src=src.float(),tgt=tgt.float()) # (bs,2,207,seq_len)
        out = out[:,[0],:,:] # (bs,1,207,12)
        preds = self.scaler.inverse_transform(out)
        
        assert preds.shape[1] == 1
        preds = preds.squeeze(1) # (bs,207,12)
        mae, rmse, mape = calc_metrics(preds=preds,labels=y_speed)

        logd = {
            "ttnet_train_loss":mae,
            "ttnet_train_rmse":rmse,
            "ttnet_train_mape":mape,
        }

        self.log_dict(logd,on_step=True,on_epoch=True,prog_bar=True)

        return {"loss":mae,"rmse":rmse,"mape":mape}

    def on_after_backward(self) -> None:
        if self.config.clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
    
    @torch.no_grad()
    def _shared_eval_step(self,batch):
        
        x ,y = batch # metr-la (bs,12,207,2)
        x, y = x.permute(0,3,2,1),y.permute(0,3,2,1) # (bs,2,207,12)
        y_speed = y[:,0,:,:] # (bs,207,12)
        
        ### eval
        src = x # (bs,2,207,12)
        tgt = x[...,[-1]] # (bs,2,207,1)
        ### 
        for i in range(1,y.size(3)+1):
            out = self.model(src=src.float(),tgt=tgt.float()) # (bs,2,207,i)
            tgt = torch.concat((tgt,out[...,[-1]]),dim=3) # (bs,2,207,i+1)
        tgt = tgt[:,[0],:,1:] # (bs,1,207,12)
        preds = self.scaler.inverse_transform(tgt)
        preds = torch.clamp(preds, min=0., max=70.)
        preds = preds.squeeze(1)
        mae, rmse, mape = calc_metrics(preds=preds,labels=y_speed)
        ###
        return mae,rmse,mape

    def validation_step(self, batch, batch_idx):

        mae,rmse,mape = self._shared_eval_step(batch)
        
        logd = {
            "ttnet_valid_loss":mae,
            "ttnet_valid_rmse":rmse,
            "ttnet_valid_mape":mape,
        }
        self.log_dict(logd,on_epoch=True,prog_bar=True) # default on_step=False,on_epoch=True

    def test_step(self, batch, batch_idx):
        
        mae,rmse,mape = self._shared_eval_step(batch)
        
        logd = {
            "ttnet_test_loss":mae,
            "ttnet_test_rmse":rmse,
            "ttnet_test_mape":mape,
        }
        self.log_dict(logd,on_epoch=True,prog_bar=True) # default on_step=False,on_epoch=True