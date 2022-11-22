import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

from torch.nn import functional as F
from common.tool import calc_metrics

from models.gwavenet_mine import GWNet

class SystemGWNet(pl.LightningModule):

    def __init__(self,cfg,scaler) -> None:
        super(SystemGWNet,self).__init__()
        self.cfg = cfg
        self.scaler = scaler

        print("load graph wavenet")
        self.model = GWNet.from_args(args=cfg,device=cfg.use_gpu)

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
            "train_loss":mae,
            "train_rmse":rmse,
            "train_mape":mape,
        }

        self.log_dict(logd,on_step=False,on_epoch=True,prog_bar=True)

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
            "valid_loss":mae,
            "valid_rmse":rmse,
            "valid_mape":mape,
        }
        self.log_dict(logd,on_epoch=True,prog_bar=True) # default on_step=False,on_epoch=True

    def test_step(self, batch, batch_idx):
        
        mae,rmse,mape = self._shared_eval_step(batch)
        
        logd = {
            "test_loss":mae,
            "test_rmse":rmse,
            "test_mape":mape,
        }
        self.log_dict(logd,on_epoch=True,prog_bar=True) # default on_step=False,on_epoch=True