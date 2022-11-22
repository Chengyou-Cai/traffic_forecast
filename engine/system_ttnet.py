import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

from torch.nn import functional as F
from common.tool import calc_metrics

from models.traffic_transformer import TrafficTransformer

class SystemTTNet(pl.LightningModule):

    def __init__(self,config,scaler):
        super(SystemTTNet,self).__init__()
        self.config = config
        self.scaler = scaler
        print("load traffic transformer")
        self.model = TrafficTransformer(
            config=config,
            device=config.use_gpu,
            d_model=config.d_model,
            feat_planes=config.feat_planes, #
            gcn_planes=config.gcn_planes, #
            drop_prob=config.drop_prob
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
        tgt = tgt[:,[0],:,:] # (bs,1,207,12)
        ###
        out = self.model(src=src.float(),tgt=tgt.float()) # (bs,1,207,seq_len)
        
        preds = self.scaler.inverse_transform(out)
        assert preds.shape[1] == 1
        preds = preds.squeeze(1) # (bs,207,12)
        mae, rmse, mape = calc_metrics(preds=preds,labels=y_speed)

        logd = {
            "train_loss":mae,
            "train_rmse":rmse,
            "train_mape":mape,
        }

        self.log_dict(logd,on_step=False,on_epoch=True,prog_bar=True)

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
        tgt = x[...,[-1]][:,[0],:,:] # (bs,1,207,1)
        ### 
        for i in range(1,y.size(3)+1):
            out = self.model(src=src.float(),tgt=tgt.float()) # (bs,1,207,i)
            tgt = torch.concat((tgt,out[...,[-1]]),dim=3) # (bs,1,207,i+1)
        tgt = tgt[:,:,:,1:] # (bs,1,207,12)
        preds = self.scaler.inverse_transform(tgt)
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