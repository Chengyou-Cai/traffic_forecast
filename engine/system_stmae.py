import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

from torch.nn import functional as F
from common.tool import calc_metrics

from models.stmae import STMAE_pretrain

class System_P(pl.LightningModule):
    
    def __init__(self, config, scaler) -> None:
        super(System_P,self).__init__()
        self.config = config
        self.scaler = scaler

        print("load stmae_pretrain ...")
        self.model = STMAE_pretrain(config=config)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),lr=self.config.lr,weight_decay=self.config.wd)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:self.config.lrd**epoch)
        return {'optimizer':optimizer,'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch # metr-la (bs,12,207,2)
        x = x.transpose(1,3) # (bs,2,207,12)

        x_feat1 = x[:,0,:,:]
        x_feat2 = x[:,1,:,:]

        outs = self.model(x.float()) # (bs,12,207,?) ?=2
        outs = outs.transpose(1,3) # (bs,2,207,12)

        pred_feat1 = self.scaler.inverse_transform(outs[:,[0],:,:]) # (bs,1,207,12)
        pred_feat1 = pred_feat1.squeeze(1) # (bs,207,12)
        pred_feat2 = outs[:,1,:,:]

        f1_mae, f1_rmse, f1_mape = calc_metrics(preds=pred_feat1,labels=x_feat1)
        f2_mae, f2_rmse, f2_mape = calc_metrics(preds=pred_feat2,labels=x_feat2)

        mae = (f1_mae+f2_mae)/2
        rmse = (f1_rmse+f2_rmse)/2
        mape = (f1_mape+f2_mape)/2

        logd = {
            "p_train_loss":mae,
            "p_train_rmse":rmse,
            "p_train_mape":mape,
        }
        self.log_dict(logd,on_step=False,on_epoch=True,prog_bar=True)

        return {"loss":mae,"rmse":rmse,"mape":mape}

    def on_after_backward(self) -> None:
        
        if self.config.clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

    @torch.no_grad()
    def _shared_eval_step(self,batch):
        
        x ,y = batch
        x = x.transpose(1,3) # (bs,2,207,12)

        x_feat1 = x[:,0,:,:]
        x_feat2 = x[:,1,:,:]
        ###
        
        outs = self.model(x.float())# (bs,12,207,?) ?=2
        outs = outs.transpose(1,3)  # (bs,2,207,12)
        
        pred_feat1 = self.scaler.inverse_transform(outs[:,[0],:,:]) # (bs,1,207,12)
        pred_feat1 = pred_feat1.squeeze(1) # (bs,207,12)
        pred_feat2 = outs[:,1,:,:]
        
        pred_feat1 = torch.clamp(pred_feat1, min=0., max=70.)
        pred_feat2 = torch.clamp(pred_feat2, min=0., max=1.)
        
        f1_mae, f1_rmse, f1_mape = calc_metrics(preds=pred_feat1,labels=x_feat1)
        f2_mae, f2_rmse, f2_mape = calc_metrics(preds=pred_feat2,labels=x_feat2)

        mae = (f1_mae+f2_mae)/2
        rmse = (f1_rmse+f2_rmse)/2
        mape = (f1_mape+f2_mape)/2

        ###
        return mae,rmse,mape

    def validation_step(self, batch, batch_idx):

        mae,rmse,mape = self._shared_eval_step(batch)
        
        logd = {
            "p_valid_loss":mae,
            "p_valid_rmse":rmse,
            "p_valid_mape":mape,
        }
        self.log_dict(logd,on_epoch=True,prog_bar=True) # default on_step=False,on_epoch=True

    def test_step(self, batch, batch_idx):
        
        mae,rmse,mape = self._shared_eval_step(batch)
        
        logd = {
            "p_test_loss":mae,
            "p_test_rmse":rmse,
            "p_test_mape":mape,
        }
        self.log_dict(logd,on_epoch=True,prog_bar=True) # default on_step=False,on_epoch=True

from models.stmae import STMAE_finetune

class System_F(pl.LightningModule):

    def __init__(self, config, scaler) -> None:
        super(System_F,self).__init__()

        self.config = config
        self.scaler = scaler

        print("load stmae_finetune ...")
        self.model = STMAE_finetune(config)

        if self.config.load_pre:
            print("load pretrain param ...")
            self.load_param()

    def load_param(self):
        model_dict = self.model.state_dict()
        
        param_dict = torch.load(self.config.param_path)['state_dict']
        state_dict = {k: v for k, v in param_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)

        self.model.load_state_dict(model_dict)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),lr=self.config.lr,weight_decay=self.config.wd)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:self.config.lrd**epoch)
        return {'optimizer':optimizer,'lr_scheduler': scheduler}

    def training_step(self,batch,batch_idx):
        x, y = batch # metr-la (bs,12,207,2)
        x, y = x.transpose(1,3), y.transpose(1,3) # (bs,2,207,12)

        y_speed = y[:,0,:,:] # (bs,207,12)
        if y_speed.max() == 0 : 
            return None
        ###
        outs = self.model(x.float()) # (bs,12,207,?)
        outs = outs.transpose(1,3) # (bs,?,207,12) ?=1  ###
        
        pred = self.scaler.inverse_transform(outs)
        assert  pred.shape[1] == 1
        pred = pred.squeeze(1) # (bs,207,12)

        mae, rmse, mape = calc_metrics(preds=pred,labels=y_speed)
        ###

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
        
        x ,y = batch
        x, y = x.transpose(1,3), y.transpose(1,3)

        y_speed = y[:,0,:,:] # (bs,207,12)
        ###
        outs = self.model(x.float())
        outs = outs.transpose(1,3)
        
        pred = self.scaler.inverse_transform(outs)
        pred = torch.clamp(pred, min=0., max=70.)
        pred = pred.squeeze(1)
        mae, rmse, mape = calc_metrics(preds=pred,labels=y_speed)
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