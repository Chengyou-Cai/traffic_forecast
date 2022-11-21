import os
import pytorch_lightning as pl

from config import Config
from engine import DataEngine,SystemGWNet

from dataset.scaler import StandardScaler

from pytorch_lightning.callbacks import ModelCheckpoint

def init_config():
    cfg = Config().parse()
    pl.seed_everything(cfg.rand_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    return cfg

def fit_model(cfg,scaler=None,data_engine=None,trainer=None):
    print("start fitting...")
    
    data_engine.setup(stage='fit')
    model_system = SystemGWNet(cfg=cfg,scaler=scaler)
    trainer.fit(model=model_system,datamodule=data_engine)

def test_model(cfg,scaler=None,data_engine=None,trainer=None):
    print("start testing...")
    print(ckpt_callback1.best_model_path)
    
    data_engine.setup("test")
    model_system = SystemGWNet.load_from_checkpoint(
        ckpt_callback1.best_model_path,
        cfg=cfg,scaler=scaler
        )
    trainer.test(model=model_system,datamodule=data_engine)

def main():
    cfg = init_config(); print(cfg,"\n"); 

    data_file_paths = {
        "train":'_metr_la/train.npz',
        "valid":'_metr_la/valid.npz',
        "test":'_metr_la/test.npz'
    }

    scaler = StandardScaler()

    data_engine = DataEngine(
        cfg=cfg,
        scaler=scaler,
        data_file_paths=data_file_paths
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        auto_select_gpus=True,
        min_epochs=1,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[ckpt_callback1]
        # precision=64
    )

    fit_model(cfg,scaler=scaler,data_engine=data_engine,trainer=trainer)
    test_model(cfg,scaler=scaler,data_engine=data_engine,trainer=trainer)

if __name__ == "__main__":

    ckpt_callback1 = ModelCheckpoint(
        save_top_k=3,
        monitor="gwnet_valid_loss",
        mode="min",
        dirpath="ckeckpoint/gwnet/",
        filename="metr_60min_{epoch:02d}_{gwnet_valid_loss:.2f}",
        save_last=True
    ) 
    main()