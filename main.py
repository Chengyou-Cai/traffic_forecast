import os
import pytorch_lightning as pl

from config import Config
from engine import DataEngine,SystemV1

from dataset.metr_la import METR_LA
from dataset.scaler import StandardScaler

def init_config():
    cfg = Config().parse()
    pl.seed_everything(cfg.rand_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    return cfg

def fit_model(cfg,train_set,valid_set,scaler=None):
    print("start fitting...")
    
    data_engine = DataEngine(cfg=cfg)
    
    data_engine.setup(stage='fit',train_set=train_set,valid_set=valid_set)
    
    model_system = SystemV1(cfg=cfg,scaler=scaler)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        auto_select_gpus=True,
        min_epochs=1,
        max_epochs=cfg.max_epochs,
        precision=16,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model=model_system,datamodule=data_engine)

def test_model(cfg,test_set,scaler=None):
    print("start testing...")

def main():
    cfg = init_config()
    print(cfg,"\n")

    scaler = StandardScaler()
    train_set = METR_LA(category='train',scaler = scaler)
    valid_set = METR_LA(category='valid',scaler = scaler)
    test_set = METR_LA(category='test',scaler = scaler)

    fit_model(cfg,train_set,valid_set,scaler=scaler)
    test_model(cfg,test_set)

if __name__ == "__main__":
    main()