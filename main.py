import os
import pytorch_lightning as pl

from config import Config
from engine import DataEngine,SystemV1

from dataset.scaler import StandardScaler

def init_config():
    cfg = Config().parse()
    pl.seed_everything(cfg.rand_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    return cfg

def fit_model(cfg,scaler=None,data_file_paths=None):
    print("start fitting...")
    
    data_engine = DataEngine(
        cfg=cfg,
        scaler=scaler,
        data_file_paths=data_file_paths
        )
    
    data_engine.setup(stage='fit')
    
    model_system = SystemV1(cfg=cfg,scaler=scaler)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        auto_select_gpus=True,
        min_epochs=1,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=1,
        # precision=64
    )
    trainer.fit(model=model_system,datamodule=data_engine)

def test_model(cfg,scaler=None,data_file_paths=None):
    print("start testing... not completed")

def main():
    cfg = init_config()
    print(cfg,"\n")

    metr_la_files = {
        "train":'_metr_la/train.npz',
        "valid":'_metr_la/valid.npz',
        "test":'_metr_la/test.npz'
    }

    scaler = StandardScaler()

    fit_model(cfg,scaler=scaler,data_file_paths=metr_la_files)
    test_model(cfg,scaler=scaler,data_file_paths=metr_la_files)

if __name__ == "__main__":
    main()