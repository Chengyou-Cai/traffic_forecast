import os
import pytorch_lightning as pl

from config import Config
from engine import DataEngine,SystemV1

def init_config():
    cfg = Config().parse()
    pl.seed_everything(cfg.rand_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    return cfg

def fit_model(cfg):
    data_engine = DataEngine(cfg=cfg)
    data_engine.setup(stage='fit')
    
    model_system = SystemV1(cfg=cfg)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        auto_select_gpus=True,
        min_epochs=1,
        max_epochs=cfg.max_epochs,
        precision=16
    )
    trainer.fit(model=model_system,datamodule=data_engine)

def test_model(cfg):
    pass

def main():
    pass

if __name__ == "__main__":
    main()