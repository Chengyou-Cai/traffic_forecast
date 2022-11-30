import os
import pytorch_lightning as pl

from config import MAEConfig
from engine.data_engine import DataEngine
from engine.system_stmae import System_P,System_F

from dataset.scaler import StandardScaler

from pytorch_lightning.callbacks import ModelCheckpoint

def init_config():
    cfg = MAEConfig().parse()
    pl.seed_everything(cfg.rand_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    return cfg

def fit_model(trainer,model_system,data_engine):
    print("start fit...")
    data_engine.setup(stage='fit')
    trainer.fit(model=model_system,datamodule=data_engine)
    print("finish fit...")

def test_model(trainer,model_system,data_engine):
    print("start test...")
    data_engine.setup(stage='test')
    trainer.test(model=model_system,datamodule=data_engine)
    print("finish test...")

def pretraining(config,scaler,data_engine,dir='pretrain'):
    print("start pretrain...")
    ckpt_callback1 = ModelCheckpoint(
        save_top_k=3,
        monitor="p_valid_loss",
        mode="min",
        dirpath=f"ckeckpoint/stmae/{dir}/",
        filename="metr_60min_{epoch:02d}_{p_valid_loss:.2f}",
        save_last=True
    ) 
    trainer_p = pl.Trainer(
        accelerator='gpu',
        devices=1,
        auto_select_gpus=True,
        min_epochs=1,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[ckpt_callback1]
        # precision=64
    )
    syst_p = System_P(config=config,scaler=scaler)
    fit_model(trainer=trainer_p,model_system=syst_p,data_engine=data_engine)
    best_p = System_P.load_from_checkpoint(
        ckpt_callback1.best_model_path,
        config=config,scaler=scaler
    )
    test_model(trainer=trainer_p,model_system=best_p,data_engine=data_engine)


def finetuning(config,scaler,data_engine,dir='finetune'):
    print("start finetune...")
    ckpt_callback2 = ModelCheckpoint(
        save_top_k=3,
        monitor="valid_loss",
        mode="min",
        dirpath=f"ckeckpoint/stmae/{dir}/",
        filename="metr_60min_{epoch:02d}_{valid_loss:.2f}",
        save_last=True
    ) 
    trainer_f = pl.Trainer(
        accelerator='gpu',
        devices=1,
        auto_select_gpus=True,
        min_epochs=1,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[ckpt_callback2]
        # precision=64
    )
    syst_f = System_F(config=config,scaler=scaler)
    fit_model(trainer=trainer_f,model_system=syst_f,data_engine=data_engine)
    best_f = System_P.load_from_checkpoint(
        ckpt_callback2.best_model_path,
        config=config,scaler=scaler
    )
    test_model(trainer=trainer_f,model_system=best_f,data_engine=data_engine)


def main():

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
    pretraining(config=cfg,scaler=scaler,data_engine=data_engine)


if __name__ == "__main__":

    cfg = init_config(); print(cfg,"\n"); 
    main()