import argparse
from easydict import EasyDict as esDict

class BaseConfig():

    def __init__(self) -> None:
        super(BaseConfig,self).__init__()
        
        self.DEVICE = esDict()
        self.DEVICE.VGPUS = "0"      # visible gpus
        self.DEVICE.WHICH = "cuda:0" # which is used

        self.HPARAM = esDict()
        self.HPARAM.RAND_SEED = 3407 # torch.manual_seed(3407) is all you need
        self.HPARAM.MAX_EPOCHS= 10
        self.HPARAM.BATCH_SIZE = 64
        self.HPARAM.NUM_WORKERS = 0

        self.HPARAM.WD = 1e-3 # weight decay
        self.HPARAM.LR = 1e-3 # learning rate
        self.HPARAM.LRD = 0.97 # learning rate decay
        self.HPARAM.CLIP = 3 # gradient clipping

        self.DATA = esDict()
        self.MODS = esDict()

    def make_parser(self):
        return argparse.ArgumentParser()

    def disp_config(self):
        print(f"\n{self.__dict__}\n")