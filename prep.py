import os
from common.prepare_adjmpkl import generate_adjmpkl
from common.prepare_dataset import generate_train_valid_test

def prepare_data(config):
    dist_file_path = config.DATA.DIST_FILE_PATH.as_posix()
    detr_file_path = config.DATA.DETR_FILE_PATH.as_posix()
    generate_adjmpkl(*os.path.split(dist_file_path))
    print("adjmatrix generated !")

    generate_train_valid_test(
        *os.path.split(detr_file_path),
        dsname=config.DATA.NAME,
        seq_x_len=config.DATA.seq_x_len,
        seq_y_len=config.DATA.seq_y_len)
    print("3datasets generated !")

if __name__ == "__main__":
    from _exps.metr_gwnet_config import Metr_GWnet_Config
    config1 = Metr_GWnet_Config()
    config1.disp_config()
    prepare_data(config1)

    from _exps.pems_gwnet_config import Pems_GWnet_Config
    config2 = Pems_GWnet_Config()
    config2.disp_config()
    prepare_data(config2)
