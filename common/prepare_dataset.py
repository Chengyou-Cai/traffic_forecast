import os
import numpy as np
import pandas as pd
from rich.progress import track

def generate_graph_seq2seq_io_data(
    df, x_offsets, y_offsets,
    add_time_in_day=True,
    add_day_in_week=False
):
    num_samples,  num_nodes = df.shape # pems bay (52116,325)
    data = np.expand_dims(df.values, axis=-1) # expand_dims (52116, 325, 1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]"))/np.timedelta64(1, "D") # (52116,)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)) # (52116, 325, 1)
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)
    
    data = np.concatenate(data_list,axis=-1) # concat data&time (52116, 325, 2)

    x, y = [], []
    min_t = abs(min(x_offsets)) # 11
    max_t = abs(num_samples - abs(max(y_offsets))) # 52104

    for t in track(range(min_t, max_t)):
        x_t = data[t + x_offsets,...]
        y_t = data[t + y_offsets,...]
        x.append(x_t)
        y.append(y_t)

    print("stacking data ...")
    x = np.stack(x,axis=0) # (52093, 12, 325, 2)
    y = np.stack(y,axis=0) # (52093, 12, 325, 2)

    return x, y

def generate_train_valid_test(fdir='_metr_la',fname='metr-la.h5',seq_x_len=12,seq_y_len=12):
    
    df = pd.read_hdf(os.path.join(fdir, fname)) # pemsbay data shape (52116,325)
    
    # 预测窗口 1 hour = 12 * 5 min
    x_offsets = np.sort(np.arange(-(seq_x_len-1), 1, 1)) # array([-11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0])
    y_offsets = np.sort(np.arange(1, (seq_y_len+1), 1)) # array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    num_samples = x.shape[0] # 52093
    num_test = round(num_samples*0.2) # 10419
    num_train = round(num_samples*0.7) # 36465
    num_valid = num_samples - num_train - num_test # 5209

    print("splitting data ...")
    # train
    x_train, y_train = x[:num_train], y[:num_train] # (36465,12,325,2)
    # valid
    x_valid, y_valid = (
        x[num_train: num_train + num_valid],
        y[num_train: num_train + num_valid],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    print("zipping data ...")
    for cat in ["train","valid","test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(fdir, f"{fdir}_x{seq_x_len}y{seq_y_len}_{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fdir', type=str, default='_metr_la',help="data file dir")
    parser.add_argument('--fname', type=str, default='metr-la.h5',help="data file name")
    parser.add_argument('--seq_x_len', type=int, default=12,help="sequence x length")
    parser.add_argument('--seq_y_len', type=int, default=12,help="sequence y length")
    args = parser.parse_args()

    print(args)

    generate_train_valid_test(
        fdir=args.fdir,fname=args.fname,
        seq_x_len=args.seq_x_len,
        seq_y_len=args.seq_y_len
        )
