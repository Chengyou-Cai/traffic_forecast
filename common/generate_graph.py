import os
import pickle
import numpy as np
import pandas as pd

def generate_adjmatrix(distances_df, sensor_ids,normalized_k=0.1):
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    for row in distances_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx

def generate_pkl(fdir='_metr_la/sensor_graph',fname='distances_la_2012.csv'):
    with open(os.path.join(fdir,'graph_sensor_ids.txt')) as f:
        sensor_ids = f.read().strip().split(',')
    distances_df = pd.read_csv(os.path.join(fdir,fname), dtype={'from': 'str', 'to': 'str'})
    _, sensor_id_to_ind, adj_mx = generate_adjmatrix(distances_df, sensor_ids) # normalized_k=0.1

    with open(os.path.join(fdir,'adjmatrix.pkl'), 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)

if __name__ == "__main__":
    generate_pkl()