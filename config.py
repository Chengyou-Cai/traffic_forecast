from ast import arg
import os
import argparse

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']
class Config():
    
    def __init__(self) -> None:
        self.ap = argparse.ArgumentParser()
        
        # machine environ
        self.ap.add_argument('--gpus', default='0,1', type=str, help='')
        
        # hyper parameter
        self.ap.add_argument('--rand_seed', type=int, default=0)
        self.ap.add_argument('--max_epochs', type=int, default=100, help='can be changed depending on your machine')
        self.ap.add_argument('--batch_size', type=int, default=128, help='can be changed depending on your machine')
        self.ap.add_argument('--num_workers', type=int, default=2, help='can be changed depending on your machine')
        
        self.ap.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.ap.add_argument('--wd', type=float, default=0.001, help='weight decay')
        self.ap.add_argument('--lrd', type=float, default=0.95, help='learning rate decay')
        self.ap.add_argument('--clip', type=int, default=3, help='Gradient Clipping')

        # file path
        self.ap.add_argument('--pkl_fn', type=str, default='/workspace/traffic_v2/_metr_la/sensor_graph/adjmatrix.pkl')

        # others
        self.ap.add_argument('--adjtype', type=str, default='doubletransition', help='adj type', choices=ADJ_CHOICES)
        self.ap.add_argument('--do_graph_conv', action='store_true',help='whether to add graph convolution layer')
        self.ap.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
        self.ap.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
        self.ap.add_argument('--randomadj', action='store_true',help='whether random initialize adaptive adj')


    def parse(self):
        self.cfg = self.ap.parse_args()

        return self.cfg