import argparse

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']
class Config():
    
    def __init__(self) -> None:
        self.ap = argparse.ArgumentParser()
        
        # machine environ
        self.ap.add_argument('--gpus', type=str, default='0')
        
        # hyper parameter
        self.ap.add_argument('--rand_seed', type=int, default=3407,help="torch.manual_seed(3407) is all you need")
        self.ap.add_argument('--max_epochs', type=int, default=105, help='can be changed depending on your machine') # 
        self.ap.add_argument('--batch_size', type=int, default=128, help='can be changed depending on your machine') # 64
        self.ap.add_argument('--num_workers', type=int, default=0, help='can be changed depending on your machine')
        
        self.ap.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.ap.add_argument('--wd', type=float, default=0.001, help='weight decay')
        self.ap.add_argument('--lrd', type=float, default=0.97, help='learning rate decay') # improvements
        self.ap.add_argument('--clip', type=int, default=3, help='gradient clipping') # improvements
        self.ap.add_argument('--dropout', type=float, default=0.3, help='dropout')

        # file path
        self.ap.add_argument('--pkl_fn', type=str, default='_metr_la/sensor_graph/adjmatrix.pkl') ###########

        self.ap.add_argument('--cat_feat_gc', action='store_true',default=True)
        # others
        self.ap.add_argument('--do_graph_conv', action='store_true',default=True,help='whether to add graph convolution layer')
        self.ap.add_argument('--adjtype', type=str, default='doubletransition', help='adj type', choices=ADJ_CHOICES)
        self.ap.add_argument('--addaptadj', action='store_true', default=True, help='whether add adaptive adj')
        
        self.ap.add_argument('--randomadj', action='store_true',help='whether random initialize adaptive adj')
        self.ap.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
        
        self.ap.add_argument('--apt_size', type=int, default=10)

        self.ap.add_argument('--num_nodes', type=int, default=207, help='num of nodes') ###########
        self.ap.add_argument('--nhid', type=int, default=40, help='num of hidden layers') # improvements
        self.ap.add_argument('--in_dim', type=int, default=2)
        self.ap.add_argument('--seq_len', type=int, default=12) ###########


    def parse(self):
        self.cfg = self.ap.parse_args()

        return self.cfg