import argparse

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']
class MAEConfig():
    
    def __init__(self):
        self.ap = argparse.ArgumentParser()
        
        # file path
        self.ap.add_argument('--adjm_fn', type=str, default='_metr_la/sensor_graph/adjmatrix.pkl')
        self.ap.add_argument('--ckpt_fn', type=str, default='stmae')
        
        # gpus
        self.ap.add_argument('--gpus', type=str, default='0')
        self.ap.add_argument('--which_gpu', type=str, default='cuda:0')

        self.ap.add_argument('--load_param', action='store_true')
        self.ap.add_argument('--param_path', type=str, default='') #####
        # pre-training
        self.ap.add_argument('--s_mask_rate', type=float, default=0.15)
        self.ap.add_argument('--t_mask_rate', type=float, default=0.5)

        self.ap.add_argument('--seq_len', type=int, default=12)
        self.ap.add_argument('--nod_num', type=int, default=207) # metr-la
        self.ap.add_argument('--d_channels', type=int, default=2) # data channels
        self.ap.add_argument('--batch_size', type=int, default=128)

        self.ap.add_argument('--mlp_out_chans', type=int, default=64) ######## =
        self.ap.add_argument('--num_fc_layers', type=int, default=1)

        self.ap.add_argument('--drop_prob', type=float, default=0.15)

        self.ap.add_argument('--d_model', type=int, default=64) ######## =
        self.ap.add_argument('--nhead', type=int, default=4) ########
        self.ap.add_argument('--dim_ffn', type=int, default=64) ########
        self.ap.add_argument('--num_layers', type=int, default=2) ########

        # fine-tuning # gwnet hyper param
        self.ap.add_argument('--rand_seed', type=int, default=3407,help="torch.manual_seed(3407) is all you need")
        self.ap.add_argument('--max_epochs', type=int, default=150, help='can be changed depending on your machine') # 
        self.ap.add_argument('--batch_size', type=int, default=128, help='can be changed depending on your machine') # 64
        self.ap.add_argument('--num_workers', type=int, default=0, help='can be changed depending on your machine')
        
        self.ap.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.ap.add_argument('--wd', type=float, default=0.001, help='weight decay')
        self.ap.add_argument('--lrd', type=float, default=0.97, help='learning rate decay') # improvements
        self.ap.add_argument('--clip', type=int, default=3, help='gradient clipping') # improvements
        self.ap.add_argument('--dropout', type=float, default=0.3, help='dropout')

        self.ap.add_argument('--cat_feat_gc', action='store_true',default=True)
        # others
        self.ap.add_argument('--do_graph_conv', action='store_true',default=True,help='whether to add graph convolution layer')
        self.ap.add_argument('--adjtype', type=str, default='doubletransition', help='adj type', choices=ADJ_CHOICES)
        self.ap.add_argument('--addaptadj', action='store_true', default=True, help='whether add adaptive adj')
        
        self.ap.add_argument('--randomadj', action='store_true',help='whether random initialize adaptive adj')
        self.ap.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
        
        self.ap.add_argument('--apt_size', type=int, default=10)

        self.ap.add_argument('--nhid', type=int, default=40, help='num of hidden layers') # improvements
        self.ap.add_argument('--in_dim', type=int, default=2)
        
        self.ap.add_argument('--seq_x_len', type=int, default=12)
        self.ap.add_argument('--seq_y_len', type=int, default=12)


    def parse(self):
        self.config = self.ap.parse_args()
        return self.config


class Config():
    
    def __init__(self) -> None:
        self.ap = argparse.ArgumentParser()

        # file path
        self.ap.add_argument('--pkl_fn', type=str, default='_metr_la/sensor_graph/adjmatrix.pkl') ###########
        self.ap.add_argument('--ckpt_fn', type=str, default='ttnet') ###########


        # machine environ
        self.ap.add_argument('--gpus', type=str, default='0')
        self.ap.add_argument('--use_gpu', type=str, default='cuda:0')

        # ttnet hyper parameter
        self.ap.add_argument('--d_model', type=int, default=64)
        self.ap.add_argument('--feat_planes', type=int, default=16)
        self.ap.add_argument('--gcn_planes', type=int, default=16)
        self.ap.add_argument('--drop_prob', type=float, default=0.15)
        self.ap.add_argument('--num_layers', type=int, default=2)
        self.ap.add_argument('--dim_ffn', type=int, default=64)

        # hyper parameter
        self.ap.add_argument('--rand_seed', type=int, default=3407,help="torch.manual_seed(3407) is all you need")
        self.ap.add_argument('--max_epochs', type=int, default=150, help='can be changed depending on your machine') # 
        self.ap.add_argument('--batch_size', type=int, default=128, help='can be changed depending on your machine') # 64
        self.ap.add_argument('--num_workers', type=int, default=0, help='can be changed depending on your machine')
        
        self.ap.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.ap.add_argument('--wd', type=float, default=0.001, help='weight decay')
        self.ap.add_argument('--lrd', type=float, default=0.97, help='learning rate decay') # improvements
        self.ap.add_argument('--clip', type=int, default=3, help='gradient clipping') # improvements
        self.ap.add_argument('--dropout', type=float, default=0.3, help='dropout')

        self.ap.add_argument('--cat_feat_gc', action='store_true',default=True)
        # others
        self.ap.add_argument('--do_graph_conv', action='store_true',default=True,help='whether to add graph convolution layer')
        self.ap.add_argument('--adjtype', type=str, default='doubletransition', help='adj type', choices=ADJ_CHOICES)
        self.ap.add_argument('--addaptadj', action='store_true', default=True, help='whether add adaptive adj')
        
        self.ap.add_argument('--randomadj', action='store_true',help='whether random initialize adaptive adj')
        self.ap.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
        
        self.ap.add_argument('--apt_size', type=int, default=10)

        self.ap.add_argument('--nod_num', type=int, default=207, help='num of nodes') ###########
        self.ap.add_argument('--nhid', type=int, default=40, help='num of hidden layers') # improvements
        self.ap.add_argument('--in_dim', type=int, default=2)
        
        self.ap.add_argument('--seq_len', type=int, default=12) ###########
        self.ap.add_argument('--seq_x_len', type=int, default=12)
        self.ap.add_argument('--seq_y_len', type=int, default=12)

    def parse(self):
        self.cfg = self.ap.parse_args()

        return self.cfg