import pathlib
from common.core.base_config import BaseConfig
ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']

class Pems_GWnet_Config(BaseConfig):

    def __init__(self,experiment_identifier:str=None) -> None:
        super(Pems_GWnet_Config,self).__init__()

        self.exid = self.__class__.__name__[:-7].lower()
        if experiment_identifier:
            self.exid = experiment_identifier.lower()

        parsed = self.parse_args()

        # DATA
        self.DATA.NAME = "pems"
        self.DATA.ROOT = pathlib.Path("_data/_pems_bay")
        self.DATA.DIST_FILE_PATH = (self.DATA.ROOT/"sensor_graph/distances_bay_2017.csv") # spatial data of detectors
        self.DATA.DETR_FILE_PATH = (self.DATA.ROOT/"pems-bay.h5")                         # tempore data of detectors
        # MODS


    def parse_args(self):
        parser = self.make_parser()
        
        # DATA
        parser.add_argument('--seq_x_len', type=int, default=12)
        parser.add_argument('--seq_y_len', type=int, default=12)

        # MODS
        parser.add_argument('--do_graph_conv', action='store_true',default=True,help='whether to add graph convolution layer')
        parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type', choices=ADJ_CHOICES)
        parser.add_argument('--addaptadj', action='store_true', default=True, help='whether add adaptive adj')
        
        parser.add_argument('--randomadj', action='store_true',help='whether random initialize adaptive adj')
        parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
        
        parser.add_argument('--apt_size', type=int, default=10)

        parser.add_argument('--nod_num', type=int, default=207, help='num of nodes') ###########
        parser.add_argument('--nhid', type=int, default=40, help='num of hidden layers') # improvements
        parser.add_argument('--in_dim', type=int, default=2)

        return parser.parse_args()
