import torch
import torch.nn as nn
import torch.nn.functional as F

from common.tool import make_graph_inputs
from models.gcn import GCN

class GWNet(nn.Module):

    def __init__(
        self,device,num_nodes,supports=None,
        
        do_graph_conv=True,cat_feat_gc=False,
        addaptadj=True, aptinit=None,apt_size=10,
        
        kernel_size=2,blocks=4,layers=2,dropout=0.3,
        in_dim=2,out_dim=12,
        residual_channels=32, dilation_channels=32,
        skip_channels=256, end_channels=512
        ):
        super().__init__()
        self.blocks = blocks
        self.layers = layers
        self.dropout = dropout

        self.do_graph_conv = do_graph_conv
        self.cat_feat_gc = cat_feat_gc
        
        if self.cat_feat_gc:
            self.start_conv = nn.Conv2d(
                in_channels=1,
                out_channels=residual_channels,
                kernel_size=(1,1)
                )
            self.cat_feature_conv = nn.Conv2d(
                in_channels=in_dim-1,
                out_channels=residual_channels,
                kernel_size=(1,1)
                )
        else:
            self.start_conv = nn.Conv2d(
                in_channels=in_dim,
                out_channels=residual_channels,
                kernel_size=(1,1)
                )
        
        self.id = nn.Identity()

        self.receptive_field = 1 # 感受野 

        self.tcna_tanh_convs = nn.ModuleList() # tanh filter
        self.tcnb_sigm_convs = nn.ModuleList() # sigmoid gate
        for B in range(blocks):
            D = 1
            additional_scope = (kernel_size-1)*D # kernel __ => _._ num of dot 
            for L in range(layers):
                # dilated convolutions
                self.tcna_tanh_convs.append(
                    nn.Conv2d(
                        residual_channels, 
                        dilation_channels, 
                        (1, kernel_size), 
                        dilation=D
                        )
                )
                self.tcnb_sigm_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        (1, kernel_size), 
                        dilation=D
                        )
                )
                self.receptive_field += additional_scope
                additional_scope *= 2
                D *= 2
        
        depth = list(range(blocks * layers))

        self.skip_convs = nn.ModuleList([nn.Conv2d(dilation_channels, skip_channels, (1, 1)) for _ in depth])

        if do_graph_conv:
            self.fixed_supports = supports or []
            self.graph_convs = nn.ModuleList(
                [GCN(
                    device=device,
                    c_in=dilation_channels, 
                    c_out=residual_channels, 
                    fixed_supports = self.fixed_supports,
                    addaptadj=addaptadj,
                    aptinit=aptinit,
                    apt_size=apt_size,
                    num_nodes=num_nodes,
                    dropout=dropout) for _ in depth]
                    )
        self.residual_convs = nn.ModuleList([nn.Conv2d(dilation_channels, residual_channels, (1, 1)) for _ in depth])
        self.bn = nn.ModuleList([nn.BatchNorm2d(residual_channels) for _ in depth])
     
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, (1, 1), bias=True)
    
    @classmethod
    def from_args(cls,args,device):
        if args.do_graph_conv:
            aptinit, supports = make_graph_inputs(args=args,device=device)
        defaults = dict(
            device=device,
            num_nodes=args.num_nodes,
            supports=supports,
            do_graph_conv=args.do_graph_conv,
            cat_feat_gc=args.cat_feat_gc,
            
            addaptadj=args.addaptadj, 
            aptinit=aptinit,
            apt_size=args.apt_size,
            
            dropout=args.dropout,
            in_dim=args.in_dim,out_dim=args.seq_len,
            residual_channels=args.nhid, 
            dilation_channels=args.nhid,
            skip_channels=args.nhid*8, 
            end_channels=args.nhid*16
        )
        model = cls(**defaults)
        return model
    
    def forward(self,x):
        
        if x.size(3) < self.receptive_field:
            x = F.pad(x,(self.receptive_field-x.size(3),0,0,0))
        
        if self.cat_feat_gc:
            f1, f2 =x[:,[0]],x[:,1:]
            x1 = self.start_conv(f1)
            x2 = F.leaky_relu(self.cat_feature_conv(f2))
            x = x1 + x2
        else:
            x = self.start_conv(x) # torch.Size([bs, 32, 207, 13])
        
        skip = 0
        for i in range(self.blocks*self.layers):
            
            identity = self.id(x)  # i=0 torch.Size([bs, 32r, 207, 13])

            # dilated convolution
            filter = torch.tanh(self.tcna_tanh_convs[i](x)) 
            gate = torch.sigmoid(self.tcnb_sigm_convs[i](x))
            x = filter * gate # i=0 torch.Size([64, 32d, 207, 12]); i=1 torch.Size([64, 32, 207, 10])

            # skip connection
            s = self.skip_convs[i](x) # i=0 torch.Size([64, 256, 207, 12]); i=1 torch.Size([64, 256, 207, 10])
            if i > 0:
                skip = skip[:, :, :,  -s.size(3):] # 对齐维度
            skip = s + skip # i=1 torch.Size([64, 256 !!!, 207, 10 ???]) 

            if i == (self.blocks*self.layers-1):
                break # last X getting ignored anyway

            # graph learning
            if self.do_graph_conv:
                graph_out = self.graph_convs[i](x)
                x = x + graph_out
            else:
                x = self.residual_convs[i](x) 
            
            x = x + identity[:, :, :,  -x.size(3):]
            x = self.bn[i](x) # i=0 torch.Size([bs, 32r, 207, 12])
        
        x = F.relu(skip) # last torch.Size([bs, 256, 207, 1])
        x = F.relu(self.end_conv_1(x)) # torch.Size([bs, 512, 207, 1])
        x = self.end_conv_2(x) # # torch.Size([bs, 12, 207, 1])

        return x
        