import torch
import torch.nn as nn

from models.gcn import GCN
from common.tool import make_graph_inputs

class GraphEmbedding(nn.Module):
    
    def __init__(
        self,config,device,
        in_channels,
        out_channels,
        order=2,
        drop_prob=0.1
        ):
        super(GraphEmbedding,self).__init__()
        # if config.do_graph_conv:
        aptinit, fixed_supports = make_graph_inputs(args=config,device=device)

        self.graph_conv = GCN(
            device=device,
            c_in=in_channels,
            c_out=out_channels,
            fixed_supports=fixed_supports,
            addaptadj=config.addaptadj,
            aptinit=aptinit,
            apt_size=config.apt_size,
            num_nodes=config.num_nodes,
            order=order,
            dropout=drop_prob
        )
    
    def forward(self,x):
        out = self.graph_conv(x)
        out = x + out
        return out

import torch.nn.functional as F

class LearnedPositionalEncoding(nn.Module):

    def __init__(self,seq_len,d_model,drop_prob=0.1):
        super(LearnedPositionalEncoding,self).__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.drop = nn.Dropout(p=drop_prob)

    def forward(self,x):
        out = x + self.pe[:,:x.size(1)]
        out = self.drop(out)
        return out

class FeatConv(nn.Module):

    def __init__(
        self,
        feat1_in_channels,
        feat2_in_channels,
        out_channels=32
        ):
        super(FeatConv,self).__init__()
        self.feat1_conv = nn.Conv2d(
            in_channels=feat1_in_channels,
            out_channels=out_channels,
            kernel_size=(1,1)
        )
        self.feat2_conv = nn.Conv2d(
            in_channels=feat2_in_channels,
            out_channels=out_channels,
            kernel_size=(1,1)
        )
        self.bn = nn.BatchNorm2d(
            num_features=out_channels
        )
    
    def forward(self,x):
        # x.shape (bs,2,207,12)
        x_feat1, x_feat2 = x[:,[0]], x[:,1:]
        x1 = self.feat1_conv(x_feat1)
        x2 = F.leaky_relu(self.feat2_conv(x_feat2))
        x = x1 + x2 # (bs,32,207,12)
        x = self.bn(x)
        return x

class TrafficTransformer(nn.Module):
    # plane 平面
    def __init__(
        self,config,device,
        d_model=256,
        in_planes=2, 
        feat_planes=32,
        gcn_planes=32,
        seq_len=12,drop_prob=0.1
        ):
        super(TrafficTransformer,self).__init__()
        
        self.feat_extractor = FeatConv(
            feat1_in_channels=1,
            feat2_in_channels=in_planes-1,
            out_channels=feat_planes
            )

        self.graph_emb = GraphEmbedding(
            config=config,
            device=device,
            in_channels=feat_planes,
            out_channels=gcn_planes, # 1
        )
        self.fc = nn.Linear(
            in_features=config.num_nodes*gcn_planes,
            out_features=d_model
            )
        
        self.pos_emb = LearnedPositionalEncoding(
            seq_len=config.seq_x_len, ####
            d_model=d_model
        )

        self.transformer = nn.Transformer(
            d_model = d_model,
            activation='gelu',
            batch_first=True
        )

        self.predictor = nn.Linear(
            in_features=d_model,
            out_features=config.num_nodes
        )

        self.graph_conv = GraphEmbedding(
            config=config,
            device=device,
            in_channels=1,
            out_channels=in_planes,
        )

    def forward(self,src,tgt):
        # metr-la src.shape = (bs,2,207,12); y.shape depends on mode
        # (batch_size,num_features,num_nodes,time_steps)

        src = self.feat_extractor(src) # (bs,32,207,seq_len)
        tgt = self.feat_extractor(tgt) 

        src = self.graph_emb(src) # (bs,32,207,seq_len)
        tgt = self.graph_emb(tgt)

        src, tgt = src.permute(0,3,2,1), tgt.permute(0,3,2,1) # (bs,seq_len,207,32)
        
        src = torch.reshape(src,(src.size(0),src.size(1),-1)) # (bs,seq_len,207*32)
        tgt = torch.reshape(tgt,(tgt.size(0),tgt.size(1),-1))

        src = self.fc(src) # (bs,seq_len,d_model)
        tgt = self.fc(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device="cuda:0")

        src, tgt = self.pos_emb(src), self.pos_emb(tgt) # (bs,seq_len,d_model)

        out = self.transformer( # (bs,seq_len,d_model)
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask
            )
        
        out = self.predictor(out) # (bs,seq_len,207)

        out = out.unsqueeze(-1).permute(0,3,2,1) # (bs,seq_len,207,1)->(bs,1,207,seq_len)
        out = self.graph_conv(out)               # (bs,2,207,seq_len)
        
        return out




