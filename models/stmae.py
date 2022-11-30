import torch
import torch.nn as nn
import numpy as np
import random

from models.stmae_blocks.encoder_decoder import MLP as SpatialEncoder
from models.stmae_blocks.encoder_decoder import TemporalEncoder,Decoder

class STMAE_pretrain(nn.Module):

    def __init__(self,config,s_mask=None,t_mask=None) -> None:
        super(STMAE_pretrain,self).__init__()
        self.config = config
        
        self.s_mask = s_mask
        self.t_mask = t_mask
        if (s_mask is None) or (t_mask is None):
            self.generate_mask()
        
        self.s_token = nn.Parameter(
            torch.randn(
                1, 
                self.config.seq_len, 
                self.config.nod_num, 
                self.config.d_channels
                )
            )
        self.t_token = nn.Parameter(
            torch.randn(
                1, 
                self.config.seq_len,
                self.config.mlp_out_chans
                )
            )
        
        self.s_encoder = SpatialEncoder(
            in_channels=self.config.d_channels*self.config.nod_num,
            out_channels=self.config.mlp_out_chans,
            num_feat=self.config.mlp_out_chans*2,
            num_fc=self.config.num_fc_layers,
            drop_prob=self.config.drop_prob
        )
        self.t_encoder = TemporalEncoder(
            seq_len=self.config.seq_len,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_ffn,
            dropout=self.config.drop_prob,
            activation='gelu',
            batch_first=True,
            num_layers=self.config.num_layers,
        )

        self.pos_emb_d = nn.Parameter(
            torch.randn(
                1,
                self.config.seq_len,
                self.config.d_model
                )
            )
        self.decoder = Decoder(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_ffn,
            dropout=self.config.drop_prob,
            activation='gelu',
            batch_first=True,
            num_layers=self.config.num_layers,
        )
        self.linear = nn.Linear(
            in_features=self.config.d_model,
            out_features=self.config.d_channels*self.config.nod_num,
        )

    def forward(self,input):
        x = input.clone() # self supervised, don't change input
        
        bats, chan, node, seql = x.shape # metr-la (bs,2,207,12)
        x = x.permute(0,3,2,1).contiguous() # (bs,12,207,2)
        
        s_token = self.s_token.expand(bats,-1,-1,-1) #####
        x[:,self.s_mask['mask']] = s_token[:,self.s_mask['mask']]

        x = x.reshape(bats,seql,-1)# (bs,12,207*2)
        x = x.permute(0,2,1)# (bs,207*2,12)
        x = self.s_encoder(x)# (bs,? 64,12)
        x = x.permute(0,2,1)# (bs,12,64)

        np.random.shuffle(self.t_mask['mask'])
        t_mask = torch.from_numpy(self.t_mask['mask']).cuda()
        
        x = x[:,~t_mask] ########################### why mask like this?
        x = self.t_encoder(x)# (bs,? seq_len-mask_num,64)

        b, s, c = x.shape

        pos_emb = self.pos_emb_d.expand(b,-1,-1)
        pos_emb_v = pos_emb[:,~t_mask].reshape(b,-1,c)
        pos_emb_m = pos_emb[:,t_mask].reshape(b,-1,c)

        t_token = self.t_token.expand(bats,-1,-1) ######
        t_token = t_token[:,t_mask].reshape(b,-1,c)
        feat = torch.cat([
            x + pos_emb_v,t_token + pos_emb_m
        ],dim=1)

        out = self.decoder(feat) # (bs,12,64)

        out = self.linear(out) # (bs,12,207*2)

        out = out.reshape(bats,seql,node,chan) # (bs,12,207,2)
        return out

    def generate_mask(self):
        seq_len = self.config.seq_len
        nod_num = self.config.nod_num

        t_mask_num = int(self.config.t_mask_rate * seq_len)
        s_mask_num = int(self.config.s_mask_rate * nod_num)

        t_mask = np.hstack([
            np.zeros(seq_len-t_mask_num,dtype=bool),
            np.ones(t_mask_num,dtype=bool)
        ])
        np.random.shuffle(t_mask)
        
        self.t_mask = {
            'mask': t_mask, # torch.from_numpy(t_mask).cuda(),
            '_num': t_mask_num
        }

        s_mask = np.zeros((seq_len,nod_num),dtype=bool)
        for time_step in range(seq_len):
            samp = random.sample(range(0,nod_num),s_mask_num)
            s_mask[time_step,samp] = True # s_mask_num*seq_len
        
        self.s_mask = {
            'mask': s_mask, # torch.from_numpy(s_mask).cuda(),
            '_num': s_mask_num
        }

from models.gwavenet_mine import GWNet

class STMAE_finetune(nn.Module):

    def __init__(self,config) -> None:
        super(STMAE_finetune,self).__init__()
        self.config = config

        if self.config.load_pretrain:
            self.s_encoder = SpatialEncoder(
                in_channels=self.config.d_channels*self.config.nod_num,
                out_channels=self.config.mlp_out_chans,
                num_feat=self.config.mlp_out_chans*2,
                num_fc=self.config.num_fc_layers,
                drop_prob=self.config.drop_prob
            )
            self.t_encoder = TemporalEncoder(
                seq_len=self.config.seq_len,
                d_model=self.config.d_model,
                nhead=self.config.nhead,
                dim_feedforward=self.config.dim_ffn,
                dropout=self.config.drop_prob,
                activation='gelu',
                batch_first=True,
                num_layers=self.config.num_layers,
            )
            self.adapter = nn.Linear(
                in_features=self.config.d_model,
                out_features=self.config.d_channels*self.config.nod_num,
            )

        self.finetuner = GWNet.from_args(args=config,device=config.which_gpu)
    
    def forward(self,input):
        x = input.clone()
        bats, chan, node, seql = x.shape # metr-la (bs,2,207,12)
        
        if self.config.load_pretrain:
            #####
            x = x.permute(0,3,2,1).contiguous() # (bs,12,207,2)
            x = x.reshape(bats,seql,-1)# (bs,12,207*2)
            x = x.permute(0,2,1)# (bs,207*2,12)
            x = self.s_encoder(x)# (bs,? 64,12)
            x = x.permute(0,2,1) # (bs,12,64)
            
            x = self.t_encoder(x) # (bs,12,64)
            
            x = self.adapter(x) # (bs,12,207*2)
            x = x.reshape(bats,seql,node,chan)# (bs,12,207,2)
            x = x.permute(0,3,2,1) # (bs,2,207,12)

            x = x + input
            #####
        x = self.finetuner(x) # (bs,12,207,1)
        return x