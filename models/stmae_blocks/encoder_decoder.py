import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self,num_feat,drop_prob) -> None:
        super(FC,self).__init__()

        self.fc_conv1 = nn.Conv1d(num_feat,num_feat,kernel_size=1)
        self.fc_bn1 = nn.BatchNorm1d(num_feat)
        
        self.fc_conv2 = nn.Conv1d(num_feat,num_feat,kernel_size=1)
        self.fc_bn2 = nn.BatchNorm1d(num_feat)

        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self,x):
        out = self.fc_conv1(x)
        out = self.fc_bn1(out)
        out = self.dropout(self.relu(out))

        out = self.fc_conv2(x)
        out = self.fc_bn2(out)
        out = self.dropout(self.relu(out))

        out = x + out
        return out

class MLP(nn.Module):
    def __init__(self,in_channels,out_channels,num_feat,num_fc,drop_prob) -> None:
        super(MLP,self).__init__()
        self.num_fc = num_fc

        self.conv1 = nn.Conv1d(in_channels,num_feat,kernel_size=1)
        self.bn1 = nn.BatchNorm1d(num_feat)
        
        self.fc_layers = nn.ModuleList()
        for _ in range(num_fc):
            self.fc_layers.append(
                FC(num_feat,drop_prob)
            )
        self.conv2 = nn.Conv1d(num_feat,out_channels,kernel_size=1)
        
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(self.relu(out))
        for L in range(self.num_fc):
            out = self.fc_layers[L](out)
        out = self.conv2(out)
        return out

class LearnedPositionalEncoding(nn.Module):

    def __init__(self,seq_len,d_model,drop_prob=0.1):
        super(LearnedPositionalEncoding,self).__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.drop = nn.Dropout(p=drop_prob)

    def forward(self,x):
        out = x + self.pe[:,:x.size(1)]
        out = self.drop(out)
        return out

class TemporalEncoder(nn.Module):

    def __init__(
        self,seq_len,d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        batch_first,
        num_layers
        ) -> None:
        super(TemporalEncoder,self).__init__()
        self.pos_emb = LearnedPositionalEncoding(seq_len,d_model,dropout)
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    batch_first=batch_first
                ),
            num_layers=num_layers
        )

    def forward(self,x):
        out = self.pos_emb(x)
        out = self.encoder(out)
        return out

class Decoder(nn.Module):
    
    def __init__(
        self,d_model,
        nhead,
        dim_feedforward,dropout,
        activation,batch_first,
        num_layers
        ) -> None:
        super(Decoder,self).__init__()

        # 使用 Transformer Encoder 作为解码器
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first
            ),
            num_layers=num_layers,
        )
    def forward(self,x):
        out = self.encoder(x)
        return out