import torch
import torch.nn as nn
import torch.nn.functional as F

def nconv(x,A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('bcnl,nm->bcml',(x,A)).contiguous()

class GCN(nn.Module):

    def __init__(
        self,
        device,
        c_in,c_out,
        fixed_supports,
        addaptadj=True, 
        aptinit=None,
        apt_size=10,
        num_nodes = 207,
        order=2,
        dropout=0.1
        ):
        super().__init__()
        self.order = order
        self.dropout = dropout
        self.fixed_supports = fixed_supports or []
        self.suppports_len = len(fixed_supports)
        
        self.addaptadj = addaptadj
        if self.addaptadj:
            if aptinit is None:
                nodevecs = torch.randn(num_nodes,apt_size), torch.randn(apt_size, num_nodes)
            else:
                nodevecs = self.svd_init(apt_size,aptinit)
            self.nodevec1, self.nodevec2 = [nn.Parameter(n.to(device), requires_grad=True) for n in nodevecs] # adaptive
            self.suppports_len += 1

        c_in = (order*self.suppports_len+1)*c_in
        self.final_conv = nn.Conv2d(
            c_in,c_out, 
            kernel_size=(1, 1),
            stride=(1, 1), 
            padding=(0, 0),
            bias=True
            )

    @staticmethod
    def svd_init(apt_size,aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2
    
    def forward(self,x):
        supports_list = self.fixed_supports
        if self.addaptadj:
             adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
             supports_list = supports_list + [adp] ### += 原地改变 id不变
        # x (b,c,n,l)
        out = [x]
        for adj in supports_list:
            temp = x
            for ord in range(1,self.order+1):
                x_ = nconv(temp,adj)
                out.append(x_)
                temp = x_
        h = torch.cat(out,dim=1)
        h = self.final_conv(h)
        h = F.dropout(h,self.dropout,training=self.training)
        return h