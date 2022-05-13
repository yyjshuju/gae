import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from gae.layers import GraphConvolution

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim,hidden_dim1, hidden_dim2, hidden_dim3,hidden_dim4, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dense = LinearNet(hidden_dim2, hidden_dim3, dropout, act=F.sigmoid)
        self.dense1 = LinearNet(hidden_dim3, 6, dropout, act=F.sigmoid)#cora 7/pumb 3/ci 6
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
                hidden1 = self.gc1(x, adj)
                return self.gc2(hidden1, adj), self.gc3(hidden1, adj)
                # return self.dense1(hidden1, adj), self.dense2(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        mu = self.dense(mu)
        logvar = self.dense(logvar)
        z = self.reparameterize(mu, logvar)
        return self.dc(z),self.dense1(z), mu, logvar

    """Decoder for using inner product for prediction."""
class LinearNet(nn.Module):
    def __init__(self, n_feature,out_features,dropout, act=F.relu):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, out_features)
        self.dropout = dropout
        self.act = act
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
