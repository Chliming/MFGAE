import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv, GINConv


class GIN(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, dropout):
        super(GIN, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.drop = nn.Dropout(dropout)

        layers = [GINConv(nn.Linear(self.in_dim, self.hidden1_dim), aggregator_type="mean", activation=F.relu)
                , GINConv(nn.Linear(self.hidden1_dim, self.hidden2_dim), aggregator_type="sum", activation=F.relu)]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, features):
        #features = self.drop(features)
        h = self.layers[0](g, features)
        h = self.drop(h)
        sampled_z = self.layers[1](g, h)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec


class GCN(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, dropout):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.drop = nn.Dropout(dropout)

        layers = [GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True)
                , GraphConv(self.hidden1_dim, self.hidden2_dim, activation=F.relu, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, features):
        #features = self.drop(features)
        h = self.layers[0](g, features)
        h = self.drop(h)
        sampled_z = self.layers[1](g, h)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec


class GAT(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, dropout, num_heads):
        super(GAT, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.num_heads = num_heads
        self.drop = nn.Dropout(dropout)

        layers = [GATConv(in_feats=self.in_dim, out_feats=self.hidden1_dim
                          , num_heads=self.num_heads, activation=F.relu, allow_zero_in_degree=True),
                  GATConv(in_feats=self.hidden1_dim, out_feats=self.hidden2_dim
                          , num_heads=self.num_heads, activation=F.relu, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, features):
        #features = self.drop(features)
        h = self.layers[0](g, features)
        h = self.drop(h)
        h = torch.mean(h.view(h.shape[0], self.num_heads, -1), dim=1)
        h = self.layers[1](g, h)
        sampled_z = torch.mean(h.view(h.shape[0], self.num_heads, -1), dim=1)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, num_neighbors, dropout):
        super(GraphSAGE, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.num_neighbors = num_neighbors
        self.drop = nn.Dropout(dropout)

        layers = [SAGEConv(in_feats=self.in_dim, out_feats=self.hidden1_dim
                           , activation=F.relu, aggregator_type="mean")
                , SAGEConv(in_feats=self.hidden1_dim, out_feats=self.hidden2_dim
                           , activation=F.relu, aggregator_type="mean")]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, features):
        seeds = torch.arange(g.number_of_nodes())
        sampled_graph = dgl.sampling.sample_neighbors(g, seeds, fanout=self.num_neighbors, replace=False)
        sampled_feats = features[sampled_graph.ndata['id']]

        h = self.layers[0](sampled_graph, sampled_feats)
        h = self.drop(h)
        sampled_z = self.layers[1](sampled_graph, h)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec


class VGAE(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, dropout):
        super(VGAE, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        layers = [GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(dropout)

    def encoder(self, g, features):
        features = self.drop(features)
        h = self.layers[0](g, features)
        h = self.drop(h)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec


