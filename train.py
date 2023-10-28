import argparse
import torch.optim
from utils import sparse_to_tuple
from model import GIN, GCN, GAT, GraphSAGE, VGAE
import torch.nn.functional as F
from cross_validation import *


parser = argparse.ArgumentParser(description='predict the potential miRNA-abiotic stress association')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epochs', '-e', type=int, default=400, help='Number of epochs to train')
parser.add_argument('--hidden1', '-h1', type=int, default=128, help='Embedding dimension of encoder layer 1')
parser.add_argument('--hidden2', '-h2', type=int, default=256, help='Embedding dimension of encoder layer 2')
parser.add_argument('--model', default='GIN', help='Model')
parser.add_argument('--dropout', default=0, help='Dropout rate')
parser.add_argument('--num_heads', default=2, help='Numbers of heads')
parser.add_argument('--num_neighbors', default=5, help='Numbers of neighbors')
args = parser.parse_args()


def Predict(model_str, in_dim, adj_train, pos_weight, norm, graph, features):

    # Create Model
    if model_str == 'GCN':
        model = GCN(in_dim, args.hidden1, args.hidden2, args.dropout)
    elif model_str == 'VGAE':
        model = VGAE(in_dim, args.hidden1, args.hidden2, args.dropout)
    elif model_str == 'GAT':
        model = GAT(in_dim, args.hidden1, args.hidden2, args.dropout, args.num_heads)
    elif model_str == 'SAGE':
        model = GraphSAGE(in_dim, args.hidden1, args.hidden2, args.num_neighbors, args.dropout)
    elif model_str == "GIN":
        model = GIN(in_dim, args.hidden1, args.hidden2, args.dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    adj_label = sparse_to_tuple(adj_train)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                         torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    # create training epoch
    for epoch in range(args.epochs):

        # Training
        model.train()

        logits = model.forward(graph, features)

        # compute loss
        loss = norm * F.binary_cross_entropy(logits.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
        if model_str == 'VGAE':
            kl_divergence = 0.5 / logits.size(0) * (
                    1 + 2 * model.log_std - model.mean ** 2 - torch.exp(model.log_std) ** 2).sum(1).mean()
            loss -= kl_divergence

        if epoch % 50 == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(loss))
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return logits


def main():

    adj = np.loadtxt(r".\data\adj\adj_final.txt")

    features = np.loadtxt(r".\data\integration_sim\heterogeneous_rwr_3_2.txt")  # integration 3+2

    cross_validation(adj, features, args.model)


if __name__ == '__main__':
    main()
