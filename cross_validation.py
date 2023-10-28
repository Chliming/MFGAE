import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import dgl
from utils import sparse_to_tuple, generate_test_index, test_evaluate, get_metrics
import random
from train import Predict


def cross_validation(adj, features, model_str):
    index_matrix = np.mat(np.where(adj == 1))

    association_nam = index_matrix.shape[1]

    random_index = index_matrix.T.tolist()
    random.shuffle(random_index)

    k_folds = 5

    CV_size = int(association_nam / k_folds)

    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]

    random_index = temp

    metric = np.zeros((1, 7))

    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k + 1))

        train_matrix = np.matrix(adj, copy=True)

        train_matrix[tuple(np.array(random_index[k]).T)] = 0

        features = sp.csr_matrix(features[:, :], dtype=np.float32).tolil()
        features = sparse_to_tuple(features.tocoo())

        # Create model
        adj_train_ = sp.csr_matrix(train_matrix[:, :], dtype=np.float32)

        graph = dgl.from_scipy(adj_train_)
        dgl.add_self_loop(graph)
        graph.ndata['id'] = torch.arange(graph.number_of_nodes())

        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                            torch.FloatTensor(features[1]),
                                            torch.Size(features[2]))

        pos_weight = float(adj_train_.shape[0] * adj_train_.shape[0] - adj_train_.sum()) / adj_train_.sum()

        norm = adj_train_.shape[0] * adj_train_.shape[0] / float(
            (adj_train_.shape[0] * adj_train_.shape[0] - adj_train_.sum()) * 2)

        features = features.to_dense()
        in_dim = features.shape[-1]

        res = Predict(model_str, in_dim, adj_train_, pos_weight, norm, graph, features)

        predict_y_proba = res.detach()

        metric_tmp = cv_model_evaluate(adj, predict_y_proba, random_index, k)
        print(metric_tmp)

        metric += metric_tmp

    print("AVG：", metric / k_folds)


def cv_model_evaluate(adj, predict_matrix, random_index, k):

    # randomly sampling 1：1
    text_neg = generate_test_index(adj, len(random_index[k]))

    real_score, predict_score = test_evaluate(predict_matrix, text_neg, random_index, k)

    return get_metrics(real_score, predict_score)
