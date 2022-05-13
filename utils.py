import pickle as pkl
import pickle
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import sparse
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import matplotlib.pyplot as plt
import os
import os.path as osp
import itertools
from collections import namedtuple
Data = namedtuple('Data', ['x', 'y', 'adjacency',
                           'train_mask', 'val_mask', 'test_mask'])
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)

def load_data(dataset):
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
    x, tx, allx, y, ty, ally, graph= tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    train_index = np.arange(y.shape[0])
    val_index = np.arange(y.shape[0], y.shape[0] + 500)
    stx = tx.A
    sallx=allx.A
    x1 = np.concatenate((sallx, stx), axis=0)
    y1= np.concatenate((ally, ty), axis=0).argmax(axis=1)

    x1[test_idx_reorder] = x1[test_idx_range]
    y1[test_idx_reorder] = y1[test_idx_range]
    num_nodes = x1.shape[0]

    train_mask = np.zeros(num_nodes, dtype=np.bool)
    val_mask = np.zeros(num_nodes, dtype=np.bool)
    test_mask = np.zeros(num_nodes, dtype=np.bool)
    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_idx_reorder] = True

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # return adj, features
    return x1, y1, train_mask, val_mask, test_mask, adj, features

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def plot_results(results, test_freq, path='results.png'):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    x_axis_train = range(len(results['train_elbo']))
    x_axis_test = range(0, len(x_axis_train), test_freq)
    # Elbo
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, results['train_elbo'])
    ax.set_ylabel('Loss (ELBO)')
    ax.set_title('Loss (ELBO)')
    ax.legend(['Train'], loc='upper right')

    # Accuracy
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x_axis_train, results['accuracy_train'])
    ax.plot(x_axis_test, results['accuracy_test'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend(['Train', 'Test'], loc='lower right')

    # ROC
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x_axis_train, results['roc_train'])
    ax.plot(x_axis_test, results['roc_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC AUC')
    ax.set_title('ROC AUC')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Precision
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x_axis_train, results['ap_train'])
    ax.plot(x_axis_test, results['ap_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Save
    fig.tight_layout()
    fig.savefig(path)

def  get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    accuracy = accuracy_score((preds_all > 0.5).astype(float), labels_all)
    return roc_score, ap_score, accuracy
