from __future__ import division
from __future__ import print_function
import os.path as osp
import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch
from torch import optim
from collections import defaultdict
from gae.model import GCNModelVAE
from gae.optimizer import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, tensor_from_numpy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--hidden', type=int, default=50)
parser.add_argument('--hidden1', type=int, default=64)
parser.add_argument('--hidden2', type=int, default=32)
parser.add_argument('--hidden3', type=int, default=16)
parser.add_argument('--hidden4', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0015)#0.0261/0.043
parser.add_argument('--dropout', type=float, default=0.00)
parser.add_argument('--dataset-str', type=str, default='citeseer')
parser.add_argument('--test_freq', type=int, default='10')
args = parser.parse_args()

# Cora数据集(引文网络)由机器学习论文组成
# PubMed数据集(引文网络)包括来自Pubmed数据库的19717篇关于糖尿病的科学出版物，分为三类
# CiteSeer数据集(引文网络)中，论文分为六类

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} dataset".format(args.dataset_str))
# adj, features = load_data(args.dataset_str)
x, y, train_mask, val_mask, test_mask, adj, features= load_data(args.dataset_str)
n_nodes, feat_dim = features.shape
# dataset = load_data2().data
node_feature = x / x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
tensor_x = tensor_from_numpy(node_feature, DEVICE)
tensor_y = tensor_from_numpy(y, DEVICE)
tensor_train_mask = tensor_from_numpy(train_mask, DEVICE)
tensor_val_mask = tensor_from_numpy(val_mask, DEVICE)
tensor_test_mask = tensor_from_numpy(test_mask, DEVICE)
num_nodes, input_dim = node_feature.shape
# normalize_adjacency = load_data2.normalization(dataset.adjacency)
# indices = torch.from_numpy(np.asarray([normalize_adjacency.row,
#                                        normalize_adjacency.col]).astype('int64')).long()
# values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
# tensor_adjacency = torch.sparse.FloatTensor(indices, values,
#                                             (num_nodes, num_nodes)).to(DEVICE)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train


adj_norm = preprocess_graph(adj)
# adj_label = adj + sp.eye(adj.shape[0])
# adj_label = torch.FloatTensor(adj_label.toarray())
# pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
# norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
model = GCNModelVAE(feat_dim, args.hidden, args.hidden1, args.hidden2, args.hidden3, args.hidden4, args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss().to(DEVICE)
    # optimizer = optim.AdaGrad(model.parameters(), lr=args.lr) RAdam

results = defaultdict(list)
hidden_emb = None




def gae_for(args):
    loss_history = []
    val_acc_history = []
    t = time.time()
    model.train()

    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        recovered, z, mu, logvar = model(features, adj_norm)
        # recovered, z, mu, logvar = model(features, tensor_adjacency)
        loss = criterion(z, tensor_y)  # 计算损失值
        # pos_weight1=torch.as_tensor(pos_weight)
        # loss = loss_function(preds=recovered, labels=adj_label,
        #                      mu=mu, logvar=logvar, n_nodes=n_nodes,
        #                      norm=norm, pos_weight=pos_weight1)
        loss.backward()
        cur_loss = loss.item()
        results['train_elbo'].append(cur_loss)
        # if 0 == epoch % 2:
            # optimizer.step()
            # optimizer.zero_grad(set_to_none=True)
        optimizer.step()
        train_acc, _, _ = test(tensor_train_mask)  # 计算当前模型训练集上的准确率
        val_acc, _, _ = test(tensor_val_mask)  # 计算当前模型在验证集上的准确率
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "train_acc=", "{:.5f}".format(train_acc),
              "train_acc2=", "{:.5f}".format(val_acc),
              # "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )
    print("Optimization Finished!")
    return loss_history, val_acc_history

def test(mask):
    model.eval()
    with torch.no_grad():
        recovered,z, mu, logvar = model(features, adj_norm)
        test_mask_logits = z[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(),tensor_y[mask].cpu().numpy()

def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


loss, val_acc = gae_for(args)
test_acc, test_logits, test_label = test(tensor_test_mask)
print("Test accuarcy: ", test_acc.item())

plot_loss_with_acc(loss, val_acc)

from sklearn.manifold import TSNE
tsne = TSNE()
out = tsne.fit_transform(test_logits)
fig = plt.figure()
for i in range(7):
    indices = test_label == i
    x, y = out[indices].T
    plt.scatter(x, y, label=str(i))
plt.legend(loc=0)
plt.savefig('tsne.png')
plt.show()


