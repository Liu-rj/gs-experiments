from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy.sparse as sp

from utils import *
from models import GCN, MLP, GCN_APPRO
import json
from networkx.readwrite import json_graph
import os
import argparse
import numpy as np
from tqdm import tqdm

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')
flags.DEFINE_string('model', 'gcn_appr', 'Model string.')  # 'gcn', 'gcn_appr'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 3, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4,
                   'Weight for L2 loss on embedding matrix.')
# flags.DEFINE_integer('early_stopping', 100,
#                      'Tolerance for early stopping (# of epochs).')
# flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data


def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in inputs]


def loadRedditFromG(dataset_dir, inputfile):
    f = open(dataset_dir+inputfile)
    objects = []
    for _ in range(pkl.load(f)):
        objects.append(pkl.load(f))
    adj, train_labels, val_labels, test_labels, train_index, val_index, test_index = tuple(
        objects)
    feats = np.load(dataset_dir + "/reddit-feats.npy")
    return sp.csr_matrix(adj), sp.lil_matrix(feats), train_labels, val_labels, test_labels, train_index, val_index, test_index


def loadDataFromNPZ(dataset_dir, graph_name):
    adj = sp.load_npz(dataset_dir+f"{graph_name}_adj.npz")
    data = np.load(dataset_dir+f"{graph_name}.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index'], data['n_classes']


def transferRedditDataFormat(dataset_dir, output_file):
    G = json_graph.node_link_graph(
        json.load(open(dataset_dir + "/reddit-G.json")))
    labels = json.load(open(dataset_dir + "/reddit-class_map.json"))

    train_ids = [n for n in G.nodes() if not G.node[n]['val']
                 and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n]['test']]
    val_ids = [n for n in G.nodes() if G.node[n]['val']]
    train_labels = [labels[i] for i in train_ids]
    test_labels = [labels[i] for i in test_ids]
    val_labels = [labels[i] for i in val_ids]
    feats = np.load(dataset_dir + "/reddit-feats.npy")
    # Logistic gets thrown off by big counts, so log transform num comments and score
    feats[:, 0] = np.log(feats[:, 0] + 1.0)
    feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
    feat_id_map = json.load(open(dataset_dir + "reddit-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.iteritems()}

    # train_feats = feats[[feat_id_map[id] for id in train_ids]]
    # test_feats = feats[[feat_id_map[id] for id in test_ids]]

    # numNode = len(feat_id_map)
    # adj = sp.lil_matrix(np.zeros((numNode,numNode)))
    # for edge in G.edges():
    #     adj[feat_id_map[edge[0]], feat_id_map[edge[1]]] = 1

    train_index = [feat_id_map[id] for id in train_ids]
    val_index = [feat_id_map[id] for id in val_ids]
    test_index = [feat_id_map[id] for id in test_ids]
    np.savez(output_file, feats=feats, y_train=train_labels, y_val=val_labels, y_test=test_labels, train_index=train_index,
             val_index=val_index, test_index=test_index)


def transferLabel2Onehot(labels, N):
    y = np.zeros((len(labels), N))
    labels[np.isnan(labels)] = 0
    labels = labels.astype(int)
    y[np.arange(labels.size), labels] = 1
    return y


def main(rank1, rank0, with_prob, args):

    # config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
    #                 inter_op_parallelism_threads = 1,
    #                 intra_op_parallelism_threads = 4,
    #                 log_device_placement=False)
    if args.dataset == 'products':
        adj, features, y_train, y_val, y_test, train_index, val_index, test_index, n_classes = loadDataFromNPZ(
            "data/products/", 'ogbn-products')
    elif args.dataset == 'papers':
        adj, features, y_train, y_val, y_test, train_index, val_index, test_index, n_classes = loadDataFromNPZ(
            "data/papers/", 'ogbn-papers100M')
    elif args.dataset == 'friendster':
        adj, features, y_train, y_val, y_test, train_index, val_index, test_index, n_classes = loadDataFromNPZ(
            "data/friendster/", 'friendster')
    print("adjcent matrix shape:", adj.shape)
    print("num of edges:", len(adj.row))
    print("train samples:", len(train_index))
    print("validation samples:", len(val_index))
    print("test samples:", len(test_index))
    # adj = adj+adj.T
    print('converting scipy coo matrix to csr')
    adj = adj.tocsr()

    print('tranfering label to one hot')
    y_train = transferLabel2Onehot(y_train, n_classes)
    y_val = transferLabel2Onehot(y_val, n_classes)
    # y_test = transferLabel2Onehot(y_test, n_classes)

    # print('converting features to scipy lil matrix')
    # features = sp.lil_matrix(features)

    adj_train = adj[train_index, :][:, train_index]
    adj_val = adj[val_index, :][:, val_index]

    numNode_train = adj_train.shape[0]
    numNode_val = adj_val.shape[0]

    train_mask = np.ones((numNode_train,))
    val_mask = np.ones((numNode_val,))
    # test_mask = np.ones((y_test.shape[0],))

    # print("numNode", numNode)

    # Some preprocessing
    # features = nontuple_preprocess_features(features).todense()
    train_features = features[train_index]
    val_features = features[val_index]

    if FLAGS.model == 'gcn_appr':
        normADJ_train = nontuple_preprocess_adj(adj_train)
        normADJ_val = nontuple_preprocess_adj(adj_val)
        normADJ = nontuple_preprocess_adj(adj)
        # normADJ_val = nontuple_preprocess_adj(adj_val)
        # normADJ_test = nontuple_preprocess_adj(adj_test)

        num_supports = 2
        model_func = GCN_APPRO
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        # helper variable for sparse dropout
        'num_features_nonzero': tf.placeholder(tf.int32)
    }

    # Create model
    model = model_func(
        placeholders, input_dim=features.shape[-1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(
            features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy],
                            feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    cost_val = []
    acc_val = []

    # testSupport = [sparse_to_tuple(normADJ), sparse_to_tuple(normADJ)]
    # valSupport = [sparse_to_tuple(
    #     normADJ), sparse_to_tuple(normADJ[val_index, :])]
    # testSupport = [sparse_to_tuple(
    #     normADJ), sparse_to_tuple(normADJ[test_index, :])]

    if with_prob:
        print('Importance Sampling...')
    else:
        print('Uniform Sampling...')

    t = time.time()
    maxACC = 0.0
    sample_time_list = []
    epoch_time_list = []
    # Train model
    for epoch in range(FLAGS.epochs):
        sample_time = 0

        t1 = time.time()

        for batch in tqdm(iterate_minibatches_listinputs([normADJ_train, y_train, train_mask], batchsize=args.batchsize, shuffle=True)):
            [normADJ_batch, y_train_batch, train_mask_batch] = batch
            if sum(train_mask_batch) < 1:
                continue

            start = time.time()

            p1 = column_prop(normADJ_batch)
            q1 = np.random.choice(np.arange(numNode_train),
                                  rank1, replace=False, p=p1)  # top layer
            # q0 = np.random.choice(np.arange(numNode_train), rank0, p=p0)  # bottom layer
            support1 = sparse_to_tuple(normADJ_batch[:, q1].dot(
                sp.diags(1.0 / (p1[q1] * rank1))))

            p2 = column_prop(normADJ_train[q1, :])
            q0 = np.random.choice(np.arange(numNode_train),
                                  rank0, replace=False, p=p2)

            support0 = sparse_to_tuple(normADJ_train[q1, :][:, q0])
            # selected nodes for approximation
            features_inputs = np.diag(
                1.0 / (p2[q0] * rank0)).dot(train_features[q0, :])

            sample_time += time.time() - start

            # Construct feed dictionary
            feed_dict = construct_feed_dict(features_inputs, [support0, support1], y_train_batch, train_mask_batch,
                                            placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss,
                            model.accuracy], feed_dict=feed_dict)

        # Validation
        for batch in tqdm(iterate_minibatches_listinputs([normADJ_val, y_val, val_mask], batchsize=args.batchsize, shuffle=True)):
            [normADJ_batch, y_val_batch, val_mask_batch] = batch
            if sum(val_mask_batch) < 1:
                continue

            start = time.time()

            p1 = column_prop(normADJ_batch)
            q1 = np.random.choice(np.arange(numNode_val),
                                  rank1, replace=False, p=p1)  # top layer
            support1 = sparse_to_tuple(normADJ_batch[:, q1].dot(
                sp.diags(1.0 / (p1[q1] * rank1))))

            p2 = column_prop(normADJ_val[q1, :])
            q0 = np.random.choice(np.arange(numNode_val),
                                  rank0, replace=False, p=p2)

            support0 = sparse_to_tuple(normADJ_val[q1, :][:, q0])
            # selected nodes for approximation
            features_inputs = np.diag(
                1.0 / (p2[q0] * rank0)).dot(val_features[q0, :])

            sample_time += time.time() - start

            # Construct feed dictionary
            feed_dict = construct_feed_dict(features_inputs, [support0, support1], y_val_batch, val_mask_batch,
                                            placeholders)

            # Evaluating step
            cost, acc = sess.run(
                [model.loss, model.accuracy], feed_dict=feed_dict)
            cost_val.append(cost)
            acc_val.append(acc)
        # cost, acc, duration = evaluate(
        #     features, valSupport, y_val, val_mask, placeholders)

        sample_time_list.append(sample_time)
        epoch_time_list.append(time.time() - t1)

        # if epoch > 50 and acc > maxACC:
        #     maxACC = acc
        #     save_path = saver.save(sess, "tmp/tmp_redditModel.ckpt")

        # Print results
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]),
              "val_loss=", "{:.5f}".format(np.mean(cost_val)),
              "val_acc=", "{:.5f}".format(np.mean(acc_val)),
              "time per epoch=", "{:.5f}".format(epoch_time_list[-1]),
              "epoch sampling time=", "{:.5f}".format(sample_time_list[-1]))

        # if epoch > FLAGS.early_stopping and np.mean(cost_val[-2:]) > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        #     print("Early stopping...")
        #     break

    print('Average epoch e2e time:', np.mean(epoch_time_list))
    print('Average epoch sampling time:', np.mean(sample_time_list))

    # train_duration = time.time() - t
    # Testing
    # if os.path.exists("tmp/tmp_redditModel.ckpt"):
    #     saver.restore(sess, "tmp/tmp_redditModel.ckpt")
    # test_cost, test_acc, test_duration = evaluate(features, testSupport, y_test, test_mask,
    #                                               placeholders)
    # print("rank1 = {}".format(rank1), "rank0 = {}".format(rank0), "cost=", "{:.5f}".format(test_cost),
    #       "accuracy=", "{:.5f}".format(
    #           test_acc), "training time=", "{:.5f}".format(train_duration),
    #       "epoch = {}".format(epoch + 1),
    #       "test time=", "{:.5f}".format(test_duration))


def transferG2ADJ():
    G = json_graph.node_link_graph(json.load(open("reddit/reddit-G.json")))
    feat_id_map = json.load(open("reddit/reddit-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.iteritems()}
    numNode = len(feat_id_map)
    adj = np.zeros((numNode, numNode))
    newEdges0 = [feat_id_map[edge[0]] for edge in G.edges()]
    newEdges1 = [feat_id_map[edge[1]] for edge in G.edges()]

    # for edge in G.edges():
    #     adj[feat_id_map[edge[0]], feat_id_map[edge[1]]] = 1
    adj = sp.csr_matrix((np.ones((len(newEdges0),)),
                        (newEdges0, newEdges1)), shape=(numNode, numNode))
    sp.save_npz("reddit_adj.npz", adj)


if __name__ == "__main__":
    # transferRedditDataFormat("reddit/","data/reddit.npz")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='products', choices=['products', 'papers', 'friendster'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=1024,
                        help="batch size for training")
    args = parser.parse_args()
    print(args)

    main(1000, 1000, True, args)
    # for k in [50, 100, 200, 400]:
    #     main(100, k)
