import numpy as np
import tensorflow as tf
import os
import time
import sys
import argparse
import itertools
import LoadData as DATA

def parse_args():
    parser = argparse.ArgumentParser(description="Run ENSFM")
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset: lastfm, frappe, ml-1m, yelp2018, amazonbook')
    parser.add_argument('--batch_size', nargs='?', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=50,  # For grid search, run a small number of epochs first
                        help='Number of epochs.')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='dropout keep_prob')
    parser.add_argument('--negative_weight', type=float, default=0.05,
                        help='weight of non-observed data')
    parser.add_argument('--topK', nargs='?', type=int, default=[5,10,20],
                        help='topK for hr/ndcg')
    return parser.parse_args()

def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()

class ENSFM:
    def __init__(self, item_attribute, user_field_M, item_field_M, embedding_size, max_item_pu, args):
        self.embedding_size = embedding_size
        self.max_item_pu = max_item_pu
        self.user_field_M = user_field_M
        self.item_field_M = item_field_M
        self.weight1 = args.negative_weight
        self.item_attribute = item_attribute
        self.lambda_bilinear = [0.0, 0.0]

    def _create_placeholders(self):
        self.input_u = tf.compat.v1.placeholder(tf.int32, [None, None], name="input_u_feature")
        self.input_ur = tf.compat.v1.placeholder(tf.int32, [None, self.max_item_pu], name="input_ur")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

    def _create_variables(self):
        self.uidW = tf.Variable(tf.random.truncated_normal(shape=[self.user_field_M, self.embedding_size], mean=0.0, stddev=0.01), dtype=tf.float32, name="uidW")
        self.iidW = tf.Variable(tf.random.truncated_normal(shape=[self.item_field_M+1, self.embedding_size], mean=0.0, stddev=0.01), dtype=tf.float32, name="iidW")
        self.H_i = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name="hi")
        self.H_s = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name="hs")
        self.u_bias = tf.Variable(tf.random.truncated_normal(shape=[self.user_field_M, 1], mean=0.0, stddev=0.01), dtype=tf.float32, name="u_bias")
        self.i_bias = tf.Variable(tf.random.truncated_normal(shape=[self.item_field_M, 1], mean=0.0, stddev=0.01), dtype=tf.float32, name="i_bias")
        self.bias = tf.Variable(tf.constant(0.0), name='bias')

    def _create_vectors(self):
        self.user_feature_emb = tf.nn.embedding_lookup(self.uidW, self.input_u)
        self.summed_user_emb = tf.reduce_sum(self.user_feature_emb, 1)
        
        # Use TF 1.x dropout syntax with keep_prob
        self.H_i = tf.compat.v1.nn.dropout(self.H_i, keep_prob=self.dropout_keep_prob)
        self.H_s = tf.compat.v1.nn.dropout(self.H_s, keep_prob=self.dropout_keep_prob)
        
        self.all_item_feature_emb = tf.nn.embedding_lookup(self.iidW, self.item_attribute)
        self.summed_all_item_emb = tf.reduce_sum(self.all_item_feature_emb, 1)
        
        self.user_cross = 0.5 * (tf.square(self.summed_user_emb) - tf.reduce_sum(tf.square(self.user_feature_emb), 1))
        self.item_cross = 0.5 * (tf.square(self.summed_all_item_emb) - tf.reduce_sum(tf.square(self.all_item_feature_emb), 1))
        
        self.user_cross_score = tf.matmul(self.user_cross, self.H_s)
        self.item_cross_score = tf.matmul(self.item_cross, self.H_s)
        
        self.user_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.u_bias, self.input_u), 1)
        self.item_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.i_bias, self.item_attribute), 1)
        
        self.I = tf.ones(shape=(tf.shape(self.input_u)[0], 1))
        self.p_emb = tf.concat([self.summed_user_emb, self.user_cross_score + self.user_bias + self.bias, self.I], 1)
        
        self.I = tf.ones(shape=(tf.shape(self.summed_all_item_emb)[0], 1))
        self.q_emb = tf.concat([self.summed_all_item_emb, self.I, self.item_cross_score + self.item_bias], 1)
        self.H_i_emb = tf.concat([self.H_i, [[1.0]], [[1.0]]], 0)

    def _create_inference(self):
        self.pos_item = tf.nn.embedding_lookup(self.q_emb, self.input_ur)
        # data.item_bind_M is assumed to be set by the LoadData module
        self.pos_num_r = tf.cast(tf.not_equal(self.input_ur, data.item_bind_M), 'float32')
        self.pos_item = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_item)
        self.pos_r = tf.einsum('ac,abc->abc', self.p_emb, self.pos_item)
        self.pos_r = tf.einsum('ajk,kl->ajl', self.pos_r, self.H_i_emb)
        self.pos_r = tf.reshape(self.pos_r, [-1, self.max_item_pu])

    def _pre(self):
        dot = tf.einsum('ac,bc->abc', self.p_emb, self.q_emb)
        pre = tf.einsum('ajk,kl->aj', dot, self.H_i_emb)
        return pre

    def _create_loss(self):
        self.loss1 = self.weight1 * tf.reduce_sum(
            tf.reduce_sum(tf.reduce_sum(tf.einsum('ab,ac->abc', self.q_emb, self.q_emb), 0)
                          * tf.reduce_sum(tf.einsum('ab,ac->abc', self.p_emb, self.p_emb), 0)
                          * tf.matmul(self.H_i_emb, self.H_i_emb, transpose_b=True), 0), 0)
        self.loss1 += tf.reduce_sum((1.0 - self.weight1) * tf.square(self.pos_r) - 2.0 * self.pos_r)
        self.l2_loss0 = tf.nn.l2_loss(self.uidW)
        self.l2_loss1 = tf.nn.l2_loss(self.iidW)
        self.loss = self.loss1 + self.lambda_bilinear[0] * self.l2_loss0 + self.lambda_bilinear[1] * self.l2_loss1
        self.reg_loss = self.lambda_bilinear[0] * self.l2_loss0 + self.lambda_bilinear[1] * self.l2_loss1

    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_vectors()
        self._create_inference()
        self._create_loss()
        self.pre = self._pre()

def train_step1(u_batch, y_batch, args):
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_ur: y_batch,
        deep.dropout_keep_prob: args.dropout,
    }
    _, loss, loss1, loss2, _ = sess.run(
        [train_op1, deep.loss, deep.loss1, deep.reg_loss, deep.p_emb],
        feed_dict)
    return loss, loss1, loss2

# Modify evaluate() as needed to return metrics; here it only prints for simplicity.
def evaluate():
    eva_batch = 128
    recall50 = []
    recall100 = []
    recall200 = []
    ndcg50 = []
    ndcg100 = []
    ndcg200 = []
    user_features = data.user_test
    ll = int(len(user_features) / eva_batch) + 1
    for batch_num in range(ll):
        start_index = batch_num * eva_batch
        end_index = min((batch_num + 1) * eva_batch, len(user_features))
        u_batch = user_features[start_index:end_index]
        batch_users = end_index - start_index
        feed_dict = {
            deep.input_u: u_batch,
            deep.dropout_keep_prob: 1.0,
        }
        pre = sess.run(deep.pre, feed_dict)
        pre = np.array(pre)
        pre = np.delete(pre, -1, axis=1)
        user_id = []
        for one in u_batch:
            user_id.append(data.binded_users["-".join([str(item) for item in one])])
        idx = np.zeros_like(pre, dtype=bool)
        idx[data.Train_data[user_id].nonzero()] = True
        pre[idx] = -np.inf
        recall = []
        for kj in args.topK:
            idx_topk_part = np.argpartition(-pre, kj, 1)
            pre_bin = np.zeros_like(pre, dtype=bool)
            pre_bin[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]] = True
            true_bin = np.zeros_like(pre, dtype=bool)
            true_bin[data.Test_data[user_id].nonzero()] = True
            tmp = (np.logical_and(true_bin, pre_bin).sum(axis=1)).astype(np.float32)
            recall.append(tmp / np.minimum(kj, true_bin.sum(axis=1)))
        ndcg = []
        for kj in args.topK:
            idx_topk_part = np.argpartition(-pre, kj, 1)
            topk_part = pre[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]]
            idx_part = np.argsort(-topk_part, axis=1)
            idx_topk = idx_topk_part[np.arange(end_index - start_index)[:, np.newaxis], idx_part]
            tp = np.log(2) / np.log(np.arange(2, kj + 2))
            test_batch = data.Test_data[user_id]
            DCG = (test_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
            IDCG = np.array([(tp[:min(n, kj)]).sum() for n in test_batch.getnnz(axis=1)])
            ndcg.append(DCG / IDCG)
        recall50.append(recall[0])
        recall100.append(recall[1])
        recall200.append(recall[2])
        ndcg50.append(ndcg[0])
        ndcg100.append(ndcg[1])
        ndcg200.append(ndcg[2])
    recall50 = np.hstack(recall50)
    recall100 = np.hstack(recall100)
    recall200 = np.hstack(recall200)
    ndcg50 = np.hstack(ndcg50)
    ndcg100 = np.hstack(ndcg100)
    ndcg200 = np.hstack(ndcg200)
    print("HR@10:", np.mean(recall100), "NDCG@10:", np.mean(ndcg100))
    # For automation, you might return these metrics
    return np.mean(recall100), np.mean(ndcg100)

def run_experiment(args, data, random_seed=2019):
    # Build a new graph for each experiment run
    with tf.Graph().as_default():
        tf.compat.v1.set_random_seed(random_seed)
        session_conf = tf.compat.v1.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        global sess, deep, train_op1  # make these global so evaluate() and train_step1() can access them
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            deep = ENSFM(data.item_map_list, data.user_field_M, data.item_field_M, args.embed_size, data.max_positive_len, args)
            deep._build_graph()
            train_op1 = tf.compat.v1.train.AdagradOptimizer(learning_rate=args.lr, initial_accumulator_value=1e-8).minimize(deep.loss)
            sess.run(tf.compat.v1.global_variables_initializer())
            print("Initial evaluation:")
            evaluate()
            for epoch in range(args.epochs):
                print("Epoch:", epoch)
                start_t = _writeline_and_time('\tUpdating...')
                shuffle_indices = np.random.permutation(np.arange(len(data.user_train)))
                data.user_train = data.user_train[shuffle_indices]
                data.item_train = data.item_train[shuffle_indices]
                ll = int(len(data.user_train) / args.batch_size)
                for batch_num in range(ll):
                    start_index = batch_num * args.batch_size
                    end_index = min((batch_num + 1) * args.batch_size, len(data.user_train))
                    u_batch = data.user_train[start_index:end_index]
                    i_batch = data.item_train[start_index:end_index]
                    train_step1(u_batch, i_batch, args)
                print('\r\tUpdating: time=%.2f' % (time.time() - start_t))
                if epoch % args.verbose == 0:
                    evaluate()
            print("Final evaluation:")
            final_hr, final_ndcg = evaluate()
    return final_hr, final_ndcg

if __name__ == '__main__':
    np.random.seed(2019)
    random_seed = 2019
    args = parse_args()
    
    # Choose dataset path based on argument
    if args.dataset == 'lastfm':
        print('load lastfm data')
        DATA_ROOT = '../data/lastfm'
    elif args.dataset == 'frappe':
        print('load frappe data')
        DATA_ROOT = '../data/frappe'
    elif args.dataset == 'ml-1m':
        print('load ml-1m data')
        DATA_ROOT = '../data/ml-1m'
    elif args.dataset == 'yelp2018':
        print('load yelp data')
        DATA_ROOT = '../data/yelp2018'
    elif args.dataset == 'amazonbook':
        print('load amazon book data')
        DATA_ROOT = '../data/amzbook'

    # Open a file to record results if needed
    f1 = open(os.path.join(DATA_ROOT, 'ENSFM_hyperparam_results.txt'), 'w')
    data = DATA.LoadData(DATA_ROOT)
    
    # Define a hyperparameter grid (you can expand or modify these lists)
    lr_values = [0.005, 0.01, 0.02, 0.05] #[0.005, 0.01, 0.02, 0.05]
    dropout_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    neg_weight_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    results = []
    # Loop over the grid and run experiments
    for lr, dropout, neg_weight in itertools.product(lr_values, dropout_values, neg_weight_values):
        print("Running experiment with lr={}, dropout_keep_prob={}, negative_weight={}".format(lr, dropout, neg_weight))
        f1.write("lr={}, dropout={}, neg_weight={}\n".format(lr, dropout, neg_weight))
        # Update hyperparameters in args
        args.lr = lr
        args.dropout = dropout
        args.negative_weight = neg_weight
        # For quick testing during search, you might use fewer epochs
        args.epochs = 50
        hr, ndcg = run_experiment(args, data, random_seed)
        results.append(((lr, dropout, neg_weight), (hr, ndcg)))
        f1.write("Final HR@10: {}  NDCG@10: {}\n\n".format(hr, ndcg))
        f1.flush()
    
    print("Hyperparameter search complete. Results:")
    for combo, metrics in results:
        print("Parameters (lr, dropout, neg_weight): {}  => HR@10: {}, NDCG@10: {}".format(combo, metrics[0], metrics[1]))
    
    # --- New: Sorting results in descending order based on HR@10 ---
    sorted_results = sorted(results, key=lambda x: x[1][0], reverse=True)
    print("\nSorted hyperparameter search results (by HR@10 in descending order):")
    f1.write("\nSorted hyperparameter search results (by HR@10 in descending order):\n")
    for combo, metrics in sorted_results:
        line = "Parameters (lr, dropout, neg_weight): {}  => HR@10: {}, NDCG@10: {}\n".format(combo, metrics[0], metrics[1])
        print(line.strip())
        f1.write(line)
    
    f1.close()
