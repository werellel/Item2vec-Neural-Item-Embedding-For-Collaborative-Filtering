import os
import time
import math
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
import tensorflow.compat.v1 as tf
import string

from utils import save_data, read_data

tf.disable_v2_behavior()

import sys

class Item2Vec(object):
    def __init__(self, config, sess):
        self.sess = sess

        self.alpha = config['alpha']
        self.embed_size = config['embed_size']
        self.neg_sample_size = config['neg_sample_size']
        self.min_frequency = config['min_frequency']
        self.window = config['window']
        self.lr = config['lr']
        self.min_lr = config['min_lr']
        self.table_size = config['table_size'] 

    def preprocessing(self, path):
        """ """    
        start_time = time.time()
        self.uid_itemlist_dic = read_data(path)
        self.item_list = []
        items = [item for itemlist in self.uid_itemlist_dic.values() for item in itemlist]
        self.total_item_count = len(items)
        self.counter = [['UNK', 0]]
        self.counter.extend([list(item) for item in Counter(items).most_common() if item[1] > self.min_frequency ])
        self.item_size = len(self.counter)
        item2idx = dict()
        for item, _ in self.counter:
            item2idx[item] = len(item2idx)
        data = list()
        unk_count = 0
        for item in items:
            if item in item2idx:
                idx = item2idx[item]
            else:
                idx = 0 # item2idx['UNK']
            data.append(idx)
        self.counter[0][1] = unk_count
        idx2item = dict(zip(item2idx.values(), item2idx.keys()))
        duration = time.time() - start_time

        print("%d item processed in %.2f seconds" % (self.total_item_count, duration))
        print("item size after eliminating item occuring less than %d times: %d" % (self.min_frequency, self.item_size))

        self.data = data
        self.items = items
        self.item2idx = item2idx 
        self.idx2item = idx2item

        self.decay = (self.min_lr-self.lr)/(self.total_item_count*self.window)
        self.labels = np.zeros([1, 1+self.neg_sample_size], dtype=np.float32); self.labels[0][0] = 1
        self.contexts = np.ndarray(1 + self.neg_sample_size, dtype=np.int32)
        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.int32, [1], name='pos_x')
        self.y = tf.placeholder(tf.int32, [1 + self.neg_sample_size], name='pos_x')

        init_width = 0.5 / self.embed_size
        self.embed = tf.Variable(tf.random_uniform([self.item_size, self.embed_size], -init_width, init_width), name='embed')
        self.w = tf.Variable(tf.truncated_normal([self.item_size, self.embed_size], stddev=1.0 / math.sqrt(self.embed_size)), name='w')

        self.x_embed = tf.nn.embedding_lookup(self.embed, self.x, name='pos_embed')
        self.y_w = tf.nn.embedding_lookup(self.w, self.y, name='pos_embed')

        self.mul = tf.matmul(self.x_embed, self.y_w, transpose_b=True)
        self.p = tf.nn.sigmoid(self.mul)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.p, labels=self.labels)
        self.train = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def train_pair(self, item_idx, contexts):
        self.sess.run(self.train, feed_dict={self.x: [item_idx], self.y: contexts})

    def build_table(self):
        start_time = time.time()
        total_count_pow = 0
        for _, count in self.counter:
            total_count_pow += math.pow(count, self.alpha)
        item_idx = 1
        self.table = np.zeros([self.table_size], dtype=np.int32)
        item_prob = math.pow(self.counter[item_idx][1], self.alpha) / total_count_pow
        for idx in range(self.table_size):
            self.table[idx] = item_idx
            try:
                if idx / self.table_size > item_prob:
                    item_idx += 1
                    item_prob += math.pow(self.counter[item_idx][1], self.alpha) / total_count_pow
            except:
                print("check the item_idx ", item_idx)
                continue

            if item_idx >= self.item_size-1:
                item_idx = item_idx - 1
        print("Done in %.2f seconds." %(time.time() - start_time))

    def sample_contexts(self, context):
        self.contexts[0] = context
        idx = 0
        while idx < self.neg_sample_size:
            neg_context = self.table[random.randrange(self.table_size)]
            if context != neg_context:
                self.contexts[idx+1] = neg_context
                idx += 1

    def train_stream(self, filename):
        print("Training...")

        start_time = time.time()
        c = 0
        uid_itemlist_dic = read_data(filename)
        items = [itemlist for itemlist in uid_itemlist_dic.values()]
        for cdx, user_itemlist in enumerate(items):
            self.window = len(user_itemlist)
            for idx, item in enumerate(user_itemlist):
                try:
                    reduced_window = random.randrange(self.window)
                    item_idx = self.item2idx[item]
                    self.items[0] = item_idx
                    # for jdx in range(idx - reduced_window, idx + reduced_window + 1):
                    for jdx in range(self.window):
                        context = user_itemlist[jdx]
                        if jdx != idx:
                            try:
                                context_idx = self.item2idx[context]
                                self.sample_contexts(context_idx)
                                self.train_pair(item_idx, self.contexts)
                                self.lr = max(self.min_lr, self.lr + self.decay)
                                c += 1
                                if c % 10000 == 0:
                                    loss = self.sess.run(self.loss, feed_dict={self.x: [item_idx], self.y: self.contexts})
                                    print("%d items trained in %.2f seconds. Learning rate: %.4f, Loss : %.4f" % (c, time.time() - start_time, self.lr, loss))
                            except:
                                continue
                except:
                    continue

    def get_sim_item(self, item_list, top_k):
        valid_arrays = np.array(item_list)

        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embed), 1, keepdims = True))
        normalized_embeddings = self.embed / norm
        valid_dataset = tf.constant(valid_arrays, dtype=tf.int32)

        valid_embeddings = tf.nn.embedding_lookup(
              normalized_embeddings, valid_dataset)

        similarity = tf.matmul(
              valid_embeddings, normalized_embeddings, transpose_b=True)

        sim = similarity.eval(session=self.sess)
        
        for i in range(len(item_list)):
            valid_item = self.idx2item[valid_arrays[i]]
            nearest = (-sim[i, :]).argsort()[1:top_k+1]
            log_str = "Nearest to %s:" % valid_item
            for k in range(top_k):
                close_item = self.idx2item[nearest[k]]
                log_str = "%s %s," % (log_str, close_item)
            print(log_str)


    def train_model(self, data):
        """train model"""
        self.train_stream(data)
