from __future__ import print_function

import tensorflow as tf
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import random
random.seed(1)

class Config:
    batch_size = 128
    window = 5
    min_time = 5
    oov = 'oov'
    dimission = 50
    learn_rate = 0.005
    epoch_num = 5
    displayStep = 500
    data_file = "data\\news.txt"
    model_dir = "model\\nn\\"
    
    

class Batch:
    seqs = []
    x_batch = []
    y_batch = []
    batch_size = 128
    
    def __init__(self, data, window, batch_size, word_p, min_time):
        self.batch_size = batch_size
        self.index = 0
        self.window = window
        self.word_p = word_p
        self.VocaNum = len(self.word_p)
        self.min_time = min_time
        for line in data:
            line = line.split("\n")[0]
            line = line.split(" ")
            for seq in line:
                if seq != " " and seq != "":
                    self.seqs.append(list(seq))
        for j in range(len(self.seqs)):
            for i in range(len(self.seqs[j])):
                if self.seqs[j][i] not in self.word_p:
                    self.seqs[j][i] = 'oov'
        print("has build batch!!! seqs length : " + str(len(self.seqs)))
     
    def getSeqBatch(self, seq):
        batch_len = len(seq)
        x_batch = []
        y_batch = []
        for i in range(batch_len):
            target = [0.01] * self.VocaNum
            p_min = (1.0-0.9) / self.VocaNum
            x = [p_min] * self.VocaNum
            w = seq[i]
            b = max(0, i - self.window)
            e = min(batch_len-1, i + self.window)
            
            x[self.word_p[w]] = 0.9
            
            for j in range(b,i):
                hou = seq[j]
#                if hou not in self.word_p:
#                    hou = 'oov'
                target[self.word_p[hou]] = 0.99
#                print(self.word_p[hou])
            for j in range(i+1, e+1):
                hou = seq[j]
#                if hou not in self.word_p:
#                    hou = 'oov'
                target[self.word_p[hou]] = 0.99
#                print(self.word_p[hou])
            x_batch.append(x)
            y_batch.append(target)
#        print(batch)
#        return x_batch, y_batch, batch_len
            return y_batch, x_batch, batch_len
    
    def next_batch(self):
        x_batch = []
        y_batch = []
        while len(self.x_batch) < self.batch_size and self.index < len(self.seqs):
            seq = self.seqs[self.index]
            xbatch, ybatch, batch_len = self.getSeqBatch(seq)
            for i in range(len(xbatch)):
                self.x_batch.append(xbatch[i])
                self.y_batch.append(ybatch[i])
            self.index += 1
        
        batch_size = self.batch_size
        if len(self.x_batch) < batch_size:
            batch_size = len(self.x_batch)
#        print("aaa" + str(batch_size))
        x_batch = self.x_batch[:batch_size]
        y_batch = self.y_batch[:batch_size]
        self.x_batch = self.x_batch[batch_size: len(self.x_batch)]
        self.y_batch = self.y_batch[batch_size: len(self.y_batch)]
#        print(len(self.batchs))
#        for i in range(batch_size):
#            batch.append(self.batchs.pop())
#        print(len(batch))
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        return x_batch, y_batch
    
    def seek(self):
        self.index = 0
        
class Model:
    
    def __init__(self, config):
        self.config = config
        self.sess = tf.Session()
    
    def tongji(self):
        self.word_dic = {}
        word_p = {}
        file = open(self.config.data_file, "r", encoding = "utf-8")
        lines = file.readlines()
        for line in lines:
            line = line.split("\n")[0]
            for w in line:
                if w not in self.word_dic:
                    self.word_dic[w] = 1
                else:
                    self.word_dic[w] += 1
        word_p = {'oov':0}
        for w in self.word_dic:
            if self.word_dic[w] < self.config.min_time:
                word_p['oov'] += self.word_dic[w]
            else:
                word_p[w] = self.word_dic[w]
                
        words = []
        p = []
        for w in word_p:
            words.append(w)
            p.append(word_p[w])
        sort_index = np.argsort(p)
        self.word_dic = {}
        for i in range(len(sort_index)):
            self.word_dic[words[sort_index[i]]] = i
        self.word_list = []
        for w in self.word_dic:
            self.word_list.append(w)
        self.VocaNum = len(self.word_dic)
        print('has tongji.  vaconum:' + str(self.VocaNum))
    
    def init(self):
        self.X = tf.placeholder('float', [None, self.VocaNum], name = "X")
        self.Y = tf.placeholder('float', [None, self.VocaNum], name = "Y")
        self.WordVec = tf.Variable(tf.random_normal([self.VocaNum, self.config.dimission]),
                            name = "WordVec")
        
        self.weight = tf.Variable(tf.random_normal([self.config.dimission, self.VocaNum]),
                            name = "Weight")
        
        self.biase = tf.Variable(tf.random_normal([self.VocaNum]), name = "biase")
        self.output1 = tf.matmul(self.X, self.WordVec)
        self.output2 = tf.nn.tanh(self.output1)
        self.output3 = tf.matmul(self.output2, self.weight) + self.biase
        self.output4 = tf.nn.tanh(self.output3)
        self.output = tf.nn.softmax(self.output4)
        self.loss_op = tf.reduce_mean(tf.abs(self.Y - self.output))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.config.learn_rate, name="GDO")
        self.train_op = self.optimizer.minimize(self.loss_op)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print('has init model!!!')
        
    def test(self):
        test_w = ['我', '，', '他', '北', '1', '红', '（', 'a', '三', '个', '月']
#        w_lo_dic = {}
#        lo = 0
        w_vec = self.WordVec.eval(session = self.sess)
        k = 20
#        for node in self.hafumanTree.leaf_list:
#            w_list.append(node.key)
#            w_vec.append(node.vector)
#            w_lo_dic[node.key] = lo
#            lo += 1
        for w in test_w:
            vec = w_vec[self.word_dic[w]]
            dics = []
            for i in range(self.VocaNum):
                dic = np.sqrt(np.sum(np.square(vec - w_vec[i])))
                dics.append(dic)
            index = np.argsort(dics)
            str_ = w + " : "
            for i in range(k):
                str_ += self.word_list[index[i]] + " "
            print(str_)
        
    def train(self):
        file = open(self.config.data_file, "r", encoding = "utf-8")
        lines = file.readlines()
        saver=tf.train.Saver(max_to_keep = 4)
        batch = Batch(lines, self.config.window, self.config.batch_size, self.word_dic, self.config.min_time)
        for epoch in range(1, self.config.epoch_num + 1):
            print('epoch : ' + str(epoch))
            step = 1
            batch_x, batch_y = batch.next_batch()
            while True:
#                print(batch_y)
                self.sess.run(self.train_op, feed_dict={self.X: batch_x, 
                                                        self.Y: batch_y})
                loss = self.sess.run(self.loss_op, feed_dict={self.X: batch_x, 
                                                        self.Y: batch_y})
                if step % self.config.displayStep == 0 or step == 1:
                    print("step : " + str(step) + "   loss : " + str(loss))
                    self.test()
                if len(batch_x) < self.config.batch_size:
                    break
                step += 1
                batch_x, batch_y = batch.next_batch()
            saver.save(self.sess, self.config.model_dir, global_step = epoch)
            batch.seek()
            
            
config = Config()
model = Model(config)
model.tongji()
model.init()
model.train()
        
