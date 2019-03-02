# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 10:42:29 2019

@author: 拔凉拔凉冰冰凉
"""
from queue import PriorityQueue as PQueue
from queue import Queue
import numpy as np
import random
np.random.seed(1)

class TreeNode:
    
    def __init__(self, frequency, key = None, is_leaf = False, dimission = 50):
        self.key = key
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.frequency = frequency
        self.father = None
        self.dimission = dimission
        self.vector = np.random.randn(dimission)
        
    
    def getSelfPath(self):
        if self.father != None:
            if self.father.left_child == self:
                return 1
            else:
                return -1
        else:
            return 0
    
    def __lt__(self, other):
        return self.frequency < other.frequency
    
class HafumanTree:
    
    leaf_list = []
    no_leaf_list = []
    word_location_dic = {}
    dimission = 50
    
    def __init__(self, node_dic, dimission):
        self.dimission = dimission
        length = 0
        pq = PQueue()
        lo = 0
        for word in node_dic:
            length += 1
            tree_node = TreeNode(node_dic[word], word, True, dimission)
            self.leaf_list.append(tree_node)
            pq.put(tree_node)
            self.word_location_dic[word] = lo
            lo += 1
        self.leaf_node_num = length
#        print("aaa")
        id_ = 0
        while length > 2:
            left_node = pq.get()
            right_node = pq.get()
#            print(length)
            left_word = left_node.key
            right_word = right_node.key
            
            add_value = left_node.frequency + right_node.frequency
            new_node = TreeNode(add_value, id_, dimission)
            if left_node.is_leaf == True:
                new_node.left_child = self.leaf_list[self.word_location_dic[left_word]]
                self.leaf_list[self.word_location_dic[left_word]].father = new_node
            else:
                new_node.left_child = self.no_leaf_list[int(left_word)]
                self.no_leaf_list[int(left_word)].father = new_node
            
            if right_node.is_leaf == True:
                new_node.right_child = self.leaf_list[self.word_location_dic[right_word]]
                self.leaf_list[self.word_location_dic[right_word]].father = new_node
            else:
                new_node.right_child = self.no_leaf_list[int(right_word)]
                self.no_leaf_list[int(right_word)].father = new_node
            
            pq.put(new_node)
            self.no_leaf_list.append(new_node)
            length -= 1
            id_ += 1
#        print("bbb")
        left_node = pq.get()
        right_node = pq.get()
        
        left_word = left_node.key
        right_word = right_node.key
            
        add_value = left_node.frequency + right_node.frequency
        self.root = TreeNode(add_value, id_, dimission)
        if left_node.is_leaf == True:
            self.root.left_child = self.leaf_list[self.word_location_dic[left_word]]
            self.leaf_list[self.word_location_dic[left_word]].father = self.root
        else:
            self.root.left_child = self.no_leaf_list[int(left_word)]
            self.no_leaf_list[int(left_word)].father = self.root
            
        if right_node.is_leaf == True:
            self.root.right_child = self.leaf_list[self.word_location_dic[right_word]]
            self.leaf_list[self.word_location_dic[right_word]].father = self.root
        else:
            self.root.right_child = self.no_leaf_list[int(right_word)]
            self.no_leaf_list[int(right_word)].father = self.root

        self.middle_node_num = id_ + 1
    
    def loadVec(self, vec_file, word_dic_file):
        vecs = np.load(vec_file)
        file = open(word_dic_file, "r", encoding = "utf-8")
        words = []
        lines = file.readlines()
        for line in lines:
            line = line.split("\n")[0]
            words.append(line)
        file.close()
        for i in range(len(words)):
            location = self.word_location_dic[words[i]]
            self.leaf_list[location].vector = vecs[i]
        print("has loaded vec!!!")
        
    
    def getLeftNode(self):
        node = self.root
        while node.is_leaf != True:
            node = node.left_child
        print(node.key)
    
    def getRightNode(self):
        node = self.root
        while node.is_leaf != True:
            node = node.right_child
        print(node.key)
    
    def getMaxDeepNode(self):
        q = Queue()
        max_deep = 0
        max_deep_node = self.root
        q.put((0,self.root))
        while q.empty() != True:
            (deep, node) = q.get()
            if deep > max_deep:
                max_deep = deep
                max_deep_node = node
            if node.left_child != None:
                q.put((deep+1, node.left_child))
            if node.right_child != None:
                q.put((deep+1, node.right_child))
        
        print(max_deep_node.key)
            
        
    def getMinDeepNode(self):
        q = Queue()
        min_deep = 1000000
        min_deep_node = None
        q.put((0,self.root))
        while q.empty() != True:
            (deep, node) = q.get()
            if deep < min_deep and node.is_leaf == True:
                min_deep = deep
                min_deep_node = node
            if node.left_child != None:
                q.put((deep+1, node.left_child))
            if node.right_child != None:
                q.put((deep+1, node.right_child))
        print(min_deep_node.key)
    
    def getWordPath(self, word):
        location = self.word_location_dic[word]
        node = self.leaf_list[location]
        path = []
        while node.father != None:
            path.append(node.getSelfPath())
            node = node.father
        path.reverse()
        return path
    
    def getWordByPath(self, path):
        node = self.root
        for p in path:
            if p == 1:
                node = node.left_child
            else:
                node = node.right_child
        print(node.key)
    
    def getPathWordAndVector(self, path):
        vecs = []
        ws = []
        vecs.append(self.root.vector)
        node = self.root
        for p in path:
            if p == 1:
                node = node.left_child
            else:
                node = node.right_child
            vecs.append(node.vector)
            ws.append(node.key)
        vecs = vecs[:-1]
        ws = ws[:-1]
        return ws, vecs
    
    def updateVectorByWord(self, word, is_leaf, vector_d):
#        print(word)
        if is_leaf == True:
            self.leaf_list[self.word_location_dic[word]].vector += vector_d
        else:
            self.no_leaf_list[int(word)].vector += vector_d
            
    def getVectorByWord(self, word, is_leaf):
        if is_leaf == True:
            return self.leaf_list[self.word_location_dic[word]].vector
        else:
            return self.no_leaf_list[int(word)].vector
    
    def save(self):
        out_file = "model\\new\\word_vec.npy"
        dic_file = "model\\new\\word_index.txt"
        all_vec = []
        out = open(dic_file, "w", encoding = "utf-8")
        for node in self.leaf_list:
            all_vec.append(node.vector)
            out.write(str(node.key) + "\n")
        all_vec = np.array(all_vec)
        np.save(out_file, all_vec)
        out.close()
            
        
        
        
class Batch:
    seqs = []
    batchs = []
    batch_size = 128
    
    def __init__(self, data, window, batch_size, word_p, min_time):
        self.batch_size = batch_size
        self.index = 0
        self.window = window
        self.word_p = word_p
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
        batch = []
        for i in range(batch_len):
            w = seq[i]
            b = max(0, i - self.window)
            e = min(batch_len-1, i + self.window)
            for j in range(b,i):
                qian = w
                hou = seq[j]
                if qian not in self.word_p:
                    qian = 'oov'
                if hou not in self.word_p:
                    hou = 'oov'
                batch.append([qian, hou])
            for j in range(i+1, e+1):
                qian = w
                hou = seq[j]
                if qian not in self.word_p:
                    qian = 'oov'
                if hou not in self.word_p:
                    hou = 'oov'
                batch.append([qian, hou])
#        print(batch)
        return batch, batch_len
    
    def next_batch(self):
        batch = []
        while len(self.batchs) < self.batch_size and self.index < len(self.seqs):
            seq = self.seqs[self.index]
            batch, batch_len = self.getSeqBatch(seq)
            for b in batch:
                self.batchs.append(b)
            self.index += 1
        
        batch_size = self.batch_size
        if len(self.batchs) < batch_size:
            batch_size = len(self.batchs)
#        print("aaa" + str(batch_size))
        batch = self.batchs[:batch_size]
        self.batchs = self.batchs[batch_size: len(self.batchs)]
#        print(len(self.batchs))
#        for i in range(batch_size):
#            batch.append(self.batchs.pop())
#        print(len(batch))
        return batch
    
    def seek(self):
        self.index = 0
        

    
class Client:
    
    filename = None
    word_p = {}
    learning_rate = 0.01
    epoch_num = 10
    batch_size = 128
    window = 5
    dimission = 50
    t = 0.00001
    
    def __init__(self, filename, min_time):
        self.filename = filename
        self.min_time = min_time
    
        
    def tongjiWordP(self):
        self.all_word_time = 0
        file = open(self.filename, "r", encoding="utf-8")
        lines = file.readlines()
#        print(lines[0])
        for line in lines:
            line = line.split("\n")[0]
            for w in line:
                if w not in self.word_p:
                    self.word_p[w] = 1
                else:
                    self.word_p[w] += 1
                self.all_word_time += 1
#        print(len(self.word_p))
#        print(self.word_p[''])
#        words = []
        new_word_p = {'oov': 0}
        for w in self.word_p:
            if self.word_p[w] < self.min_time:
                new_word_p['oov'] += self.word_p[w]
            else:
                new_word_p[w] = self.word_p[w]
        self.word_p = new_word_p
        self.word_pp = {}
        for w in self.word_p:
#            print(1.0 * self.word_p[w] / self.all_word_time)
            self.word_pp[w] = 1.0 * self.word_p[w] / self.all_word_time
        file.close()
    
    def buildHafumanTree(self):
        self.hafumanTree = HafumanTree(self.word_p, self.dimission)
        print("has build hafuman tree")
        
    def test(self):
        test_w = ['我', '，', '他', '北', '1', '红', '（', 'a', '三', '个', '月']
        w_vec = []
        w_list = []
        w_lo_dic = {}
        lo = 0
        k = 20
        for node in self.hafumanTree.leaf_list:
            w_list.append(node.key)
            w_vec.append(node.vector)
            w_lo_dic[node.key] = lo
            lo += 1
        for w in test_w:
            vec = w_vec[w_lo_dic[w]]
            dics = []
            for i in range(len(w_list)):
                dic = np.sqrt(np.sum(np.square(vec - w_vec[i])))
                dics.append(dic)
            index = np.argsort(dics)
            str_ = w + " : "
            for i in range(k):
                str_ += w_list[index[i]] + " "
            print(str_)

            
    def train(self):
        file = open(self.filename, "r", encoding="utf-8")
        lines = file.readlines()
        batch = Batch(lines, self.window, self.batch_size, self.word_p, self.min_time)
        file.close()
        for epoch in range(self.epoch_num):
            batch_index = 1
            print("epoch: " + str(epoch))
            seq_batch = batch.next_batch()
            
#            print(len(seq_batch))
            while True:
                seq_len = len(seq_batch)
                if batch_index % 10000  == 0 or batch_index == 1:
                    print("batch : " + str(batch_index))
                    self.test()
                update_dic = {}
                leaf_update_dic = {}
                for t in seq_batch:
#                    print(t)
                    p_discard = 1 - np.sqrt(self.t / self.word_pp[t[0]])
                    rand = random.random()
#                    print(rand)
                    if rand < p_discard:
#                        print("discard")
                        continue
                    path = self.hafumanTree.getWordPath(t[1])
#                    print(path)
                    ws, vecs = self.hafumanTree.getPathWordAndVector(path)
#                    print(ws)
                    w_vector = self.hafumanTree.getVectorByWord(t[0], True)
                    leaf_update_dic[t[0]] = np.zeros((self.dimission))
                    for i in range(len(ws)):
                        if ws[i] not in update_dic:
#                            print(ws[i])
                            update_dic[ws[i]] = np.zeros((self.dimission,))
#                        print(vecs[i])
#                        print(w_vector)
                        x = path[i]*np.dot(vecs[i], w_vector.T)
#                        print(x)
#                        if np.abs(x) > 150:
#                            update_dic[ws[i]] += w_vector
#                            leaf_update_dic[t[0]] +=  vecs[i]
#                        else:
#                            update_dic[ws[i]] += (1.0*np.exp(x)/(1+np.exp(x))) * w_vector
#                            leaf_update_dic[t[0]] += (1.0*np.exp(x)/(1+np.exp(x))) * vecs[i]
                        
                        update_dic[ws[i]] += (1.0*np.exp(x)/(1+np.exp(x))) * path[i] * w_vector
                        leaf_update_dic[t[0]] += (1.0*np.exp(x)/(1+np.exp(x))) * path[i] * vecs[i]
                        
#                        print(update_dic[ws[i]])
                        
#                        if batch_index > 1000:
#                            print(leaf_update_dic[t[0]])
#                print("update leaf")
                for w in leaf_update_dic:
#                    print(w)
                    self.hafumanTree.updateVectorByWord(w, True, -1.0*leaf_update_dic[w]*self.learning_rate/seq_len)
#                print("update no leaf")
                for w in update_dic:
#                    print(w)
                    self.hafumanTree.updateVectorByWord(w, False, -1.0*update_dic[w]*self.learning_rate/seq_len)
                if len(seq_batch) < self.batch_size:
                    break
                seq_batch = batch.next_batch()
                batch_index += 1
                
#            self.test()
            self.hafumanTree.save()
            batch.seek()
            
                    
        

    

client = Client("data\\news.txt", 5)

client.tongjiWordP()
client.buildHafumanTree()
client.hafumanTree.loadVec("model\\word_vec.npy", "model\\word_index.txt")

client.train()



#
#def test():
#        test_w = ['《', '‘', '，', '。', '？', '！', 'a', '他', '南', '男', 'z','你', '吃']
#        w_vec = []
#        w_list = []
#        w_lo_dic = {}
#        lo = 0
#        k = 5
#        for node in client.hafumanTree.leaf_list:
#            w_list.append(node.key)
#            w_vec.append(node.vector)
#            w_lo_dic[node.key] = lo
#            lo += 1
#        for w in test_w:
#            vec = w_vec[w_lo_dic[w]]
#            dics = []
#            for i in range(len(w_list)):
#                dic = np.sqrt(np.sum(np.square(vec - w_vec[i])))
#                dics.append(dic)
#            index = np.argsort(dics)
#            str_ = w + " : "
#            for i in range(k):
#                str_ += w_list[index[i]] + " "
#            print(str_)
#test()



#test(client)
#hafuman.getMinDeepNode()
#hafuman.getMaxDeepNode()
#path = hafuman.getWordPath('oov')
#print(path)
#hafuman.getWordByPath(path)
#print(client.word_p['oov'])
#print(len(client.word_p))