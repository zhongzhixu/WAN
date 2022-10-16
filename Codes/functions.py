from gensim.models import word2vec
import gensim.models.keyedvectors as word2vec
from sklearn.metrics import classification_report
import opencc
import re
import emoji
import pandas as pd
from copy import deepcopy
import networkx as nx
import random 
import numpy as np
import jieba
#### s score 
def s_score(moduleA, moduleB, nodes, pathlength):  
    #------------------
    #moduleA, moduleB: lists of nodes in A and B;
    #nodes: valid nodes (i.e. in the network).
    
    moduleA = set(moduleA).intersection(set(nodes))
    moduleB = set(moduleB).intersection(set(nodes))
    
    dAB, dAA, dBB = 0, 0, 0
    wet = None
    #wet = 'weight'
    for i in moduleA:
        for j in moduleB: 
            try: dAB = dAB + pathlength[i][j] #nx.shortest_path_length(G, i,j,weight = wet) 
            except: print ('ERROR', i,j)
    dAB = dAB/(len(moduleA)*len(moduleB))
     
    #===========-dAA-===========
    moduleA_copy = deepcopy(moduleA)
    for i in moduleA:
        for j in moduleA_copy:
            try: dAA = dAA + pathlength[i][j]
            except: print ('ERROR', i,j)         
        moduleA_copy.remove(i)
        
    dAA=dAA*2
    ctA = len(moduleA)*len(moduleA)     
    dAA = dAA/ctA
    
    #===========-dBB-===========
    moduleB_copy = deepcopy(moduleB)
    for i in moduleB:
        for j in moduleB_copy:
            try: dBB = dBB + pathlength[i][j] 
            except: print ('ERROR', i,j)
        moduleB_copy.remove(i)

    dBB=dBB*2
    ctB = len(moduleB)*len(moduleB) 
    dBB = dBB/ctB
    
    rs_score = dAB-(dAA+dBB)/2
    return rs_score
 
#---  
def mean_sp(module, nodes, pathlength):
    module = set(module).intersection(set(nodes))
    d = 0
    
    module_copy = deepcopy(module)
    for i in module:
        for j in module_copy:
            try: d = d + pathlength[i][j]
            except: print ('ERROR', i,j)         
        module_copy.remove(i)
        
    d=d*2
    ct = len(module)*len(module)     
    d = d/ct  
    return d
    
def word_polish (lst, nodes):
    lst = list(set(lst).intersection(set(nodes)))
    return lst

def z_score(moduleA, moduleB, G): #from A to B
    wet = None
    #wet = 'weight'
    dAB = 0
    sp, min_sp =[], []
    
    for i in moduleA:
        for j in moduleB:
            try:
                sp.append(nx.shortest_path_length(G, i,j,weight = wet))
            except: 
                print ('ERROR', i,j)
        min_sp.append(min(sp))
        
    dAB = sum(min_sp)/len(min_sp)
    return dAB
    
## weight 不要用反了！ weight越大关系越紧密，shortest path 算做路阻了却
  
import pickle
def save_pkl(obj, pkl_name):
    with open(pkl_name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(pkl_name):
    with open(pkl_name, 'rb') as f:
        return pickle.load(f) 

from math import sqrt
def z_test(lst_a, lst_b):
    z_value = (np.mean(lst_a)-np.mean(lst_b)) / sqrt( np.std(lst_a)/len(lst_a)+np.std(lst_b)/len(lst_b) )
    print ('mean', round(np.mean(lst_a),3),round(np.mean(lst_b),3),'std',round(np.std(lst_a),3),round(np.std(lst_b),3))
    return round(z_value,3)

def max_module_size(nodes, graph):
    H = graph.subgraph(nodes)
    graphs = sorted(nx.connected_components(H), key=len, reverse=True)
    return len(graphs[0])
        
def tokenize(df2):
    #-------------read stopword
    stopword = []
    f = open('D:/research/openai/openup2.0/word2vec/stopwords.txt','r', encoding='utf-8')
    while True:
        line = f.readline()
        if not line: break
        l = line.strip('\n')
        stopword.append(l)
    
    #--字符操作和严格匹配不同的
    import emoji    
    eng_words = ['selfharm','DSE','dse','快D死','快d死','sad','die']
    label = ['鬼魍一','鬼魍二','鬼魍三','鬼魍四','鬼魍五','鬼魍六','鬼魍七']
    for i, j in zip(eng_words, label):
        df2['message'] = df2['message'].replace(i, j)
    
    #----------delete emoji and english (except selfharm, DSE, etc.)
    
    df2['message'] = df2['message'].str.replace(
        r'[\x00-\x7F]+', ' ').replace('\s+', ' ').apply(
        lambda x: ''.join(c for c in x if c not in emoji.UNICODE_EMOJI) ).str.strip()
    
           
    for i, j in zip(eng_words, label):
        df2['message'] = df2['message'].replace(j, i)
    
    
    #----------delete nan
    import numpy as np
    df2['message'].replace('', np.nan, inplace=True)
    df2 = df2.dropna(subset=['message'])
    
    #----------remove all punctuations
    df2['message'] = df2['message'].apply(lambda x: re.sub("[\s+\!\/_,$%^*(+\"\']+|[+——！，？、~@#￥%……&*（）]+", "",x))
    df2['message'] = df2['message'].apply(lambda x: re.sub("\.\。\【\】", " ",x))
    df2 = df2.dropna(subset=['message'])
    
    #----------to cantonese
    import opencc
    cc = opencc.OpenCC('s2hk')
    def to_hk(text):
        return cc.convert(text)
    
    df2['message'] = df2['message'].apply(lambda x: to_hk(x))
    train_texts_orig = df2['message'].to_list()

    #-----------output
    jieba.set_dictionary('D:/research/openai/yucan/model/dict.txt')
    jieba.load_userdict('D:/research/openai/yucan/model/hk_dict.txt')
    
    for word in ['上咗癮','食到死咗','selfharm','DSE','dse','快D死','快d死','sad','die']:
        jieba.add_word(word, freq=None, tag=None)
    
    #---output
    column = []
    for i in ['武']:
        stopword.append(i)
    for text in train_texts_orig:
        text = text.strip('\n')
        words = jieba.cut(text, cut_all=False)
        row = ''
        for word in words:
            if word not in stopword:
                row = row + word + ' '
        column.append(row)
    df2 ['tokenize'] = column
    df2.to_csv('tokenized.csv', index=False)





