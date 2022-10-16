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

from function import s_score, word_polish, z_score, mean_sp
from function import save_pkl, load_pkl
from function import z_test, max_module_size
#%%
#read stopword
stopword = []
f = open('D:/research/openai/openup2.0/word2vec/stopwords.txt','r', encoding='utf-8')
while True:
    line = f.readline()
    if not line: break
    l = line.strip('\n')
    stopword.append(l)
#%%
### read conversation
import jieba
import jieba.analyse

jieba.set_dictionary('D:/research/openai/yucan/model/dict.txt')
jieba.load_userdict('D:/research/openai/yucan/model/hk_dict.txt')


df_conversation1 =  pd.read_excel('../submission to CT/submission to CT/0917_openup_evaluation_0_MC.xlsx')
df_conversation2 =  pd.read_excel('../submission to CT/submission to CT/0917_openup_evaluation_1_MC.xlsx')
df_conversation3 =  pd.read_excel('../submission to CT/submission to CT/0917_openup_evaluation_4_FC.xlsx')
df_conversation4 =  pd.read_excel('../submission to CT/submission to CT/0917_openup_evaluation_5_FC.xlsx')

df_conversation =  df_conversation1.append(df_conversation2,ignore_index=True)
df_conversation =  df_conversation.append(df_conversation3,ignore_index=True)
df_conversation =  df_conversation.append(df_conversation4,ignore_index=True)

indx = []

temp = []
for i in df_conversation['Unnamed: 0']:
    l = i.strip().split('\n')
    for j in l:
        if (('S:' in j) or ('C:' in j)):
            l.remove(j)
    strr = ' '.join(l)
    temp.append(strr)
df_conversation['Unnamed: 0'] = temp 


#%%
import jieba

import pandas as pd
df2 = pd.DataFrame()
df2['message'] = df_conversation['Unnamed: 0'].astype(str)

#%%
#字符操作和严格匹配不同的
import emoji
train2 = df2[['message']].copy()
a = df2[:100]
eng_words = ['selfharm','DSE','dse','快D死','快d死','sad','die']
label = ['鬼魍一','鬼魍二','鬼魍三','鬼魍四','鬼魍五','鬼魍六','鬼魍七']
for i, j in zip(eng_words, label):
    train2['message'] = train2['message'].replace(i, j)

#----------delete emoji and english (except selfharm, DSE, etc.)

train2['message'] = train2['message'].str.replace(
    r'[\x00-\x7F]+', ' ').replace('\s+', ' ').apply(
    lambda x: ''.join(c for c in x if c not in emoji.UNICODE_EMOJI) ).str.strip()

       
for i, j in zip(eng_words, label):
    train2['message'] = train2['message'].replace(j, i)

#----------delete nan
import numpy as np
train2['message'].replace('', np.nan, inplace=True)
train2 = train2.dropna()

#----------remove all punctuations
train2['message'] = train2['message'].apply(lambda x: re.sub("[\s+\!\/_,$%^*(+\"\']+|[+——！，？、~@#￥%……&*（）]+", "",x))
train2['message'] = train2['message'].apply(lambda x: re.sub("\.\。\【\】", " ",x))
train2 = train2.dropna()

#----------to cantonese
import opencc
cc = opencc.OpenCC('s2hk')
def to_hk(text):
    return cc.convert(text)

train2['message'] = train2['message'].apply(lambda x: to_hk(x))
train_texts_orig = train2['message'].to_list()
#%%tokenize
#-----------output  (run when necessary)
#output = open('keywords_corpus.txt', 'w', encoding='utf-8')

jieba.set_dictionary('D:/research/openai/yucan/model/dict.txt')
jieba.load_userdict('D:/research/openai/yucan/model/hk_dict.txt')

for word in ['上咗癮','食到死咗','快d死','俾人欺蝦']:
    jieba.add_word(word, freq=None, tag=None)
    
for text in train_texts_orig:
    text = text.strip('\n')
    words = jieba.cut(text, cut_all=False)
    for word in words:
        if word not in stopword:
            output.write(word + ' ')
    output.write('\n')

#%%
# keywords
#  how many words in total 
textlist=[]
all_words = [] 
f = open('keywords_corpus.txt', 'r', encoding='utf-8')    
while True:
    line = f.readline()
    if not line: break
    l = line.strip().split(' ')    
    textlist.append(l)
    for i in l:
        all_words.append(i)
f.close()
all_words = set(all_words)

#%%
#words filtering
import jieba.analyse
import numpy as np
f = open('D:/research/openai/openup2.0/distance_on_graph/keywords_corpus.txt', 'r', encoding='utf-8')
tt = f.read()
f.close()
wordss = jieba.analyse.extract_tags(tt, topK=3000, withWeight=False, allowPOS=())       
np.save('D:/research/openai/openup2.0/word2vec/testset_words.npy', wordss)
#x = np.load('testset_words.npy')


