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
import itertools
from function import s_score, word_polish, z_score, mean_sp
from function import save_pkl, load_pkl
from function import z_test, max_module_size,tokenize


#%%
### read conversation
import jieba
import jieba.analyse

jieba.set_dictionary('D:/research/openai/yucan/model/dict.txt')
jieba.load_userdict('D:/research/openai/yucan/model/hk_dict.txt')

#============#-----------tokenize start
'''
pre_df2 = pd.read_csv('D:/research/openai/openupCOVID/data/2020_valid_join.csv')
pre_df2.dropna(subset=['msg'], inplace=True)
pre_df2['is_helpseeker'] = pre_df2['is_helpseeker'].astype(str)
print (set(list(pre_df2['is_helpseeker'])))
pre_df2 = pre_df2[pre_df2['is_helpseeker']=='True'] #1275713 vs 1469764
pre_df2['message'] = pre_df2['msg'].astype(str) 

#tokenize(df2) 
'''
#============#-----------tokenize end

pre_df2 = pd.read_csv('tokenized.csv')
pre_df2.dropna(subset=['tokenize'], inplace=True)
#---------risky row
suicidal = ['想死','自殺','跳樓', '離開世界', '逃離', '死咗', '天台', '遺書', 
 '落去', '跳落去', '安樂死', '尋死', '去死', 
 '跌落','介手','界手','界刀','不想活','跳樓','快D死','快d死']

temp = []
for i in pre_df2['tokenize']:
    line = i.strip().split(' ')
    if len(set(line).intersection(set(suicidal)))>0:
        temp.append(1)
    else:
        temp.append(0)
pre_df2['risky_index'] = temp
pre_df2.sort_values(by = ['conversationId','datetime'], ascending=True, inplace=True, ignore_index=True) #this is important
print (len((set(list(pre_df2['conversationId'])))))
pp = pre_df2[pre_df2['pingpongCount']>10]
print (len((set(list(pp['conversationId'])))))

'''
clst = ['問題','網上','主動','學校','半年','返學','還好','社工','嚴重']
ppp = pp.groupby('conversationId')['tokenize'].sum()
for i in ppp.index:
    l = ppp[i].strip().split(' ')
    if set(l).intersection(set(clst)) == set(clst):
        print (i)    
aaa = pp[pp['conversationId'] == 'cfa4a8ce-6ee4-426f-9e8c-ac6ae2b04e16']
'''
#%% 
#-----------position
temp = [0]
c=0
for i, j in zip( pre_df2['conversationId'][1:],  pre_df2['conversationId'][:-1]):
    if i==j:
        c = c+1
        temp.append(c)
    else:
        c = 0
        temp.append(c)
pre_df2['position'] = temp
aa = pre_df2[:500]
#-----------position of last sentence
last_position = dict()
last_sentence = pre_df2.drop_duplicates(subset='conversationId', inplace=False, keep='last', ignore_index=False) 
for i in last_sentence.index:
    last_position[last_sentence.loc[i, 'conversationId']] = last_sentence.loc[i, 'position']
temp = []
for i in pre_df2['conversationId']:
    temp.append(last_position[i])
pre_df2['last_position'] = temp

#---最后一句到目标句的gap
gap = np.array(pre_df2['last_position'])-np.array(pre_df2['position'])
pre_df2['gap'] = gap
#%%---risky block
#---first risky sentence position is valid
risky_sentence_only = pre_df2[pre_df2['risky_index']==1] 
first_risky_sentence = risky_sentence_only.drop_duplicates(subset=['conversationId'], inplace=False, keep='first', ignore_index=False) 
valid_risky_index = first_risky_sentence[first_risky_sentence['position']>=16]    #position of the first_risky_sentence

#---last sentence position is valid 
valid_risky_index = valid_risky_index[valid_risky_index['gap']>=5 ]

risky_msg = []
for i in valid_risky_index.index:
    ambient =  pre_df2.loc[i, 'tokenize']+ \
    pre_df2.loc[i+1, 'tokenize']+ pre_df2.loc[i+2, 'tokenize']+ pre_df2.loc[i+3, 'tokenize']+ pre_df2.loc[i+4, 'tokenize']+ pre_df2.loc[i+5, 'tokenize']+ \
    pre_df2.loc[i-1, 'tokenize']+ pre_df2.loc[i-2, 'tokenize']#+ pre_df2.loc[i-3, 'tokenize']+ pre_df2.loc[i-4, 'tokenize']+ pre_df2.loc[i-5, 'tokenize']
    risky_msg.append(ambient)

risky_words_lst = []
for i in risky_msg:
    risky_words_lst.append(i.strip().split(' '))
    
complete_risky_msg = []  #--------需要有顺序
for i in valid_risky_index.index:
    ambient =  pre_df2.loc[i-2, 'msg']+' '+ pre_df2.loc[i-1, 'msg']+' '+pre_df2.loc[i, 'msg']+' '+\
    pre_df2.loc[i+1, 'msg']+' '+ pre_df2.loc[i+2, 'msg']+' '+ pre_df2.loc[i+3, 'msg']+' '+ pre_df2.loc[i+4, 'msg']+' '+\
    pre_df2.loc[i+5, 'tokenize']
    complete_risky_msg.append(ambient)
#%%
#---moderate block
moderate = pre_df2[~pre_df2['conversationId'].isin(first_risky_sentence['conversationId'])]  
moderate = moderate[(moderate['position']>=5) & (moderate['gap']>=16)]  
moderate = moderate.sample(n=valid_risky_index.shape[0], random_state=40)
    
moderate_msg = []
for i in moderate.index:
    ambient =  pre_df2.loc[i, 'tokenize']+\
    pre_df2.loc[i+1, 'tokenize']+ pre_df2.loc[i+2, 'tokenize']+ pre_df2.loc[i+3, 'tokenize']+ pre_df2.loc[i+4, 'tokenize']+\
    pre_df2.loc[i-1, 'tokenize']+ pre_df2.loc[i-2, 'tokenize']+ pre_df2.loc[i-3, 'tokenize']+ pre_df2.loc[i-4, 'tokenize']
    moderate_msg.append(ambient)

moderate_words_lst = []
for i in moderate_msg:
    moderate_words_lst.append(i.strip().split(' '))    
#%%
#-----------------block before high risk bloc
msg_before = []
for i in valid_risky_index.index:
    lookback = 1
    before = pre_df2.loc[i-lookback-0, 'tokenize'] + pre_df2.loc[i-lookback-1, 'tokenize']  + pre_df2.loc[i-lookback-2, 'tokenize']+\
    pre_df2.loc[i-lookback-3, 'tokenize'] + pre_df2.loc[i-lookback-4, 'tokenize']  + pre_df2.loc[i-lookback-5, 'tokenize']+\
    pre_df2.loc[i-lookback-6, 'tokenize'] + pre_df2.loc[i-lookback-7, 'tokenize']  + pre_df2.loc[i-lookback-8, 'tokenize']
    msg_before.append(before)

before_words_lsttt = []
for i in msg_before:
    before_words_lsttt.append(i.strip().split(' '))
before_words_lst = [i[1] for i in list(enumerate(before_words_lsttt))]
before_index = [i[0] for i in list(enumerate(before_words_lsttt))]

#-----------------complete_msg_before(为了看具体对话)
complete_msg_before = []
for i in valid_risky_index.index:
    complete_before = pre_df2.loc[i-lookback-8, 'msg']+' '+ pre_df2.loc[i-lookback-7, 'msg']+' '+ pre_df2.loc[i-lookback-6, 'msg']+' '+\
    pre_df2.loc[i-lookback-5, 'msg']+' '+ pre_df2.loc[i-lookback-4, 'msg']+' '+ pre_df2.loc[i-lookback-3, 'msg']+' '+\
    pre_df2.loc[i-lookback-2, 'msg']+' '+ pre_df2.loc[i-lookback-1, 'msg']+' '+ pre_df2.loc[i-lookback-0, 'msg']
    complete_msg_before.append(complete_before)

Id = valid_risky_index.loc[valid_risky_index.index[0], 'conversationId']
aa = pre_df2[pre_df2[ 'conversationId'] == Id]
#%%
### read network
#networkfile = 'words_simi_network_fewENGwords.csv'

#---w2v only
networkfile = 'words_simi_network_30.csv'
df_network =  pd.read_csv('D:/research/openai/openup2.0/word2vec/network_collection/'+networkfile)

'''
networkfile = 'combinatory_network.csv'
print(networkfile)
df_network =  pd.read_csv('../word2vec/'+networkfile)
df_network = df_network[df_network['weight']>0.0]
'''
G = nx.Graph()

for m, n, w in zip(df_network['node1'], df_network['node2'], df_network['weight']):
    
    G.add_edge(m,n,weight= 1/w)
   
print (nx.number_connected_components(G))
print ('# of nodes: ', nx.number_of_nodes(G), '# of edges: ', nx.number_of_edges(G))

nbunch = sorted(nx.connected_components(G), key=len, reverse=True)    
G = G.subgraph(nbunch[0])

print (nx.number_connected_components(G))
print ('# of nodes: ', nx.number_of_nodes(G), '# of edges: ', nx.number_of_edges(G))
#%%

risky_words_lst    = [word_polish(i, G.nodes()) if len(word_polish(i, G.nodes()))>0 else '' for i in risky_words_lst ]
moderate_words_lst = [word_polish(i, G.nodes()) if len(word_polish(i, G.nodes()))>0 else '' for i in moderate_words_lst ]  
before_words_lst   = [word_polish(i, G.nodes()) if len(word_polish(i, G.nodes()))>0 else '' for i in before_words_lst   ]

#%%

goodId = []
goodId.append (valid_risky_index.iloc[134,0])
goodId.append (valid_risky_index.iloc[161,0])
goodId.append (valid_risky_index.iloc[175,0])
goodId.append (valid_risky_index.iloc[238,0])
'''
#goodId[
     #'1345fc79-c840-4792-9ce3-425cb4db9ed6',
     #'168584ea-d4ab-4413-8cee-2ac7ae4cd083',
     #'181c731a-8002-46a2-bb72-08fa1e67099d',
     #'1fc76c1b-41c0-42ba-96ce-afb1355bfa3d']
#df_ori = pd.read_csv('D:/research/openai/openupCOVID/data/2020_valid_join.csv')
aa = df_ori[df_ori['conversationId']=='eb5a9426-35f6-4ddb-b4bf-f39fe5917e59']
helpseeker = ['H: ' if i else 'C: ' for i in aa['is_helpseeker']]
HCmsg = np.array(helpseeker)+np.array(aa['msg'])
aa['HCmsg'] = HCmsg
'''
#--index in xx_lst:
#goodId_b =   '1345fc79-c840-4792-9ce3-425cb4db9ed6'    #valid_risky_index.iloc[134,0]
#goodId_r =  '2b882192-5412-4093-b92d-67906d5fc838'    #valid_risky_index.iloc[328,0]
#goodId_m =  'eb5a9426-35f6-4ddb-b4bf-f39fe5917e59'    #moderate.iloc[22,0]
goodIndex_b = [134]
goodIndex_r = [328]
goodIndex_br = goodIndex_b+goodIndex_r
goodId=[]
for i in goodIndex_br:
    goodId.append(valid_risky_index.loc[valid_risky_index.index[i], 'conversationId'])
    
goodIndex_m = [22]
for i in goodIndex_m:
    goodId.append(moderate.loc[moderate.index[i], 'conversationId'])

#%% 
before_words_lst   = [before_words_lst[i] for i in goodIndex_b]
risky_words_lst    = [risky_words_lst[i] for i in goodIndex_r]
moderate_words_lst = [moderate_words_lst[i] for i in goodIndex_m]

#%%
#------save the path length
pathlength = dict()

ct = 0
for lst in risky_words_lst:
    ct = ct+1
    lst = word_polish(lst, G.nodes())
    for w in lst:
        if w not in pathlength: pathlength[w] = nx.single_source_shortest_path_length(G, w)
    if ct%10==0: print (ct)

ct = 0
for lst in moderate_words_lst:
    ct = ct+1
    lst = word_polish(lst, G.nodes())
    for w in lst:
        if w not in pathlength: pathlength[w] = nx.single_source_shortest_path_length(G, w)
    if ct%10==0: print (ct)

ct = 0
for lst in before_words_lst:
    ct = ct+1
    lst = word_polish(lst, G.nodes())
    for w in lst:
        if w not in pathlength: pathlength[w] = nx.single_source_shortest_path_length(G, w)
    if ct%10==0: print (ct)
    
ct = 0
random.seed(40)
random_nodes = random.sample(G.nodes(),10)
for w in random_nodes:
    ct = ct+1
    pathlength[w] = nx.single_source_shortest_path_length(G, w)
    if ct%100==0: print (ct)
    
nodes = G.nodes()
#save_pkl(pathlength, 'pathlength_dict.pkl')
#pathlength = load_pkl( 'pathlength_dict.pkl')
    
#%%

#--------distance
print (s_score(before_words_lst[0], risky_words_lst[0], nodes, pathlength))
model = word2vec.KeyedVectors.load_word2vec_format('D:/research/openai/openup2.0/word2vec/open_cn_word2vec_fewENGwords.model', binary=False)  #!!   

#%% viz
caseId = 0
moderate_words_lst[caseId] = [i for i in moderate_words_lst[caseId] if i not in risky_words_lst[caseId]]
moderate_words_lst[caseId] = [i for i in moderate_words_lst[caseId] if i not in before_words_lst[caseId]]
before_words_lst[caseId]   = [i for i in before_words_lst[caseId]   if i not in risky_words_lst[caseId]]


#----------node file node_id: input words, output id. what in G are words
node_id, id_node = dict(), dict()
c = 0
for i in G.nodes():
    c=c+1
    node_id[i] = c
    id_node[c] = i
    
viz = pd.DataFrame()
viz['Source'] = [node_id[i[0]] for i in G.edges()]
viz['Target'] = [node_id[i[1]] for i in G.edges()]


feature = dict()
for i in G.nodes():
    feature[i] = 0

for i in before_words_lst[caseId]:
    feature[i] = 1

for i in risky_words_lst[caseId]:
    feature[i] = 2
    
for i in moderate_words_lst[caseId]:
    feature[i] = 3
    
f = open('node_feature.csv','w',encoding='utf-8')
f.write('id'+','+'name'+','+'feature'+'\n')
for i in feature:
    f.write(str(node_id[i])+','+i+','+str(feature[i])+'\n')
f.close()    

##=============edge file
weight = dict()

for i, j in zip(viz['Source'], viz['Target']):
    weight[(i, j)] = 1
    weight[(j, i)] = 1
    
#-----------add edges to viz
for j in list(itertools.combinations(before_words_lst[caseId], 2)):
    i, k = j[0], j[1]
    viz = viz.append([{'Source':node_id[i], 'Target':node_id[k]}])
    try :
        weight[(node_id[i], node_id[k])] 
        weight[(node_id[i], node_id[k])]=10
    except:                             
        weight[(node_id[i], node_id[k])]=11
        
for j in list(itertools.combinations(risky_words_lst[caseId], 2)):
    i, k = j[0], j[1]
    viz = viz.append([{'Source':node_id[i], 'Target':node_id[k]}])
    try :
        weight[(node_id[i], node_id[k])] 
        weight[(node_id[i], node_id[k])]=10
    except:                             
        weight[(node_id[i], node_id[k])]=12
        
for j in list(itertools.combinations(moderate_words_lst[caseId], 2)):
    i, k = j[0], j[1]
    viz = viz.append([{'Source':node_id[i], 'Target':node_id[k]}])
    try :
        weight[(node_id[i], node_id[k])] 
        weight[(node_id[i], node_id[k])]=10
    except:                             
        weight[(node_id[i], node_id[k])]=13
        
#=======path and weight
b_words = ['坐監', '吸毒', '打人']    
r_words = ['想死', '跳落去', '覆診']  
m_words = ['尷尬',  '放工', '糾結'] 
path, length = dict(), dict()
for i in b_words:
    for j in r_words:
        path[(i, j)] = nx.shortest_path(G, source=i, target=j)
        length[(i, j)] = len(path[(i, j)])
    
for i in m_words:
    for j in r_words:
        path[(i, j)] = nx.shortest_path(G, source=i, target=j)
        length[(i, j)] = len(path[(i, j)])
        
for i in path:
    if length[i]>1:
        for e1, e2 in zip( path[i][:-1], path[i][1:]): 
            weight[(node_id[e1], node_id[e2])] = length[i]
            weight[(node_id[e2], node_id[e1])] = length[i]

#%%

#----column 3
wt=[]
for i, j in zip(viz['Source'], viz['Target']):
    wt.append(weight[(i, j)])
viz['Weight'] = wt

dfbase = viz[viz['Weight']==1]
dfbase = dfbase.sample(n=12000)
dfcolor = viz[viz['Weight']>1]
vizf = dfbase.append(dfcolor,ignore_index=True)

edg=[]
for i, j in zip(vizf['Source'], vizf['Target']):
    edg.append(set([i,j]))
vizf['edg'] = edg    
vizf['edg'] =vizf['edg'].astype(str)
vizf = vizf.drop_duplicates(subset='edg')
vizf.to_csv('viz.csv', index= False)


