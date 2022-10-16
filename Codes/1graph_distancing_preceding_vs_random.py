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

#tokenize(pre_df2) #毙掉是怕意外擦除
'''
#============#-----------tokenize end

df2 = pd.read_csv('tokenized.csv')
df2.dropna(subset=['tokenize'], inplace=True)
#---------risky row
suicidal = ['想死','自殺','跳樓', '離開世界', '死咗', '遺書', 
            '跳落去', '安樂死', '尋死', '去死','介手','界手','界刀','不想活','割脈'
            '跳樓','快D死','快d死','自刎','天台','跌落','企跳','自殘','鎅']
#=['唔係想自殺', '唔係想死', '唔想死','冇勇氣自殺','死極都死唔去']
#"過咗身"
#别人想死（不是他自己），
temp = []
for i in df2['tokenize']:
    line = i.strip().split(' ')
    if len(set(line).intersection(set(suicidal)))>0:
        temp.append(1)
    else:
        temp.append(0)
df2['risky_index'] = temp
df2.sort_values(by = ['conversationId','datetime'], ascending=True, inplace=True, ignore_index=True) #this is important
print (len((set(list(df2['conversationId'])))))

#%%
#-----------position
temp = [0]
c=1
for i, j in zip( df2['conversationId'][1:],  df2['conversationId'][:-1]):
    if i==j:
        temp.append(c)
        c = c+1
    else:
        c = 0
        temp.append(c)
df2['position'] = temp

#-----------position of last sentence
last_position = dict()
last_sentence = df2.drop_duplicates(subset='conversationId', inplace=False, keep='last', ignore_index=False) 
for i in last_sentence.index:
    last_position[last_sentence.loc[i, 'conversationId']] = last_sentence.loc[i, 'position']
temp = []
for i in df2['conversationId']:
    temp.append(last_position[i])
df2['last_position'] = temp

#---最后一句到目标句的gap
gap = np.array(df2['last_position'])-np.array(df2['position'])
df2['gap'] = gap
#%%---risky block
#---first risky sentence position is valid
risky_sentence_only = df2[df2['risky_index']==1] 
first_risky_sentence = risky_sentence_only.drop_duplicates(subset=['conversationId'], inplace=False, keep='first', ignore_index=False) 
valid_risky_index = first_risky_sentence[first_risky_sentence['position']>=16]    #position of the first_risky_sentence

#---last sentence position is valid 
valid_risky_index = valid_risky_index[valid_risky_index['gap']>=5 ]

risky_msg = []
for i in valid_risky_index.index:
    ambient =  df2.loc[i, 'tokenize']+ \
    df2.loc[i+1, 'tokenize']+ df2.loc[i+2, 'tokenize']+ df2.loc[i+3, 'tokenize']+ df2.loc[i+4, 'tokenize']+ df2.loc[i+5, 'tokenize']+ \
    df2.loc[i-1, 'tokenize']+ df2.loc[i-2, 'tokenize']#+ df2.loc[i-3, 'tokenize']+ df2.loc[i-4, 'tokenize']+ df2.loc[i-5, 'tokenize']
    risky_msg.append(ambient)

risky_words_lst = []
for i in risky_msg:
    risky_words_lst.append(i.strip().split(' '))

riskyId=[]    
complete_msg_risky = []
for i in valid_risky_index.index:
    ambient = df2.loc[i-2, 'msg']+ df2.loc[i-1, 'msg']+df2.loc[i, 'msg']+ \
    df2.loc[i+1, 'msg']+ df2.loc[i+2, 'msg']+ df2.loc[i+3, 'msg']+ df2.loc[i+4, 'msg']+ df2.loc[i+5, 'msg']
    complete_msg_risky.append(ambient)
    riskyId.append(df2.loc[i, 'conversationId'])
'''    
riskyOut=pd.DataFrame()
riskyOut['conversationId'], riskyOut['block'] = riskyId, complete_msg_risky
riskyOut.to_csv('confirm_risky_blocks.csv', encoding='utf_8_sig', index=False) 
'''  
#%%
#---moderate block
moderate = df2[~df2['conversationId'].isin(first_risky_sentence['conversationId'])]  
moderate = moderate[(moderate['position']>=5) & (moderate['gap']>=16)]  
moderate = moderate.sample(n=valid_risky_index.shape[0], random_state=40)
    
moderate_msg = []
for i in moderate.index:
    ambient =  df2.loc[i, 'tokenize']+\
    df2.loc[i+1, 'tokenize']+ df2.loc[i+2, 'tokenize']+ df2.loc[i+3, 'tokenize']+ df2.loc[i+4, 'tokenize']+\
    df2.loc[i-1, 'tokenize']+ df2.loc[i-2, 'tokenize']+ df2.loc[i-3, 'tokenize']+ df2.loc[i-4, 'tokenize']
    moderate_msg.append(ambient)

moderate_words_lsttt = []
for i in moderate_msg:
    moderate_words_lsttt.append(i.strip().split(' '))  
    
#%%
#-----------------block before high risk bloc
msg_before = []
for i in valid_risky_index.index:
    lookback = 1
    before = df2.loc[i-lookback-0, 'tokenize'] + df2.loc[i-lookback-1, 'tokenize']  + df2.loc[i-lookback-2, 'tokenize']+\
    df2.loc[i-lookback-3, 'tokenize'] + df2.loc[i-lookback-4, 'tokenize']  + df2.loc[i-lookback-5, 'tokenize']+\
    df2.loc[i-lookback-6, 'tokenize'] + df2.loc[i-lookback-7, 'tokenize']  + df2.loc[i-lookback-8, 'tokenize']
    msg_before.append(before)

before_words_lsttt = []
for i in msg_before:
    before_words_lsttt.append(i.strip().split(' '))
before_words_lst = [i[1] for i in list(enumerate(before_words_lsttt))]
before_index = [i[0] for i in list(enumerate(before_words_lsttt))]

#-----------------complete_msg_before(为了看具体对话)
complete_msg_before = []
for i in valid_risky_index.index:
    complete_before = df2.loc[i-lookback-0, 'msg'] + df2.loc[i-lookback-1, 'msg']  + df2.loc[i-lookback-2, 'msg']+\
    df2.loc[i-lookback-3, 'msg'] + df2.loc[i-lookback-4, 'msg']  + df2.loc[i-lookback-5, 'msg']+\
    df2.loc[i-lookback-6, 'msg'] + df2.loc[i-lookback-7, 'msg']  + df2.loc[i-lookback-8, 'msg']
    complete_msg_before.append(complete_before)

#%%
#--------sampling risky and moderate blocks
random.seed(40)
risky_words_lst = random.sample(risky_words_lst, 1000)
random.seed(40)    
moderate_words_lst = random.sample(moderate_words_lsttt, 1000)  
random.seed(40)    
before_words_lst = random.sample(before_words_lst, 1000)  
#%%
### read network
#networkfile = 'words_simi_network_fewENGwords.csv'
networkfile = 'combinatory_network.csv'
print(networkfile)

df_network =  pd.read_csv('../word2vec/'+networkfile)
df_network = df_network[df_network['weight']>0.0]
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
random_nodes = random.sample(G.nodes(),100)
for w in random_nodes:
    ct = ct+1
    pathlength[w] = nx.single_source_shortest_path_length(G, w)
    if ct%100==0: print (ct)
    
nodes = G.nodes()
save_pkl(pathlength, 'pathlength_dict.pkl')
pathlength = load_pkl( 'pathlength_dict.pkl')
    
#%%

risky_words_lst = [word_polish(i, G.nodes()) for i in risky_words_lst if len(word_polish(i, G.nodes()))>0]
moderate_words_lst = [word_polish(i, G.nodes()) for i in moderate_words_lst if len(word_polish(i, G.nodes()))>0]  
before_words_lst   = [word_polish(i, G.nodes()) for i in before_words_lst   if len(word_polish(i, G.nodes()))>0]

#===--------mean sp
meansp_rk = []
for i in risky_words_lst:
    i = word_polish(i, G.nodes())
    meansp_rk.append(mean_sp(i, G.nodes(), pathlength))

meansp_m = []
for i in moderate_words_lst:
    i = word_polish(i, G.nodes())
    meansp_m.append(mean_sp(i, G.nodes(), pathlength))   

random.seed(40)
meansp_rm = []
for i in risky_words_lst:
    i = word_polish(i, G.nodes())
    meansp_rm.append(mean_sp(random.sample(random_nodes,len(i)), G.nodes(), pathlength))  

print('Experiment 1: Proof of Module')    
print (z_test(meansp_rk, meansp_rm), z_test(meansp_m, meansp_rm)) 

#%%
all_risky_words = list(set(sum(risky_words_lst,[])))
all_risky_words = word_polish(all_risky_words, G.nodes()) 
nbunch = all_risky_words
H = G.subgraph(nbunch)
graphs = sorted(nx.connected_components(H), key=len, reverse=True)
for module in graphs:
    print (len(module))
    
print ('number_connected_components:', nx.number_connected_components(H))
print ('# of nodes: ', nx.number_of_nodes(H), '# of edges: ', nx.number_of_edges(H))
#%%

#--------parameters
print('Experiment 3: Proof of risk estimation')   

m,n,o,p = 0, len(risky_words_lst), 0, len(moderate_words_lst)

u, v = 0, len(before_words_lst)

ZSCORE=False
SSCORE = not ZSCORE
if ZSCORE: print('Z-SCORE:')
else: print ('S-SCORE:')

#--------------Befor-Risky---------------------
lst0 = [] 
dist_fr  = []
for i in before_words_lst[u:v]:
    for j in risky_words_lst[m:n]:
        
        if SSCORE:
            lst0.append( s_score(i, j, nodes, pathlength) )

    dist_fr.append(np.mean(lst0))
    lst0=[]        
print ('Befor-Risky, X-score:', sum(dist_fr) / len(dist_fr), np.std(dist_fr) )
print ('\n')


#----------------Risky-Risky-------------------
random.seed(40)
lst1 = [] 
randomm1, dist_rr  = [], []
#, dist_rm, dist_mr, , [], [], []
for i in risky_words_lst[m:n]:
    #i = word_polish(i, G.nodes())  
    for j in risky_words_lst[m:n]:
        #j = word_polish(j, G.nodes()) 
        
        if SSCORE:
            lst1.append( s_score(i, j, nodes, pathlength) )
            #randomm1.append(s_score(i, random.sample(random_nodes,len(j)), nodes, pathlength)) 
    dist_rr.append(np.mean(lst1))
    lst1=[]        
print ('Risky Chat-Risky Chat, X-score:', sum(dist_rr) / len(dist_rr), np.std(dist_rr) )
print ('\n')

#----------------moderate-Risky-------------------
random.seed(40)
lst3, dist_mr = [], []    
for i in moderate_words_lst[o:p]:
    #i = word_polish(i, G.nodes())  
    for j in risky_words_lst[m:n]:
        #j = word_polish(j, G.nodes()) 
        if SSCORE:
            lst3.append( s_score(i, j, nodes, pathlength) )
        if ZSCORE:
            lst3.append( z_score(i, j) , G)
    dist_mr.append(np.mean(lst3))
    lst3=[]
print ('Moderate Chat - Risky Chat, X-score:',sum(dist_mr) / len(dist_mr), np.std(dist_mr) )

#%%
'''
#--output distance
df_distance=pd.DataFrame()
df_distance['dist_fr'] = dist_fr
df_distance['dist_mr'] = dist_mr
df_distance.to_csv('output_distance.csv', index=False)
'''
#---update good samples
df_rank = pd.DataFrame()
df_rank['msg_index'] = moderate_samples_index
df_rank['score'] = dist_mr
df_rank.sort_values(by = 'score', ascending=True, inplace=True, ignore_index=True)
aa, bb = [], []
for i in df_rank['msg_index'][:50]:
    aa.append(candidate_msg[i])
for i in df_rank['msg_index'][-50:]:
    bb.append(candidate_msg[i])
#%%
#----plot
import matplotlib.pyplot as plt
import seaborn as sns
df_distance = pd.read_csv('D:/research/openai/openup2.0/distance_on_graph/output_distance.csv')
dist_fr = list(df_distance['dist_fr'])
dist_mr = list(df_distance['dist_mr'])
dist_fr.sort()
dist_mr.sort(reverse=True)
sns.distplot(dist_fr[:],hist=True,kde=True,color='red')
sns.distplot(dist_mr[:],hist=True,kde=True,color='royalblue')
   
#%%    
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, roc_auc_score      
def plot_pr(auc_score, precision, recall, label=None, pr=True):  
    pylab.figure(num=None, figsize=(6, 5))  
    pylab.xlim([0.0, 1.0])  
    pylab.ylim([0.0, 1.0])  
     
    if pr:
        pylab.xlabel('Recall') 
        pylab.ylabel('Precision')
        pylab.title('P/R CURVE') 
    else:
        pylab.xlabel('FPR') 
        pylab.ylabel('TPR')
        pylab.title('ROC CURVE') 
     
    pylab.fill_between(recall, precision, alpha=0.5)  
    pylab.grid(True, linestyle='-', color='0.75')  
    pylab.plot(recall, precision, lw=1)      
    pylab.show()

from random import sample
n_cases = 100
y_true = [0]*n_cases+[1]*n_cases
y_scores = sample(dist_fr[:800], n_cases)+sample(dist_mr[:800], n_cases)   #注意：  1 refers to low risk
print(roc_auc_score(y_true, y_scores))
#precision, recall, thresholds1 = precision_recall_curve(y_true, y_scores)
#plot_pr(0.5,  precision,recall, "pos")

fpr, tpr, thresholds2 = roc_curve(y_true, y_scores, pos_label=1)
plot_pr(0.5,  tpr,fpr, "pos",pr=False)


#%%
#illustrate the distance
all_scores = y_scores+dist_fr
all_labels = y_true + [2]*len(dist_fr)

dd = pd.DataFrame()
dd ['score'],dd ['label'] = all_scores, all_labels
dd.sort_values(by='score', inplace=True, ignore_index=True)

temp = dd[dd['label']==1]
low_x = temp.index.to_list()
low_y = temp['score']

temp = dd[dd['label']==0]
high_x = temp.index.to_list()
high_y = temp['score']

temp = dd[dd['label']==2]
prior_x = temp.index.to_list()
prior_y = temp['score']

pylab.bar(high_x, high_y, facecolor = 'r',width = 0.35)
pylab.bar(low_x, low_y, facecolor = 'blue',width = 0.35)
pylab.bar(prior_x, prior_y, facecolor = 'y',width = 0.35)
#%%

print('Experiment 2: Proof of 葡萄干模型')   
#print (z_test(randomm1, lst1)) 
#print (z_test(randomm2, lst2)) 
print (z_test(lst1, lst2)) 
#%% viz

node_id, id_node = dict(), dict()
c = 0
for i in G.nodes():
    c=c+1
    node_id[i] = c
    id_node[c] = i

#----------node file
feature = dict()
for i in G.nodes():
    feature[i] = 0
for i in si:
    feature[i] = 1
for i in sj:
    feature[i] = 2
for i in sk:
    feature[i] = 3
    
f = open('node_feature.csv','w',encoding='utf-8')
f.write('id'+','+'name'+','+'feature'+'\n')
for i in feature:
    f.write(str(node_id[i])+','+i+','+str(feature[i])+'\n')
f.close()    

#----------edge file
#----column 1,2
viz = pd.DataFrame()
viz['Source'] = [node_id[i[0]] for i in G.edges()]
viz['Target'] = [node_id[i[1]] for i in G.edges()]
'''
#找10个risky module: (1)内部连接较多； （2）彼此连接紧密 
viz_risky_modules = []
for i in risky_words_lst[:10]:
    if max_module_size(i, G)>10:
        for j in risky_words_lst[20:30]:
            if s_score(i, j, nodes, pathlength) > 0.1:       #####this setting is important
                if max_module_size(j, G)>10:
                    viz_risky_modules.append(word_polish(i, nodes))
                    viz_risky_modules.append(word_polish(j, nodes))

#找10个moderate module: (1)内部连接较多； （2）彼此连接xishu 
viz_moderate_modules = []
for i in moderate_words_lst[:30]:
    if max_module_size(i, G)>10:
        for j in moderate_words_lst[30:40]:
            if s_score(i, j, nodes, pathlength) > 0.3:      #####this setting is important
                if max_module_size(j, G)>10:
                    viz_moderate_modules.append(word_polish(i, nodes))
                    viz_moderate_modules.append(word_polish(j, nodes))

'''
'''
viz_risky_modules = []
for i in risky_words_lst[:]:
    if max_module_size(i, G)>10:
        print (i)
'''        
def characteristics(mA, mB):
    print ('max_modele_size_A:', max_module_size(mA, G))
    print ('max_modele_size_B:', max_module_size(mB, G))
    print ('distance:', s_score(mA, mB, nodes, pathlength))
    return 0
    
def meet_requirement(ri,rj,mk):
    if ri != rj:
        if ((max_module_size(ri, G)>4) and  (max_module_size(rj, G)>4) and  (max_module_size(mk, G)>4)):
            if s_score(ri, rj, nodes, pathlength)<0.14:
                if s_score(ri, mk, nodes, pathlength) > 0.18:
                    if s_score(rj, mk, nodes, pathlength) > 0.18:
                        return True
            

viz1, viz2, viz3 = [],[],[]

for mk in moderate_words_lst[:10]:
        for ri in risky_words_lst[:10]:
            for rj in before_words_lst[:10]:
                if meet_requirement(ri,rj,mk):
                    viz1.append(ri)
                    viz2.append(rj)
                    viz3.append(mk)
'''                    
viz_risky_modules = []
viz_moderate_modules = []       
for i in moderate_words_lst:
    if max_module_size(i, G)>5:        
        viz_moderate_modules.append(word_polish(i, nodes))
for i in risky_words_lst:    
    if max_module_size(i, G)>5: 
        viz_risky_modules.append(word_polish(i, nodes))

for i in viz_moderate_modules:
    for j in viz_risky_modules:
        characteristics(i, j)


temp = []
for i in viz_moderate_modules:
    for j in viz_risky_modules:
        temp.append(s_score(i, j, nodes, pathlength))
print (np.mean(np.array(temp)))
''' 
#%%
#edge feature
random.seed(40)
weight = dict()

for i, j in zip(viz['Source'], viz['Target']):
    weight[(i, j)] = 1
    weight[(j, i)] = 1

#注意： 如果一条边同时属于两种module, 后面的code会改变前面code的赋值

#--------change index for sampling
risky_nodes = []
idx1, indx2, indx3 = 54,65,482#44, 44, 29
#for concept in random.sample(viz_risky_modules,5): 
for concept in viz1[idx1:idx1+1]:                
    H = G.subgraph(concept)
    for e in H.edges():
        
        risky_nodes.append(node_id[e[0]])
        risky_nodes.append(node_id[e[1]])
        weight[(node_id[e[0]], node_id[e[1]])] = 2
        weight[(node_id[e[1]], node_id[e[0]])] = 2

for concept in viz2[indx2: indx2+1]:                
    H = G.subgraph(concept)
    for e in H.edges():
        if weight[(node_id[e[0]], node_id[e[1]])] == 2:
            weight[(node_id[e[0]], node_id[e[1]])] = 4
            weight[(node_id[e[1]], node_id[e[0]])] = 4
        else:
            weight[(node_id[e[0]], node_id[e[1]])] = 3
            weight[(node_id[e[1]], node_id[e[0]])] = 3
        
for concept in viz3[indx3: indx3+1]:                
    H = G.subgraph(concept)

    for e in H.edges():
        if weight[(node_id[e[0]], node_id[e[1]])] > 1:
            weight[(node_id[e[0]], node_id[e[1]])] = 6
            weight[(node_id[e[1]], node_id[e[0]])] = 6
        else:
            weight[(node_id[e[0]], node_id[e[1]])] = 5
            weight[(node_id[e[1]], node_id[e[0]])] = 5
        
#----column 3
wt=[]
for i, j in zip(viz['Source'], viz['Target']):
    wt.append(weight[(i, j)])
viz['Weight'] = wt

dfbase = viz[viz['Weight']==1]
dfbase = dfbase.sample(n=12000)
dfcolor = viz[viz['Weight']>1]
vizf = dfbase.append(dfcolor,ignore_index=True)

vizf.to_csv('viz2.csv', index= False)

#%%                 
#edge feature dict
random.seed(41)
weight = dict()

for i, j in zip(viz['Source'], viz['Target']):
    weight[(i, j)] = 1
    weight[(j, i)] = 1

#注意： 如果一条边同时属于两种module, 后面的code会改变前面code的赋值
cutting1, cutting2 = [], []    
for i in viz_risky_modules:
    if '流血' in i:
        cutting1.append(i)
    if '紧血' in i:
        cutting2.append(i)
    
cc=2
risky_nodes = []
#for concept in random.sample(viz_risky_modules,5): 
for concept in viz_risky_modules:                
    H = G.subgraph(concept)
    cc=cc+2
    for e in H.edges():
        
        risky_nodes.append(node_id[e[0]])
        risky_nodes.append(node_id[e[1]])
        weight[(node_id[e[0]], node_id[e[1]])] = 2
        weight[(node_id[e[1]], node_id[e[0]])] = 2
        
for concept in viz_moderate_modules[:]:                
    H = G.subgraph(concept)

    for e in H.edges():
        if weight[(node_id[e[0]], node_id[e[1]])] == 2:
            weight[(node_id[e[0]], node_id[e[1]])] = 4
            weight[(node_id[e[1]], node_id[e[0]])] = 4
        else:
            weight[(node_id[e[0]], node_id[e[1]])] = 3
            weight[(node_id[e[1]], node_id[e[0]])] = 3
        
'''
for concept in random.sample(viz_moderate_modules, 10):                 
    H = G.subgraph(concept)
    for e in H.edges():
        weight[(node_id[e[0]], node_id[e[1]])] = 3
        weight[(node_id[e[1]], node_id[e[0]])] = 3 
'''        
        
        
        
        
        
'''         
risky_nodes = set(risky_nodes)
existing_nodes = []
for concept in random.sample(viz_moderate_modules, len(viz_moderate_modules)):                 
    H = G.subgraph(concept)
    
    
    for e in H.edges():
        if ((node_id[e[0]] not in risky_nodes) and (node_id[e[1]] not in risky_nodes) and (node_id[e[0]] not in existing_nodes) and (node_id[e[1]] not in existing_nodes)):
          
                existing_nodes.append(node_id[e[0]])
                existing_nodes.append(node_id[e[1]])
                weight[(node_id[e[0]], node_id[e[1]])] = 3
                weight[(node_id[e[1]], node_id[e[0]])] = 3 
                
        else:
            temp = random.uniform(0,1)
            if temp>0.95:
                
                existing_nodes.append(node_id[e[0]])
                existing_nodes.append(node_id[e[1]])
                weight[(node_id[e[0]], node_id[e[1]])] = 3
                weight[(node_id[e[1]], node_id[e[0]])] = 3 
                
'''           
                
'''        
    for e in H.edges():
        if ((node_id[e[0]] not in risky_nodes) and (node_id[e[1]] not in risky_nodes) ):
          
                existing_nodes.append(node_id[e[0]])
                existing_nodes.append(node_id[e[1]])
                weight[(node_id[e[0]], node_id[e[1]])] = 3
                weight[(node_id[e[1]], node_id[e[0]])] = 3 
                
        else:
            temp = random.uniform(0,1)
            if temp>0.00:
                for s in weight:
                    if ((s[0] == node_id[e[0]]) or (s[0] == node_id[e[1]])):
                        weight[s] = 1
                    if ((s[1] == node_id[e[0]]) or (s[1] == node_id[e[1]])):
                        weight[s] = 1
                existing_nodes.append(node_id[e[0]])
                existing_nodes.append(node_id[e[1]])
                weight[(node_id[e[0]], node_id[e[1]])] = 3
                weight[(node_id[e[1]], node_id[e[0]])] = 3         
                
'''          
                
                
                
                
                
''' 
for concept in random.sample(viz_moderate_modules,1):                 
    H = G.subgraph(concept)
    for e in H.edges():
        weight[(node_id[e[0]], node_id[e[1]])] = 5
        weight[(node_id[e[1]], node_id[e[0]])] = 5
'''        
#----column 3
wt=[]
for i, j in zip(viz['Source'], viz['Target']):
    wt.append(weight[(i, j)])
viz['Weight'] = wt

viz.to_csv('viz.csv', index= False)

viz.head(200)






