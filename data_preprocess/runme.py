import pandas as pd
import numpy as np
from tqdm import tqdm
from preprocess import *
from feature import *
from utils import *
import gc

train_dir1='../data/cikm_english_train_20180516.txt'
train_dir2='../data/cikm_spanish_train_20180516.txt'
test_dir='../data/cikm_test_b_20180730.txt'

unlabel_dir='../data/cikm_unlabel_spanish_train_20180516.txt'
doc_max_length=18

train1=(pd.read_csv(train_dir1,sep='\t',header=None,encoding='utf-8')).rename(columns={0:'en1',1:'sp1',2:'en2',3:'sp2',4:'tag'})
train2=(pd.read_csv(train_dir2,sep='\t',header=None,encoding='utf-8')).rename(columns={0:'sp1',1:'en1',2:'sp2',3:'en2',4:'tag'})
test = (pd.read_csv(test_dir , sep='\t',header=None,encoding='utf-8')).rename(columns={0:'sp1',1:'sp2'})
unlabel_data=(pd.read_csv(unlabel_dir,sep='\t',header=None,encoding='utf-8')).rename(columns={0:'sp',1:'en'})
train=pd.concat([train1[['sp1','sp2','tag']],train2[['sp1','sp2','tag']]]).reset_index(drop=True)
print ('train1',train1.shape)
print ('train2',train2.shape)
print ('train',train.shape)
print ('test',test.shape)
print ('Load data complete')

'''
preprocess
word correction
'''
train['sp1']=train['sp1'].apply(text_preprocess2)
train['sp2']=train['sp2'].apply(text_preprocess2)
test['sp1']=test['sp1'].apply(text_preprocess2)
test['sp2']=test['sp2'].apply(text_preprocess2)
unlabel_data['sp']=unlabel_data['sp'].apply(text_preprocess2)

#find word spell wrongly
alldoc=list(set(unlabel_data['sp'].values)|set(train['sp1'].values)|set(train['sp2'].values)|set(test['sp1'].values)|set(test['sp2'].values))
word_in_doc=set()
for s in tqdm(alldoc):
    word_in_doc|=set(s.strip().lower().split())
vec_list=generate_vec_list(word_in_doc)
trian_test_doc=list(set(train['sp1'].values)|set(train['sp2'].values)|set(test['sp1'].values)|set(test['sp2'].values))
lack_set=find_lack_word(trian_test_doc,vec_list)
print (len(lack_set))
llls=pd.DataFrame({'lackword':list(lack_set)})

llls.to_csv('lackword.txt',index=False)
# alldoc=list(set(unlabel_data['sp'].values)|set(train['sp1'].values)|set(train['sp2'].values))
# word_in_doc=set()
# for s in tqdm(alldoc):
#     word_in_doc|=set(s.strip().lower().split())

#word spell correct
wc=generate_candidata_word(lack_set,word_in_doc)
all_unigram_cnt=cal_word_cnt(alldoc)
all_bigram_cnt=generate_bigram(alldoc)
train=replace_word(train,all_bigram_cnt,all_unigram_cnt,lack_set,wc)                    
test=replace_word(test,all_bigram_cnt,all_unigram_cnt,lack_set,wc)  

#check again, except 0
trian_test_doc=list(set(train['sp1'].values)|set(train['sp2'].values)|set(test['sp1'].values)|set(test['sp2'].values))
new_lack_set=find_lack_word(trian_test_doc,vec_list)
print (len(new_lack_set))
print ('text preprocess complete')


'''
shuffe the sentence
'''
# def shuffer_sentence(data):
#     sent_set=set()
#     np.random.seed(10)
#     for col in ['sp1','sp2']:
#         ls=data[col].values
#         for row in range(ls.shape[0]):
#             s=ls[row]
#             if s in sent_set:
#                 s=s.strip().split()
#                 c=np.random.randint(0,len(s),1)[0]
#                 news=s[c:]
#                 news.extend(s[:c])
#                 ls[row]=" ".join(news)
#             else:
#                 sent_set.add(s)
#     return data
'''
generate word 2 vec
'''
word_in_doc=set()
for s in tqdm(trian_test_doc):
    word_in_doc|=set(s.strip().lower().split())
word_vec_dict=generate_w2v(word_in_doc)
pretrain_word_vec=[]
vocab=[]


for k,v in word_vec_dict.items():
    vocab.append(k)
    pretrain_word_vec.append(v)
print ('vocab size is ',len(vocab))
#zero padding data
pretrain_word_vec.append([0.0]*300)
# #unseen data
# np.random.seed(2018)
# for i in range(100):
#     pretrain_word_vec.append((np.random.rand(300)-0.5)/2)
pretrain_word_vec=np.array(pretrain_word_vec)
print (pretrain_word_vec.shape)

'''generate more training data'''
#pos_uni,new_pos=find_pos_set(train)
#new_neg=find_neg_set(train,pos_uni)
#sp1=[]
#sp2=[]
#for comb in new_pos:
#    sp1.append(comb[0])
#    sp2.append(comb[1])
#train_pos=pd.DataFrame({'sp1':sp1,'sp2':sp2,'tag':[1]*len(sp1)})
#sp1=[]
#sp2=[]
#for comb in new_neg:
#    sp1.append(comb[0])
#    sp2.append(comb[1])
#train_neg=pd.DataFrame({'sp1':sp1,'sp2':sp2,'tag':[0]*len(sp1)})

#aaa=set(train['sp1'])|set(train['sp2'])
#print ('now train data has ',len(aaa),'different sentences')
#train=pd.concat([train,train_pos,train_neg]).reset_index(drop=True)
#print ('traning data shape is ',train.shape,'postive sample is ',train['tag'].sum(),'postive rate is ',train['tag'].sum()/train.shape[0])
#aaa=set(train['sp1'])|set(train['sp2'])
#print ('now train data has ',len(aaa),'different sentences')


'''word embedding'''
train_embed=word_embedding2(train,vocab,max_length=doc_max_length)
test_embed=word_embedding2(test,vocab,max_length=doc_max_length)
labels=train['tag'].values
np.save('../data/train_labels4.npy',labels)
np.save('../data/train_embed4.npy',train_embed)
np.save('../data/test_embed4.npy',test_embed)
np.save('../data/wordvec4.npy',pretrain_word_vec)

print ('save embed in doc already')

#remove the stop words
sp_stops=set(pd.read_csv('../data/spanish_stop.txt',header=None)[0].values)
def text_preprocess(raw_text):
    word_list=[]
    for word in raw_text.strip().lower().split():
        if word not in sp_stops:
            word_list+=[word]
    words=' '.join(word_list) 
    return  words

#word embedding
train['sp1']=train['sp1'].apply(text_preprocess)
train['sp2']=train['sp2'].apply(text_preprocess)
test['sp1']=test['sp1'].apply(text_preprocess)
test['sp2']=test['sp2'].apply(text_preprocess)
word_vec_dict=generate_w2v(word_in_doc)
train=word_embedding(train,word_vec_dict)
test=word_embedding(test,word_vec_dict)

train=sta_cos_similar(train)
test=sta_cos_similar(test)

alldoc=list(set(unlabel_data['sp'].values)|set(train['sp1'].values)|set(train['sp2'].values)|set(test['sp1'].values)|set(test['sp2'].values))
idf=cal_word_idf(alldoc)
train=shared_word_idf(train,idf)
test=shared_word_idf(test,idf)

train=shared_word_rate(train)
test=shared_word_rate(test)

train=length_feat(train)
test=length_feat(test)

words_power=power_word(train)

train=doubel_side_rate(train,words_power)
test=doubel_side_rate(test,words_power)

train=one_side_rate(train,words_power)
test=one_side_rate(test,words_power)


train=tfidf_feat(train,alldoc)
test=tfidf_feat(test,alldoc)


train=doubel_side_word(train,words_power)
test=doubel_side_word(test,words_power)

words_dul_num=generate_dul_num(train[['sp1','sp2']],test[['sp1','sp2']])
train=dul_num_feat(train,words_dul_num)
test=dul_num_feat(test,words_dul_num)

train=ngramjaccard(train)
test=ngramjaccard(test)

train=edit_distance(train)
test=edit_distance(test)

train=n_gram_dist_feat(train)
test=n_gram_dist_feat(test)

train=word_edit_dist_feat(train)
test=word_edit_dist_feat(test)

n_gram_idf={}
for k,v in all_bigram_cnt.items():
    n_gram_idf[k]=math.log(len(alldoc) / (v + 1.)) / math.log(2.)
for k,v in all_unigram_cnt.items():
    n_gram_idf[k]=math.log(len(alldoc) / (v + 1.)) / math.log(2.)

train=word_avg_cos_sim(train,word_vec_dict,n_gram_idf)
test=word_avg_cos_sim(test,word_vec_dict,n_gram_idf)


from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(penalty='l2', tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=2018)

features=['share_word_rate','share_word_idf','doubel_side_rate','one_side_rate']+['l%d'%i for i in range(8)]\
            +['tfidf%d'%i for i in range(6)]+['double_side_word%d'%i for i in range(5)]+['dul_feat%d'%i for i in range(4)]\
            +['ngram_jaccard%d'%i for i in range(4)]+['str_edit_distance','word_edit_dist_feat']+\
        ['n_gram_editdist%d'%i for i in range(20)]+['wv_cos_similar_%d'%i for i in range(20)]+\
        ['ave_cos_sim','ave_idf_cos_sim']
target='tag'
len(features)

x_train=train[features].values
y_train=train[target].values
x_test=test[features].values
np.save('../data/train_tradition_feat.npy',x_train)
np.save('../data/test_tradition_feat.npy',x_test)

test_pred=cv_nfold(model_lr,train=x_train,label=y_train,test=x_test,n_fold=5,seed=2018)
