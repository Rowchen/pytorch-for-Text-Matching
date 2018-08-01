import pandas as pd
import numpy as np
import re
import Levenshtein
from tqdm import tqdm
from utils import *
import io
import gc
def change_sent(s1,s2):
    s_word1=s1.strip().split()
    c=np.random.randint(0,len(s_word1),1)[0]
    news1=s_word1[c:]
    news1.extend(s_word1[:c])
                        
    s_word2=s2.strip().split()
    c=np.random.randint(0,len(s_word2),1)[0]
    news2=s_word2[c:]
    news2.extend(s_word2[:c])
    
    return [" ".join(news2)," ".join(news1)]

def find_pos_set(data):
    ls=data[data['tag']==1].reset_index(drop=True)
    union_set=[]
    his_dict={}
    ls1=ls['sp1'].values
    ls2=ls['sp2'].values
    print (ls1.shape[0])
    for row in range(ls1.shape[0]):
        s1=ls1[row]
        s2=ls2[row]
        if s1 not in his_dict:
            his_dict[s1]=set([s1])
        if s2 not in his_dict:
            his_dict[s2]=set([s2])
        '''his_dict[s1] is a sentence sets in the training data are similar to s1'''
        his_dict[s1].add(s2)
        his_dict[s2].add(s1)
        
        flags1=flags2=True
        id1=id2=-1
        for i,us in enumerate(union_set):
            if s1 in us:
                id1=i
                flags1=False
            if s2 in us:
                id2=i
                flags2=False
        '''can not find s1 and s2'''
        if flags1&flags2:
            union_set.append(set([s1,s2]))
            
        '''find s1 without s2'''
        if (not flags1)&(flags2):
            union_set[id1].add(s2)
            
        '''find s1 without s2'''
        if (flags1)&(not flags2):
            union_set[id2].add(s1)
            
        '''find s1 and s2 and they are in different set'''
        if (not flags1)&(not flags2):
            if id1!=id2:
                union_set[id1]= (union_set[id1]|union_set[id2])
                del union_set[id2]
    
    np.random.seed(2018)
    new_data=[]
    for us in union_set:
        for s1 in us:
            for s2 in us:
                if s2 not in his_dict[s1]:
                    r=np.random.randint(0,5000)
                    if r<20:
                        new_data.append(change_sent(s1,s2))                     
                        his_dict[s1].add(s2)
                        his_dict[s2].add(s1)
    print ('generate postive newdata ',len(new_data))
    return union_set,new_data

def find_neg_set(data,pos_uni):
    gc.collect()
    pos_dict={}
    for us in pos_uni:
        for s1 in us:
            for s2 in us:
                if s1 not in pos_dict:
                    pos_dict[s1]=list([s2])
                else:
                    pos_dict[s1].append(s2)
                if s2 not in pos_dict:
                    pos_dict[s2]=list([s1])
                else:
                    pos_dict[s2].append(s1)
    print (len(pos_dict))
    
    ls=data[data['tag']==0].reset_index(drop=True)
    ls1=ls['sp1'].values
    ls2=ls['sp2'].values
    new_data=[]
    np.random.seed(2018)
    for row in range(ls1.shape[0]):
        s1=ls1[row]
        s2=ls2[row]
        if s1 in pos_dict and s2 not in pos_dict:
            idx=np.random.randint(0,len(pos_dict[s1]))
            new_data.append(change_sent(pos_dict[s1][idx],s2))
            
        if s2 in pos_dict and s1 not in pos_dict:
            idx=np.random.randint(0,len(pos_dict[s2]))
            new_data.append(change_sent(s1,pos_dict[s2][idx]))
        
        if s1 in pos_dict and s2 in pos_dict :
            if s2 in pos_dict[s1] or s1 in pos_dict[s2]:
                continue
            idx1=np.random.randint(0,len(pos_dict[s1]))
            idx2=np.random.randint(0,len(pos_dict[s2]))
            new_data.append(change_sent(pos_dict[s1][idx1],pos_dict[s2][idx2]))
            
    print ('generate negtive newdata ',len(new_data))
    return new_data
def text_preprocess2(raw_text):
    raw_text=raw_text.strip().lower()
    words=re.sub('!|,|\.|\?|\(|\)|:|\*|/|%|&|#|"|`|;|}', ' ', raw_text)
    words=re.sub("'", ' ', words)
    words=re.sub('(\d+)usd', lambda m: m.group(1) + u' usd ', words)
    words=re.sub('(\d+)cm', lambda m: m.group(1) + u' cm ', words)
    words=re.sub('(\d+)h', lambda m: m.group(1) + u' hora ', words)
    words=re.sub('(\d+)anos', lambda m: m.group(1) + u' anos ', words)
    words=re.sub('(\d+)\$', lambda m: m.group(1) + u' $ ', words)
    words=re.sub(u'(\d+)€', lambda m: m.group(1) + u' euro ', words)
    words=re.sub('\$(\d+)', lambda m: u' $ '+m.group(1), words)
    words=re.sub('(\d+)-(\d+)-(\d+)', 'fecha', words)
    words=re.sub('-', ' ', words)
    words=re.sub('e-mail|@', ' email ', words)
    words=re.sub('2nd', 'segundo', words)
    words=re.sub('mp4', 'music player', words)
    return  words

def generate_vec_list(word_in_doc):
    vec_list=set()
    with io.open('../data/wiki.es.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')as f:
        for line in tqdm(f):
            ls=line.split()
            if ls[0] in word_in_doc:
                if ls[1][-1].isdigit():
                    vec_list.add(ls[0])
                elif ls[2][-1].isdigit():
                    vec_list.add(ls[0]+'_'+ls[1])
    return vec_list


def find_lack_word(trian_test_doc,vec_list):
    lack_set=set()
    count=0
    for ls in trian_test_doc:
        s=[u'/s']+ls.strip().lower().split()+[u'/s']
        for i in range(1,len(s)-1):
            w1=s[i-1]+'_'+s[i]
            w2=s[i]+'_'+s[i+1]
            if w1 not in vec_list and w2 not in vec_list:
                if s[i] not in vec_list:
                    lack_set.add(s[i])
                    count+=1
    print ('lack word number is' ,count)
    return lack_set


def generate_candidata_word(lack_set,word_in_doc):
    all_candi_word=set()
    word_candidate_dict=[]
    word_in_doc_not_lack=word_in_doc-lack_set
    for word in tqdm(lack_set):
        min_dist=1000
        candidata=list()
        cloest_word=list()
        for w in word_in_doc_not_lack:
            d=Levenshtein.distance(word, w)
            if d<min_dist:
                min_dist=d
                cloest_word=[w]
            if d==min_dist:
                cloest_word.append(w)
            if d<=2:
                candidata.append(w)
        candidata=set(candidata)|set(cloest_word)
        word_candidate_dict.append(' '.join(list(candidata)))
        all_candi_word|=candidata
    wc=pd.DataFrame({'word':list(lack_set),'candidata_word':word_candidate_dict})
    return wc


def replace_word(data,all_bigram_cnt,all_unigram_cnt,lack_set,wc):
    '''
    input:data:  tarin\test
    all_bigram_cnt\all_unigram_cnt:a dict for all bigram/unigram in all doc ,including unlabel data
    lack_set:    a set including train and test word which didn't appear in wordvec
    wc: a df, maintain the candidata correct word for each wrong word
    '''
    for col in ['sp1','sp2']:
        new_words=[]
        ls=data[col].values
        for row in range(data.shape[0]):
            s=[u'/s']+ls[row].strip().split()+[u'/s']
            for i in range(1,len(s)-1):
                if s[i] in lack_set:
                    #detect digit or id
                    nc=0
                    for ch in s[i]:
                        if ch.isdigit():
                            nc+=1
                    if nc>=8:
                        s[i]=u'id'
                    elif 1.0*nc/len(s[i])>0.8:
                        s[i]=u'valor'
                    else:
                        cand_words=wc.loc[wc['word']==s[i],'candidata_word'].values[0].split()
                        if len(cand_words)==1:
                            s[i]=cand_words[0]
                        else:
                            maxlike_word=None
                            max_proba=0
                            for cw in cand_words:
                                bigram_word1=s[i-1]+'_'+cw
                                bigram_word2=cw+'_'+s[i+1]
                                proba=(all_bigram_cnt.get(bigram_word1,0)+1)*(all_bigram_cnt.get(bigram_word2,0)+1)
                                if proba>max_proba:
                                    max_proba=proba
                                    maxlike_word=cw
                                if proba==max_proba:
                                    if all_unigram_cnt[cw]>all_unigram_cnt[maxlike_word]:
                                        maxlike_word=cw


                            s[i]=maxlike_word
            new_words.append(' '.join(s[1:-1]))
        data[col]=np.array(new_words)
    return data


def generate_bigram(data):
    all_bigram_cnt={}
    for s in tqdm(data):
        words=[u'/s']+s.strip().split()+[u'/s']
        for i in range(len(words)-1):
            w=words[i]+'_'+words[i+1]
            all_bigram_cnt[w]=all_bigram_cnt.get(w,0)+1
    return all_bigram_cnt


def generate_w2v(word_in_doc):
    word_vec_dict={}
    with io.open('../data/wiki.es.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')as f:
        for line in tqdm(f):
            ls=line.split()
            if ls[0] in word_in_doc:
                if ls[1][-1].isdigit():
                    word_vec_dict[ls[0]]=np.array(ls[1:]).astype(float)
                elif ls[2][-1].isdigit():
                    word_vec_dict[ls[0]+'_'+ls[1]]=np.array(ls[2:]).astype(float)
    return word_vec_dict


def word_embedding(data,word_vec_dict):
    for col in ['sp1','sp2']:
        word_embeding=[]
        ls1=ls=data[col].values
        for row in tqdm(range(data.shape[0])):
            embeding_result=list()
            s=[u'/s']+ls1[row].strip().lower().split()+[u'/s']
            for i in range(1,len(s)-1):
                w1=s[i-1]+'_'+s[i]
                w2=s[i]+'_'+s[i+1]
                if w1 not in word_vec_dict and w2 not in word_vec_dict:
                    embeding_result.append(word_vec_dict[s[i]])
                if w2 in word_vec_dict:
                    embeding_result.append(word_vec_dict[w2])
            word_embeding.append(embeding_result)    
        data['word_embed_'+col]=np.array(word_embeding)
    return data

def word_embedding2(data,vocab,max_length,stop=False,sp_stops=None):
    count=0
    word_embeding=np.zeros((data.shape[0],2,max_length),dtype=int)
    for n,col in enumerate(['sp1','sp2']):
        ls1=data[col].values
        for row in tqdm(range(data.shape[0])):
            embeding_result=list()
            s=[u'/s']+ls1[row].strip().lower().split()+[u'/s']
            for i in range(1,len(s)-1):
                w1=s[i-1]+'_'+s[i]
                w2=s[i]+'_'+s[i+1]
                if w1 not in vocab and w2 not in vocab:
                    if s[i] in vocab:
                        embeding_result.append(vocab.index(s[i]))
                    else:
                        print ('error,encount word that not in vocab')
                        #embeding_result.append(len(vocab)+1+hash(s[i])%100)
                        count+=1
                if w2 in vocab:
                    embeding_result.append(vocab.index(w2))
        #padding zeros
            if len(embeding_result)<max_length:
                 for i in range(max_length-len(embeding_result)):
                    embeding_result.append(len(vocab))
                    
            word_embeding[row,n,:]= embeding_result[:max_length]
    print ('random embed word is',count)
    return word_embeding


from sklearn.metrics.pairwise import cosine_similarity
def cos_sim(vec1,vec2):
    return cosine_similarity(np.array(vec1).reshape(1,-1),np.array(vec2).reshape(1,-1))[0,0]

def sta_cos_similar(data):
    inner=['max','min','mean','median']
    outer=['max','min','mean','median','std']
    ls1=data['word_embed_sp1'].values
    ls2=data['word_embed_sp2'].values
    fs=[]
    for row in tqdm(range(ls1.shape[0])):
        similar=[] 
        if len(ls1[row])==0 or len(ls2[row])==0:
            fs.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            continue
        for w1 in ls1[row]:
            cos_similar=[]
            for w2 in ls2[row]:
                cos_similar.append(cos_sim(w1,w2))
            similar.append([Mathutil.agg(op1,cos_similar) for op1 in inner])
        similar=np.array(similar)
        tmp=list()
        for col in range(similar.shape[1]):
            for op2 in outer:
                tmp.append(Mathutil.agg(op2,similar[:,col]))
        fs.append(tmp)
    fs=np.array(fs)
    for i in range(20):
        data['wv_cos_similar_%d'%i]=fs[:,i]
    return data
