import numpy as np
import math
from utils import *
from tqdm import tqdm

def shared_word_rate(data):
    fs=[]
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    for row in range(data.shape[0]):
        s1words = {}
        s2words = {}
        for w1 in ls1[row].strip().split():
            s1words[w1]=s1words.get(w1, 0)+1
        for w2 in ls2[row].strip().split():
            s2words[w2]=s2words.get(w2, 0)+1
        total_word=sum(list(s1words.values())+list(s2words.values()))
        shared_word=sum([s1words[w] for w in s1words if w in s2words])
        fs.append(1.0*shared_word/total_word)
    data['share_word_rate']=np.array(fs)
    LogUtil.log('share word complete')
    return data

def cal_word_cnt(data):
    s_set=set()
    idf = {}
    for row in range(len(data)):
        s1=data[row]
        if s1 not in s_set:
            s_set.add(s1)
            words=s1.strip().split()
            for word in words:
                idf[word] = idf.get(word, 0) + 1
    return idf  
def cal_word_idf(data):
    s_set=set()
    idf = {}
    for row in range(len(data)):
        s1=data[row]
        if s1 not in s_set:
            s_set.add(s1)
            words=s1.strip().split()
            for word in words:
                idf[word] = idf.get(word, 0) + 1
    num_docs=len(s_set)
    for word in idf:
        idf[word]= math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
    return idf 

def shared_word_idf(data,idf):
    fs=[]
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    for row in range(data.shape[0]):
        s1words = {}
        s2words = {}
        for w1 in ls1[row].strip().split():
            s1words[w1]=s1words.get(w1, 0)+1
        for w2 in ls2[row].strip().split():
            s2words[w2]=s2words.get(w2, 0)+1 
        shared_word=sum([s1words[w]*idf[w] for w in s1words if w in s2words])
        total_word=sum([s1words[w]*idf[w] for w in s1words])+sum([s2words[w]*idf[w] for w in s2words])
        
        fs.append(1.0*shared_word/total_word)
    data['share_word_idf']=np.array(fs)
    LogUtil.log('share word rate complete')
    return data

def length_feat(data):
    fs=[]
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    for row in range(data.shape[0]):
        s1=ls1[row].strip()
        s2=ls2[row].strip()
        l1=len(s1)
        l2=len(s2)
        l3=len(s1.split())
        l4=len(s2.split())
        diff1=abs(l1-l2)
        diff2=abs(l3-l4)
        diff3=1.0*min(l1,l2)/max(l1,l2)
        diff4=1.0*min(l3,l4)/max(l3,l4)
        fs.append([l1,l2,l3,l4,diff1,diff2,diff3,diff4])
    fs=np.array(fs)
    for i in range(8):
        data['l%d'%i]=fs[:,i]
    LogUtil.log('length feature complete')
    return data

def power_word(data):
    '''
    when we make this feat,we need to drop the validation set to generate powerful word to avoid overfitting
    '''
    np.random.seed(2018)
    r1=(np.random.uniform(0,1,data.shape[0]))
    
    
    words_power = {}
    ls1=data['sp1'].values[r1<0.5]
    ls2=data['sp2'].values[r1<0.5]
    tag=data['tag'].values[r1<0.5]
    for row in range(ls1.shape[0]):
        s1=ls1[row].strip().split()
        s2=ls2[row].strip().split()
        all_words = set(s1 + s1)
        s1 = set(s1)
        s2 = set(s2)
        share=s1&s2
        for w in all_words:
            if w not in words_power:
                words_power[w]=[0.0 for i in range(7)]
            words_power[w][0]+=1#number of word 
            words_power[w][1]+=1#rate of word
            if w not in share:
                words_power[w][3]+=1#single side word 
                if tag[row]==1:
                    words_power[w][4]+=1#single side and match word
                    words_power[w][2]+=1#match word
            else:
                words_power[w][5]+=1#double side word
                if tag[row]==1:
                    words_power[w][6]+=1#double side and match word
                    words_power[w][2]+=1#match word
                    
    for w in words_power:
        words_power[w][1] /= ls1.shape[0]
        words_power[w][2] /= words_power[w][0]
        if words_power[w][3]>1e-6:
            words_power[w][4] /= words_power[w][3]
        words_power[w][3] /= words_power[w][0]
        if words_power[w][5]>1e-6:
            words_power[w][6] /= words_power[w][5]
        words_power[w][5] /= words_power[w][0]
    #sorted_words_power = sorted(words_power.iteritems(), key=lambda d: d[1][0], reverse=True)
    LogUtil.log('powerful word calculate')
    return words_power

def doubel_side_rate(data,words_power,num_least=100):
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    fs=[]
    for row in range(ls1.shape[0]):
        rate=1
        s1=set(ls1[row].strip().split())
        s2=set(ls2[row].strip().split())
        share_word=s1&s2
        for w in share_word:
            if w not in words_power:
                continue
            if words_power[w][0]*words_power[w][5]>num_least:
                rate*=(1-words_power[w][6])
        rate=1-rate
        fs.append(rate)
    data['doubel_side_rate']=np.array(fs)
    LogUtil.log('doubel_side_rate feature complete')
    return data


def one_side_rate(data,words_power,num_least=100):
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    fs=[]
    for row in range(ls1.shape[0]):
        rate=1
        s1=set(ls1[row].strip().split())
        s2=set(ls2[row].strip().split())
        share_word=s1&s2
        all_word=s1|s2
        diff_word=all_word-share_word
        for w in diff_word:
            if w not in words_power:
                continue
            if words_power[w][0]*words_power[w][3]>num_least:
                rate*=(1-words_power[w][4])
        rate=1-rate
        fs.append(rate)
    data['one_side_rate']=np.array(fs)
    LogUtil.log('one_side_rate feature complete')
    return data

def tfidf_feat(data,allword):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfv = TfidfVectorizer(min_df=3,  max_features=10000,
                          strip_accents='unicode', 
                          analyzer='word'
                          ,norm=None,
            ngram_range=(1,1), use_idf=1,smooth_idf=1,sublinear_tf=1)
    train_tfidf = tfv.fit(list(allword))
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    fs=[]
    for row in range(ls1.shape[0]):
        s1=ls1[row].strip()
        s2=ls2[row].strip()
        trans1=tfv.transform([s1]).data
        trans2=tfv.transform([s2]).data
        if len(trans1)==0:
            tfidf1=0
            tfidf3=0
            tfidf5=0
        else:
            tfidf1=np.sum(trans1)
            tfidf3=np.mean(trans1)
            tfidf5=len(trans1)
        if len(trans2)==0:
            tfidf2=0
            tfidf4=0
            tfidf6=0
        else:
            tfidf2=np.sum(trans2)
            tfidf4=np.mean(trans2)
            tfidf6=len(trans2)
        fs.append([tfidf1,tfidf2,tfidf3,tfidf4,tfidf5,tfidf6])
    fs=np.array(fs)
    for i in range(6):
        data['tfidf%d'%i]=fs[:,i]
    LogUtil.log('tfidf feature complete')
    return data

def doubel_side_word(data,words_power,thresh_num=50,thresh_rate=0.75):
    pword_double=[]
    for w in words_power:
        if (words_power[w][0]*words_power[w][5]>=thresh_num) and (words_power[w][6]>=thresh_rate):
            pword_double.append(w)
    
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    fs=np.zeros((ls1.shape[0],len(pword_double)))
    
    
    for row in range(ls1.shape[0]):
        s1=set(ls1[row].strip().split())
        s2=set(ls2[row].strip().split())
        share_word=s1&s2
        for i,w in enumerate(pword_double):
            if w in share_word:
                fs[row,i]=1
    
    for i in range(len(pword_double)):
        data['double_side_word%d'%i]=fs[:,i]
    LogUtil.log('doubel_side_word feature complete,pword len is %d'%len(pword_double))
    return data

def generate_dul_num(train,test):
    words_dul_num={}
    ls1=train['sp1'].values
    ls2=train['sp2'].values
    ls3=test['sp1'].values
    ls4=test['sp2'].values
    for row in range(ls1.shape[0]):
        s1=ls1[row].strip()
        s2=ls2[row].strip()
        words_dul_num[s1]=words_dul_num.get(s1,0)+1
        words_dul_num[s2]=words_dul_num.get(s2,0)+1
    for row in range(ls3.shape[0]):
        s3=ls3[row].strip()
        s4=ls4[row].strip()
        words_dul_num[s3]=words_dul_num.get(s3,0)+1
        words_dul_num[s4]=words_dul_num.get(s4,0)+1
    return words_dul_num

def dul_num_feat(data,words_dul_num):
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    fs=[]
    for row in range(ls1.shape[0]):
        s1=ls1[row].strip()
        s2=ls2[row].strip()
        d1=words_dul_num[s1]
        d2=words_dul_num[s2]
        fs.append([d1,d2,max(d1,d2),min(d1,d2)])
    fs=np.array(fs)
    for i in range(4):
        data['dul_feat%d'%i]=fs[:,i]
    LogUtil.log('dul feature complete')
    return data

def ngramjaccard(data):
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    fs=[]
    jaccard_coef=[0 for i in range(4)]
    for row in range(ls1.shape[0]):
        s1=ls1[row].strip().split()
        s2=ls2[row].strip().split()
        for i in range(1,5):
            s1_ngram=NgramUtil.ngram(s1,i)
            s2_ngram=NgramUtil.ngram(s2,i)
            jaccard_coef[i-1]=jaccard(s1_ngram,s2_ngram)
        fs.append(jaccard_coef[:])
    fs=np.array(fs)
    for i in range(4):
        data['ngram_jaccard%d'%i]=fs[:,i]
    LogUtil.log('ngram_jaccard feature complete')
    return data

import Levenshtein
def str_edit_distance(str1,str2):
    d = Levenshtein.distance(str1, str2)/ float(max(len(str1), len(str2)))
    return d
def edit_distance(data):
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    fs=[]
    for row in range(ls2.shape[0]):
        s1=ls1[row].strip()
        s2=ls2[row].strip()
        try:
            fs.append(str_edit_distance(s1,s2))
        except:
            fs.append(1)
    fs=np.array(fs)
    data['str_edit_distance']=fs
    LogUtil.log('edit_distance feature complete')
    return data


def n_gram_dist_feat(data):
    inner=['max','min','mean','median']
    outer=['max','min','mean','median','std']
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    fs=[]
    jaccard_coef=[0 for i in range(4)]
    for row in tqdm(range(ls1.shape[0])):
        s1=ls1[row].strip().split()
        s2=ls2[row].strip().split()
        feat=list()
        for i in range(1,2):
            s1_ngram=NgramUtil.ngram(s1,i)
            s2_ngram=NgramUtil.ngram(s2,i)
            dist_list1=list()
            for w1 in s1_ngram:
                dist_list2=list()
                for w2 in s2_ngram:
                    if len(w1)==0 or len(w2)==0:
                        dist_list2.append(1)
                        continue
                    dist_list2.append(str_edit_distance(w1,w2))
                dist_list1.append(dist_list2)
            
            for op1 in inner:
                tmp=list()
                for l in dist_list1:
                    tmp.append(Mathutil.agg(op1,l))
                for op2 in outer:
                    feat.append(Mathutil.agg(op2,tmp))
        fs.append(feat)
    fs=np.array(fs)
    for i in range(20):
        data['n_gram_editdist%d'%i]=fs[:,i]
    LogUtil.log('n_gram_editdist feature complete')
    return data

def word_edit_dist_feat(data):
    ls1=data['sp1'].values
    ls2=data['sp2'].values
    fs=[]
    for row in range(ls2.shape[0]):
        s1=ls1[row].strip().split()
        s2=ls2[row].strip().split()
        s1=sorted(s1)
        s2=sorted(s2)
        new_vec1=' '.join(s1)
        new_vec2=' '.join(s2)
        try:
            fs.append(str_edit_distance(new_vec1,new_vec2))
        except:
            fs.append(1)
    fs=np.array(fs)
    data['word_edit_dist_feat']=fs
    LogUtil.log('word_edit_dist_feat feature complete')
    return data

def word_avg_cos_sim(data,word_vec_dict,n_gram_idf):
    tx1=data['sp1'].values
    tx2=data['sp2'].values
    fs1=[]
    fs2=[]
    for row in tqdm(range(tx2.shape[0])):
        avg_vec1=np.zeros(300)
        avg_vec2=np.zeros(300)
        avg_vec3=np.zeros(300)
        avg_vec4=np.zeros(300)
        s=[u'/s']+tx1[row].strip().lower().split()+[u'/s']
        for i in range(1,len(s)-1):
            w1=s[i-1]+'_'+s[i]
            w2=s[i]+'_'+s[i+1]
            if w1 not in word_vec_dict and w2 not in word_vec_dict:
                avg_vec1+=word_vec_dict[s[i]]
                avg_vec3+=word_vec_dict[s[i]]*n_gram_idf[s[i]]
            if w2 in word_vec_dict:
                avg_vec1+=word_vec_dict[w2]
                avg_vec3+=word_vec_dict[w2]*n_gram_idf[w2]

        s=[u'/s']+tx2[row].strip().lower().split()+[u'/s']
        for i in range(1,len(s)-1):
            w1=s[i-1]+'_'+s[i]
            w2=s[i]+'_'+s[i+1]
            if w1 not in word_vec_dict and w2 not in word_vec_dict:
                avg_vec2+=word_vec_dict[s[i]]
                avg_vec4+=word_vec_dict[s[i]]*n_gram_idf[s[i]]
            if w2 in word_vec_dict:
                avg_vec2+=word_vec_dict[w2]
                avg_vec4+=word_vec_dict[w2]*n_gram_idf[w2]
        
        fs1.append(cos_sim(avg_vec1,avg_vec2))
        fs2.append(cos_sim(avg_vec3,avg_vec4))
        
    data[u'ave_cos_sim']=np.array(fs1)
    data[u'ave_idf_cos_sim']=np.array(fs2)
    return data




