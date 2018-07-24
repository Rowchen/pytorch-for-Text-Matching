# -*- coding: utf-8 -*
import time
import sys
import pandas as pd
import numpy as np
import re
from sklearn.metrics import log_loss
import sys
import io


class TimeUtil(object):
    def __init__(self):
        return
    
    @staticmethod
    def t_now():
        """
        Get the current time, e.g. `2016-12-27 17:14:01`
        :return: string represented current time
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    
class LogUtil(object):
    def __init__(self):
        pass

    @staticmethod
    def log(msg):
        """
        Print message of log
        :param typ: type of log
        :return: none
        """
        print ("[%s]\t%s" % (TimeUtil.t_now(), str(msg)))
        sys.stdout.flush()
        return
    
class Mathutil(object):
    def __init__():
        pass
    
    @staticmethod
    def agg(op,data):
        if np.array(data).shape[0]==0:
            return 0
        if op=='max':
            return np.max(data)
        if op=='min':
            return np.min(data)
        if op=='mean':
            return np.mean(data)
        if op=='median':
            return np.median(data)
        if op=='std':
            return np.std(data)

class NgramUtil(object):
    def __init__():
        pass
    '''
    input need to be a word list
    '''
    @staticmethod
    def unigram(words):
        return words
    
    @staticmethod
    def bigram(words):
        L=len(words)
        bi_word=[]
        for l in range(L-1):
            bi_word.append(str(words[l]+'_'+words[l+1]))
        return bi_word
        
    @staticmethod
    def trigram(words):
        L=len(words)
        tri_word=[]
        for l in range(L-2):
            tri_word.append(str(words[l]+'_'+words[l+1]+'_'+words[l+2]))
        return tri_word
    @staticmethod
    def fourgram(words):
        L=len(words)
        four_word=[]
        for l in range(L-3):
            four_word.append(str(words[l]+'_'+words[l+1]+'_'+words[l+2]+'_'+words[l+3]))
        return four_word
    
    @staticmethod
    def ngram(words,n):
        if n==1:return NgramUtil.unigram(words)
        if n==2:return NgramUtil.bigram(words)
        if n==3:return NgramUtil.trigram(words)
        if n==4:return NgramUtil.fourgram(words)
    



def cv_nfold(model,train,label,test=None,n_fold=5,early_stop=False,seed=2018):
    np.random.seed(seed)
    r1=(np.random.uniform(0,1,train.shape[0])*n_fold).astype(int)
    loss=np.zeros(n_fold)
    if test is not None:
        test_pred=np.zeros((test.shape[0],n_fold))
    for v in range(n_fold):
        filter_t=(r1!=v)
        filter_v=(r1==v)
        x_train , y_train = train[filter_t,:] , label[filter_t]
        x_val  ,  y_val  = train[filter_v,:] , label[filter_v]
        if early_stop:
            model.fit(x_train,y_train,eval_set=[(x_val,y_val)],
                    eval_metric='binary_logloss',
                    early_stopping_rounds=30,verbose=0)
        else:
            model.fit(x_train,y_train)
        val_pred=model.predict_proba(x_val)[:,1]
        loss[v]=log_loss(y_val,val_pred)
        if test is not None:
            test_pred[:,v]=model.predict_proba(test)[:,1]
        train_pred=model.predict_proba(x_train)[:,1]
        print ('train_loss',log_loss(y_train,train_pred),'val_loss',loss[v])
    print ('cv_score',np.mean(loss))
    if test is not None:
        return np.mean(test_pred,axis=1)
    
def jaccard(A,B):
    A=set(A)
    B=set(B)
    if len(A)==0 or len(B)==0:
        return 0
    else:
        return 1.0*(len(A&B))/(len(A|B))
    
    
from sklearn.metrics.pairwise import cosine_similarity
def cos_sim(vec1,vec2):
    return cosine_similarity(np.array(vec1).reshape(1,-1),np.array(vec2).reshape(1,-1))[0,0]

