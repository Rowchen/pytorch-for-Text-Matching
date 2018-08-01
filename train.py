import torch as tr
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as opt
import pandas as pd
import numpy as np
import torchvision
from torch.autograd import Variable
import gc
import torch.nn.utils as utils
from sklearn.metrics import log_loss
import time
import sys

from config import *
from model.model_train import *
from model.TextCNN import TextCNN
from model.TextRNN import TextRNN
from model.Inception import *
from model.interactive_attention import DA
from model.self_attention import AttendRNN
from model.ABCNN import ABCNN

train_embed=np.load('data/train_embed4.npy')
test_embed=np.load('data/test_embed4.npy')
labels=np.load('data/train_labels4.npy')
val_fold=5
stacking=True
device = tr.device('cuda')

use_tradition_feat=False
if use_tradition_feat:
    print ('using traiditional feature')
    train_sta=np.load('data/train_tradition_feat2.npy')
    test_sta=np.load('data/test_tradition_feat2.npy')
    sta_feat=np.concatenate((train_sta,test_sta),axis=0)
    print (sta_feat.shape)
    
    train_idx=np.zeros((train_embed.shape[0],2,1),dtype=np.int)
    for i in range(train_embed.shape[0]):
        train_idx[i]=[[i],[i]]
    train_embed=np.concatenate((train_embed,train_idx),axis=2)
    test_idx=np.zeros((test_embed.shape[0],2,1),dtype=np.int)
    for i in range(test_embed.shape[0]):
        test_idx[i]=[[i+train_embed.shape[0]],[i+train_embed.shape[0]]]
    test_embed=np.concatenate((test_embed,test_idx),axis=2)
    
#models={'ABCNN':ABCNN}
#models={'DA':DA}
#models={'TextCNN':TextCNN}
#models={'TextRNN':TextRNN}
#models={'AttendRNN':AttendRNN}
#models={'CNN_inception':CNN_inception}
models={'DA':DA,
        'AttendRNN':AttendRNN,
        'ABCNN':ABCNN,         
        'CNN_inception':CNN_inception,         
        'TextCNN':TextCNN,
       'TextRNN':TextRNN
        }
parameter={'AttendRNN':AR_hyparameter,'DA':DA_hyparameter,'CNN_inception':inception_hyparameter,
        'TextCNN':CNN_hyparameter,'TextRNN':RNN_hyparameter,'ABCNN':CNN_hyparameter}
com_arg=param()
x_test=tr.from_numpy(test_embed).long().to(device)


for runseed in [1000,2001,3000,4000,5000,5555,6666,7777,8888,9999]:
    np.random.seed(runseed)
    r1=(np.random.uniform(0,1,train_embed.shape[0])*5).astype(int)

    for name,model in models.items():
        print (name,runseed)
        arg=parameter[name]()

        print (arg.weight_decay)
        print (arg.doc_length)
        
        f = open('checkpoint/log.txt','a')
        f.write('\n'+name+'%d'%runseed+'\n')
        f.close()

        cv_score=0.0
        stacking_result=[]
        for v in range(val_fold):##5 means cv,1 means no cv
            print ('now training fold %d'%v)
            gc.collect()
            filter_t=(r1!=v)
            filter_v=(r1==v)
            x_train , y_train = tr.from_numpy(train_embed[filter_t]).long().to(device),tr.from_numpy(labels[filter_t]).float().to(device)
            x_val  ,  y_val  = tr.from_numpy(train_embed[filter_v]).long().to(device),tr.from_numpy(labels[filter_v]).float().to(device)
            print (x_train.shape[0],x_val.shape[0])

            tr.manual_seed(runseed)
            if use_tradition_feat:
                net=model(arg,sta_feat).to(device)
            else:
                net=model(arg).to(device)

            opter=net.get_opter(arg.lr1,arg.lr2)
            criterion = nn.BCELoss(size_average = False)

            net_train.fit(x_train,y_train,net=net,opter=opter,criterion=criterion,batch_size=100,
                    num_epoch=arg.epoch,batch_decay=45,x_val=x_val,y_val=y_val,name=name)


            y_pred=net_train.predict_proba(net,x_val)
            
            cv_score+=log_loss(y_val,y_pred)
            if stacking:
                stacking_result.extend(list(y_pred))

            y_pred=net_train.predict_proba(net,x_test)
            filename='submit/%s_fold_%d_submit.txt'%(name,v)
            np.savetxt(filename,y_pred)

            f = open('checkpoint/log.txt','a')
            f.write("\n")
            f.close()


        print ("cv_score is ",cv_score/5)
        f = open('checkpoint/log.txt','a')
        f.write("cv_score is %f\n"%(cv_score/5))
        f.close()
        if use_tradition_feat:
            filename='stacking/%s_val_result_%d_withsta.txt'%(name,runseed)
        else:  
            filename='stacking/%s_val_result_%d.txt'%(name,runseed)
        np.savetxt(filename,np.array(stacking_result))
        print ('saving stacking files in the dir stacking')

        y_pred=np.zeros(test_embed.shape[0])
        for v in range(5):
            filename='submit/%s_fold_%d_submit.txt'%(name,v)
            y_pred+=np.loadtxt(filename)
        y_pred/=5
        print (y_pred.mean())
        
        if use_tradition_feat:
            filename='submit/%s_pred_%d_withsta.txt'%(name,runseed)
        else:
            filename='submit/%s_pred_%d.txt'%(name,runseed)
        np.savetxt(filename,y_pred)



