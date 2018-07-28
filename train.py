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
from model.TextCNN import *
from model.TextRNN import *
from config import *
from model.RCNN import *
from model.model_train import *
from model.Inception import *
from model.interactive_attention import DA
from model.self_interactive_attention import SDA
from model.self_attention import AttendRNN
from model.ABCNN import ABCNN
from model.attention_Inception import at_inception

train_embed=np.load('data/train_embed4.npy')
test_embed=np.load('data/test_embed4.npy')
labels=np.load('data/train_labels4.npy')
val_fold=5
stacking=True
device = tr.device('cpu')

use_tradition_feat=False
if use_tradition_feat:
    print ('using traiditional feature')
    train_sta=np.load('data/train_tradition_feat.npy')
    test_sta=np.load('data/test_tradition_feat.npy')
    sta_feat=np.concatenate((train_sta,test_sta),axis=0)

    train_idx=np.zeros((train_embed.shape[0],2,1),dtype=np.int)
    for i in range(train_embed.shape[0]):
        train_idx[i]=[[i],[i]]
    train_embed=np.concatenate((train_embed,train_idx),axis=2)
    test_idx=np.zeros((test_embed.shape[0],2,1),dtype=np.int)
    for i in range(test_embed.shape[0]):
        test_idx[i]=[[i+train_embed.shape[0]],[i+train_embed.shape[0]]]
    test_embed=np.concatenate((test_embed,test_idx),axis=2)

models={'ABCNN':ABCNN}
models={'at_inception':at_inception}
#models={'DA':DA}
#models={'TextCNN':TextCNN}
#models={'AttendRNN':AttendRNN}
#models={'CNN_inception':CNN_inception,'TextCNN':TextCNN,'TextRNN':TextRNN}
#models={'CNN_inception':CNN_inception,'TextRNN':TextRNN}
#models={'CNN_inception':CNN_inception,'TextCNN':TextCNN,'TextRNN':TextRNN,
       # 'DA':DA,'AttendRNN':AttendRNN}

parameter={'AttendRNN':AR_hyparameter,'DA':DA_hyparameter,'CNN_inception':inception_hyparameter,
        'TextCNN':CNN_hyparameter,'TextRNN':RNN_hyparameter,'at_inception':inception_hyparameter,'ABCNN':CNN_hyparameter}


np.random.seed(2018)
r1=(np.random.uniform(0,1,train_embed.shape[0])*5).astype(int)
x_test=tr.from_numpy(test_embed).long().to(device)

for name,model in models.items():
    print (name)
    arg=parameter[name]
    f = open('checkpoint/log.txt','a')
    f.write('\n'+name+'\n')
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

        tr.manual_seed(2018)
        if use_tradition_feat:
            net=model(arg,sta_feat).to(device)
        else:
            net=model(arg).to(device)
        opter=net.get_opter(arg.lr1,arg.lr2)
        criterion = nn.BCELoss(size_average = False)

        net_train.fit(x_train,y_train,net=net,opter=opter,criterion=criterion,batch_size=100,
                num_epoch=arg.epoch,batch_decay=43,x_val=x_val,y_val=y_val,name=name)


        y_pred=net_train.predict_proba(net,x_val)
        print (y_pred.min(),y_pred.max())
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
    

    filename='stacking/%s_val_result.txt'%name
    np.savetxt(filename,np.array(stacking_result))
    print ('saving stacking files in the dir stacking')

    y_pred=np.zeros(test_embed.shape[0])
    for v in range(5):
        filename='submit/%s_fold_%d_submit.txt'%(name,v)
        y_pred+=np.loadtxt(filename)
    y_pred/=5
    print (y_pred.mean())

    filename='submit/%s_pred.txt'%(name)
    np.savetxt(filename,y_pred)

    on=0.3
    off=0.2433
    y_pred=(on/off*y_pred)/((on/off)*y_pred+((1-on)/(1-off)*(1-y_pred)))
    print (y_pred.mean(),y_pred.max(),y_pred.min())

    filename='submit/%s_fold_submit.txt'%(name)
    np.savetxt(filename,y_pred)




