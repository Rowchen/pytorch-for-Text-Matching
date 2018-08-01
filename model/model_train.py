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

class net_train():
    @staticmethod
    def fit(x,y,net,opter,criterion,batch_size=100,num_epoch=1,seed=2018,x_val=None,y_val=None,batch_decay=100,stop_ite=12,eps=0.005,name=None):
        device = tr.device('cuda')
        best_score=1000
        lr1=net.lr1
        net.train()
        dataset=tr.utils.data.TensorDataset(x,y)
        train_loader= tr.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
        best_score=99999
        valloss=0
        val_stop_cnt=0
        total_ite=len(train_loader)
        for epoch in range(num_epoch):
            epoch_loss=0.0
            for ite, (train, label) in enumerate(train_loader):  
                train=train.to(device)
                label=label.to(device)
                output=net(train)
                loss=criterion(output,label)/train.shape[0]
                epoch_loss+=loss*train.shape[0]
                opter.zero_grad()
                loss.backward()
                opter.step()
                
                if (ite+1)%batch_decay==0 or ite==(total_ite-1):
                    valloss=net_train.cal_all_loss(net,x_val,y_val,criterion)  
                    #print('epoch[{}/{}],decay[{}],validation loss {:.4f}'.format(epoch+1,num_epoch,ite//batch_decay,valloss))  
                    
                    if best_score>valloss+eps:
                        best_score=valloss
                        tr.save(net.state_dict(), 'checkpoint/best_params_%s.pkl'%name)
                        val_stop_cnt=0
                    else:
                        val_stop_cnt+=1
                        if val_stop_cnt>=stop_ite:
                            print ('stop improve ! quit training ! best score is {:.4f}'.format(best_score))
                            break

            
            epoch_loss/= x.shape[0]
            if val_stop_cnt>=stop_ite:
                #load the best model
                net.load_state_dict(tr.load('checkpoint/best_params_%s.pkl'%name))
                epoch_loss=net_train.cal_all_loss(net,x,y,criterion)  
                best_score=net_train.cal_all_loss(net,x_val,y_val,criterion)  

            f = open('checkpoint/log.txt','a')
            print('epoch[{}/{}],train loss {:.4f},best_score {:.4f},lr {:.5f}'
                          .format(epoch+1,num_epoch,epoch_loss,best_score,lr1))  
            f.write('epoch[{}/{}],train loss {:.4f},best_score {:.4f},lr {:.5f}\n'
                          .format(epoch+1,num_epoch,epoch_loss,best_score,lr1))
            f.close()

            if val_stop_cnt>=stop_ite:
                break

    @staticmethod
    def cal_all_loss(net,x,y,criterion):
        device = tr.device('cuda')
        net.eval()
        dataset=tr.utils.data.TensorDataset(x,y)
        test_loader=tr.utils.data.DataLoader(dataset=dataset,
                                             batch_size=100,
                                             shuffle=False)
        all_loss=0.0
        with tr.no_grad():
            for i,(train,label) in enumerate(test_loader):
                train=train.to(device)
                label=label.to(device)
                output=net(train)
                all_loss+=criterion(output,label)
        net.train()
        return all_loss/x.shape[0]
    
    @staticmethod
    def predict_proba(net,x):
        device = tr.device('cuda')
        net.eval()
        dataset=tr.utils.data.TensorDataset(x,tr.zeros(x.shape[0]).long())
        test_loader=tr.utils.data.DataLoader(dataset=dataset,
                                             batch_size=100,
                                             shuffle=False)
        ans=[]
        with tr.no_grad():
            for i,(train,label) in enumerate(test_loader):
                train=train.to(device)
                label=label.to(device)
                output=net(train)
                out=output.cpu()
                ans.extend(list(out.numpy()))
        ans=np.array(ans)
        net.train()
        return ans




