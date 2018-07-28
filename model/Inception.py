import torch as tr
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as opt
import pandas as pd
import numpy as np
import torchvision
from torch.autograd import Variable
import torch.nn.utils as utils
class Inception(nn.Module):
    def __init__(self,cin,mid,co):
        super(Inception, self).__init__()
        self.branch1 =nn.Conv1d(cin,co[0], 1,stride=1)
        self.branch2 =nn.Sequential(
             nn.Conv1d(cin,mid, 1),
             nn.ReLU(inplace=True),
             nn.Conv1d(mid,co[1],2,stride=1,padding=1),
            )
        self.branch3 =nn.Sequential(
            nn.Conv1d(cin,mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid,co[2], 3,stride=1,padding=1),
            )
        self.branch4 =nn.Sequential(
            nn.Conv1d(cin,mid,1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid,co[3], 4,stride=1,padding=2),
            )
        self.branch5 =nn.Sequential(
            nn.Conv1d(cin,mid,1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid,co[4], 5,stride=1,padding=2),
            )
        init.xavier_normal_(self.branch1.weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch2[0].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch2[2].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch3[0].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch3[2].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch4[0].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch4[2].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch5[0].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch5[2].weight.data, gain=np.sqrt(1))

    def forward(self,x):
        branch1=self.branch1(x)        
        branch2=self.branch2(x)[:,:,:-1]
        branch3=self.branch3(x)
        branch4=self.branch4(x)[:,:,:-1]
        branch5=self.branch5(x)
        result=F.relu(tr.cat((branch1,branch2,branch3,branch4,branch5),1))
        return result

#if __name__=='__main__':
 #   a=tr.rand(100,300,18)
  #  net=Inception(300,100)
  #  out=net(a)
  #  print (out.size())





class CNN_inception(nn.Module):
    def __init__(self,arg):
        '''
        wordembed_dim:dim of word embedding
        doc_length:length of a doc,here is a sentence
        the input size is doc_length*wordembed_dim
        '''
        super(CNN_inception,self).__init__()
        self.N=arg.doc_length
        V=arg.embed_dim
        co=arg.kernel_num  #a list
        self.weight_decay=arg.weight_decay
        self.lr1=arg.lr1
        self.lr2=arg.lr2
        self.kmax=arg.kmax

        self.embed=nn.Embedding(arg.vcab_size,V,scale_grad_by_freq=True,max_norm=5)
        self.embed.weight.data.copy_(tr.from_numpy(arg.pretrained_weight))
        self.embed.weight.requires_grad = arg.finetune
        self.finetune=arg.finetune
        


        self.conv=nn.Sequential(
            Inception(V,arg.incept_dim,co),#(batch_size,64,opt.title_seq_len)->(batch_size,32,(opt.title_seq_len)/2)
            #Inception(300,incept_dim,co2),
            nn.MaxPool1d(arg.doc_length)
            )

        self.fc=nn.Sequential(
            nn.Dropout(0.2),
            nn.BatchNorm1d(sum(co)*2),
            nn.Linear(sum(co)*2,64,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64,1,bias=True)
            )

        #weight init
        
    def forward(self,x):
        '''
        input  x is [n,2,N,V]
        '''
        x=self.embed(x)#[num,2,N,V]
        x1=x[:,0,:,:].permute(0,2,1)
        x2=x[:,1,:,:].permute(0,2,1)
        
        out1a=self.conv(x1).view(x.shape[0],-1)
        out1b=self.conv(x2).view(x.shape[0],-1)

        delta1=tr.abs(out1a-out1b)
        delta2=(out1a)*(out1b)

        out=tr.cat([delta1,delta2],1)
        out=self.fc(out)
        out=F.sigmoid(out).view(-1)
        return out


    
    def get_opter(self,lr1,lr2):
        ignored_params = list(map(id, self.embed.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                        self.parameters())
        if not self.finetune:
            opter=opt.Adam ([dict(params=base_params,weight_decay =self.weight_decay,lr=lr1)])
        else:
            opter=opt.Adam ([
                            dict(params=base_params,weight_decay =self.weight_decay,lr=lr1),
                            {'params': self.embed.parameters(), 'lr': lr2}
                            ]) 
        return opter


if __name__ =='__main__':
    from config import *
    arg=CNN_hyparameter()
    net=CNN_inception(arg)
    a=tr.rand(100,2,18).long()
    out=net(a)
    print (out.size())



