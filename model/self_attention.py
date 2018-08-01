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
class AttendRNN(nn.Module):
    def __init__(self,arg,sta_feat=None):
        super(AttendRNN,self).__init__()
        self.N=arg.doc_length
        V=arg.embed_dim
        self.V=V
        self.weight_decay=arg.weight_decay
        self.lr1=arg.lr1
        self.lr2=arg.lr2

        #embedding   normalize
        pretrain_weight=tr.from_numpy(arg.pretrained_weight)
        self.embed=nn.Embedding(arg.vocab_size,V)
        self.embed.weight.data.copy_(pretrain_weight)
        self.embed.weight.requires_grad = arg.finetune
        self.finetune=arg.finetune

        self.lstm=nn.GRU(V, 
                          arg.hidden_size, 
                          num_layers=arg.num_layers,
                          batch_first=True,
                          bidirectional = (arg.useBi==2)
                        )
        
        self.sta_feat=sta_feat
        if sta_feat is not None:
            add_dim=sta_feat.shape[1]
            sta_feat2=(sta_feat-np.mean(sta_feat,axis=0))/np.std(sta_feat,axis=0)
            self.sta_feat=tr.from_numpy(sta_feat2).float().cuda()
        else:
            add_dim=0
            
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(arg.hidden_size*arg.useBi*4+add_dim,arg.fc_hiddim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(arg.fc_hiddim,1)
        )

        #weight ini
        init.xavier_normal_(self.fc[1].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.fc[4].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.lstm.all_weights[0][0], gain=np.sqrt(1))
        init.xavier_normal_(self.lstm.all_weights[0][1], gain=np.sqrt(1))
        
        
        self.sigma=nn.Parameter(tr.Tensor([0.95]))
        self.sigma.requires_grad=False
        self.dist=tr.empty(self.N,self.N).float()
        self.dist.requires_grad=False
        for i in range(self.N):
            for j in range(self.N):
                self.dist[i,j]=(i-j)*(i-j)
        self.dist=self.dist.cuda()
               
        self.count=0
        
       
        
    def forward(self,x):
        '''
        input  x is [n,2,N,V]
        '''
       
        B=x.size(0)
        emb=self.embed(x[:,:,:self.N])#[num,2,N,V]  self.N is the idx of sta_feat
        x1=emb[:,0,:,:]
        x2=emb[:,1,:,:]

        #calculate the self-attention
        o1a,_=self.lstm(x1)
        o2a=o1a
        o3a=o2a@(o2a.permute(0,2,1))#(b,N,N),self-attention
        o3a=o3a-self.dist/self.sigma[0]
        
        o4a=F.softmax(o3a,dim=2)#attention proba
        o5a=(o4a@o2a).permute(0,2,1)#(b,da,N)
        o6a=F.avg_pool1d(o5a,o5a.size(2)).squeeze()
        o7a=F.max_pool1d(o5a,o5a.size(2)).squeeze()
        o8a=tr.cat([o6a,o7a],1)

        o1b,_=self.lstm(x2)
        o2b=o1b
        o3b=o2b@(o2b.permute(0,2,1))#(b,N,N),self-attention
        #here some model add the sequence impact!!  score=(eij+di-j),the  word j is more far from word i ,the score is lower!
        o3b=o3b-self.dist/self.sigma[0]
        
        o4b=F.softmax(o3b,dim=2)#attention proba   here dim=1 or dim=2 has different meaning!!!!!
        o5b=(o4b@o2b).permute(0,2,1)#(b,da,N)
        o6b=F.avg_pool1d(o5b,o5b.size(2)).squeeze()
        o7b=F.max_pool1d(o5b,o5b.size(2)).squeeze()
        o8b=tr.cat([o6b,o7b],1)

        delta1=tr.abs(o8a-o8b)    
        delta2=o8a*o8b
        
        idx=x[:,0,-1]
        if self.sta_feat is not None:
            out=tr.cat([delta1,delta2,self.sta_feat[idx]],1)
        else:
            out=tr.cat([delta1,delta2],1)

        out=F.sigmoid(self.fc(out)).view(-1)
        self.count+=1
#         if self.count%100==0:
#             print (self.sigma[0])
        
        return out
    
    
    
    
    def get_opter(self,lr1,lr2):
        ignored_params = list(map(id, self.embed.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params , self.parameters())
        if not self.finetune:
            opter=opt.Adam ([dict(params=base_params,weight_decay =self.weight_decay,lr=lr1)])
        else:
            opter=opt.Adam ([
                            dict(params=base_params,weight_decay =self.weight_decay,lr=lr1),
                            {'params': self.embed.parameters(), 'lr': lr2}
                            ]) 
        return opter

