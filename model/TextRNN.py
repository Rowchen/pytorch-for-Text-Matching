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

class TextRNN(nn.Module):
    def __init__(self,arg):
        super(TextRNN,self).__init__()
        #self.N=arg.doc_length
        V=arg.embed_dim
        self.weight_decay=arg.weight_decay
        self.lr1=arg.lr1
        self.lr2=arg.lr2
        self.kmax=arg.kmax
        #embedding
        self.embed=nn.Embedding(arg.vcab_size,V,scale_grad_by_freq=True,max_norm=5)
        self.embed.weight.data.copy_(tr.from_numpy(arg.pretrained_weight))
        self.embed.weight.requires_grad = arg.finetune
        self.finetune=arg.finetune
        #BiLSTM
     
        self.num_layers=arg.num_layers
        self.hidden_size=arg.hidden_size
        self.lstm=nn.LSTM(arg.input_size, 
                          arg.hidden_size, 
                          num_layers=arg.num_layers,
                          batch_first=True,
                          bidirectional = (arg.useBi==2))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size*2*self.kmax*arg.useBi,50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(50,1)
        )
        
        #self.dropout=nn.Dropout(arg.dropout)
        
         #weight init
        init.xavier_normal_(self.fc[1].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.fc[4].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.lstm.all_weights[0][0], gain=np.sqrt(1))
        init.xavier_normal_(self.lstm.all_weights[0][1], gain=np.sqrt(1))
        
        
        
    def forward(self,x):
        '''
        input  x is [n,2,N,V]
        '''
        x=self.embed(x)#[num,2,N,V]
        x1=x[:,0,:,:]
        x2=x[:,1,:,:]
        
        oo1,_=self.lstm(x1)
        oo2,_=self.lstm(x2)
        
        oo1=oo1.permute(0,2,1)
        oo2=oo2.permute(0,2,1)#(batch,hidden,seq)
        
        oo1=oo1.topk(self.kmax,dim=2)[0].view(oo1.size(0),-1)
        oo2=oo2.topk(self.kmax,dim=2)[0].view(oo2.size(0),-1)
        delta1=tr.abs(oo1-oo2)     

        oo3=oo1/tr.sqrt(tr.sum(oo1**2,dim=1,keepdim=True))
        oo4=oo2/tr.sqrt(tr.sum(oo2**2,dim=1,keepdim=True))
        delta2=oo3*oo4
        
        out=tr.cat([delta1,delta2],1)
            
        
        out=F.sigmoid(self.fc(out)).view(-1)
        
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

