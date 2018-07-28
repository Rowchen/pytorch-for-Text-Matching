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
class DA(nn.Module):
    def __init__(self,arg):
        super(DA,self).__init__()
        self.N=arg.doc_length

        V=arg.embed_dim
        self.V=V
        self.weight_decay=arg.weight_decay
        self.lr1=arg.lr1
        self.lr2=arg.lr2

        #embedding
        self.embed=nn.Embedding(arg.vocab_size,V,scale_grad_by_freq=True,max_norm=5)
        self.embed.weight.data.copy_(tr.from_numpy(arg.pretrained_weight))
        self.embed.weight.requires_grad = arg.finetune
        self.finetune=arg.finetune

        self.lstm=nn.GRU(arg.input_size, 
                          arg.hidden_size, 
                          num_layers=arg.num_layers,
                          batch_first=True,
                          bidirectional = (arg.useBi==2)
                          )

        #compare
        self.compare = nn.Sequential(
            nn.Linear(arg.hidden_size*arg.useBi*2,arg.compare_dim),
            nn.ReLU(inplace=True),
            #nn.Linear(arg.compare_dim,arg.compare_dim),
            #nn.ReLU(inplace=True),
        )

        #full connect
        self.fc = nn.Sequential(
            nn.Dropout(arg.dropout),
            nn.Linear(arg.compare_dim*4,arg.fc_hiddim),
            nn.ReLU(inplace=True),
            nn.Dropout(arg.dropout),
            nn.Linear(arg.fc_hiddim,1)
        )


        #weight ini
        init.xavier_normal_(self.compare[0].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.fc[1].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.fc[4].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.lstm.all_weights[0][0], gain=np.sqrt(1))
        init.xavier_normal_(self.lstm.all_weights[0][1], gain=np.sqrt(1))

               
        
    def forward(self,x):
        '''
        input  x is [n,2,N,V]
        '''
        x=self.embed(x)#[num,2,N,V]
        B=x.size(0)
        x1=x[:,0,:,:]
        x2=x[:,1,:,:]

        o1a,_=self.lstm(x1)
        o2a=o1a
        o1b,_=self.lstm(x2)
        o2b=o1b
        o3=o2a@(o2b.permute(0,2,1))  # dim1 is a ,dim 2 is b
        o4a=F.softmax(o3,dim=2)   #the attentnion of b for a ,the sum of dim 2 is 1       
        o4b=F.softmax(o3,dim=1).permute(0,2,1)   #the attention of a for b ,after permuting the sum of dim 2 is 1
        #interactive_attention
        o5c=(o4b@o2b)
        o5d=(o4a@o2a)#using compare dim 
        o5a=self.compare(tr.cat([o1a,o5c],dim=2)).permute(0,2,1)
        o5b=self.compare(tr.cat([o1b,o5d],dim=2)).permute(0,2,1)

        #pooling
        o6a=F.avg_pool1d(o5a,o5a.size(2)).squeeze()
        o7a=F.max_pool1d(o5a,o5a.size(2)).squeeze()
        o8a=tr.cat([o6a,o7a],1)
        o6b=F.avg_pool1d(o5b,o5b.size(2)).squeeze()
        o7b=F.max_pool1d(o5b,o5b.size(2)).squeeze()
        o8b=tr.cat([o6b,o7b],1)

        delta1=tr.abs(o8a-o8b)    
        delta2=o8a*o8b
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

