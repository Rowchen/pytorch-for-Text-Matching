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


class TextCNN(nn.Module):
    def __init__(self,arg,sta_feat=None):
        '''
        wordembed_dim:dim of word embedding
        doc_length:length of a doc,here is a sentence
        the input size is doc_length*wordembed_dim
        '''
        super(TextCNN,self).__init__()
        self.N=arg.doc_length
        V=arg.embed_dim
        Co=arg.kernel_num  #a list
        Ks=arg.kernel_size #a list
        self.weight_decay=arg.weight_decay
        self.lr1=arg.lr1
        self.lr2=arg.lr2
        self.kmax=arg.kmax

        pretrain_weight=tr.from_numpy(arg.pretrained_weight)
        print ('embed_size is ',pretrain_weight.shape)

        self.embed=nn.Embedding(arg.vocab_size,V)
        self.embed.weight.data.copy_(pretrain_weight)
        self.embed.weight.requires_grad = arg.finetune
        self.finetune=arg.finetune
        
        self.branch1 =nn.Sequential(
            nn.Conv1d(V, Co[0], Ks[0],stride=(1)),
            nn.ReLU(inplace=True),
            )
        self.branch2 =nn.Sequential(
            nn.Conv1d(V, Co[1], Ks[1],stride=(1)),
            nn.ReLU(inplace=True),
            )
        self.branch3 =nn.Sequential(
            nn.Conv1d(V, Co[2], Ks[2],stride=(1)),
            nn.ReLU(inplace=True),
            )
        self.branch4 =nn.Sequential(
            nn.Conv1d(V, Co[3], Ks[3],stride=(1)),
            nn.ReLU(inplace=True),
            )
        self.branch5 =nn.Sequential(
            nn.Conv1d(V, Co[4], Ks[4],stride=(1)),
            nn.ReLU(inplace=True),
            )


        self.sta_feat=sta_feat
        if sta_feat is not None:
            add_dim=sta_feat.shape[1]
            self.sta_feat=tr.from_numpy(sta_feat).float()
        else:
            add_dim=0

        self.fc=nn.Sequential(
            nn.Dropout(0.3),
            nn.BatchNorm1d(sum(Co)*2*2*self.kmax),
            nn.Linear(sum(Co)*2*2*self.kmax,50,bias=True),
            nn.ReLU(inplace=True)
            )

        self.fc2=nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm1d(50+add_dim),
            nn.Linear(50+add_dim,1),
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.4),
            #nn.BatchNorm1d(50),
            #nn.Linear(50,1),
            )
        print (self.branch1)
        #weight init
        init.xavier_normal_(self.branch1[0].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch2[0].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch3[0].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch4[0].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.branch5[0].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.fc[2].weight.data, gain=np.sqrt(1))
        init.xavier_normal_(self.fc2[2].weight.data, gain=np.sqrt(1))


    def forward(self,x):
        '''
        input  x is [n,2,N,V]
        '''
        emb=self.embed(x[:,:,:self.N])#[num,2,N,V]
        x1=emb[:,0,:,:].permute(0,2,1)
        x2=emb[:,1,:,:].permute(0,2,1)

        out1=self.branch1(x1)
        out2=self.branch2(x1)
        out3=self.branch3(x1)
        out4=self.branch4(x1)
        out5=self.branch5(x1)
        out6=self.branch1(x2)
        out7=self.branch2(x2)
        out8=self.branch3(x2)
        out9=self.branch4(x2)
        out10=self.branch5(x2)

        out11=F.avg_pool1d(out1,out1.size(2)).view(out1.shape[0],-1)
        out12=F.avg_pool1d(out2,out2.size(2)).view(out1.shape[0],-1)
        out13=F.avg_pool1d(out3,out3.size(2)).view(out1.shape[0],-1)
        out14=F.avg_pool1d(out4,out4.size(2)).view(out1.shape[0],-1)
        out15=F.avg_pool1d(out5,out5.size(2)).view(out1.shape[0],-1)
        out16=F.avg_pool1d(out6,out6.size(2)).view(out1.shape[0],-1)
        out17=F.avg_pool1d(out7,out7.size(2)).view(out1.shape[0],-1)
        out18=F.avg_pool1d(out8,out8.size(2)).view(out1.shape[0],-1)
        out19=F.avg_pool1d(out9,out9.size(2)).view(out1.shape[0],-1)
        out20=F.avg_pool1d(out10,out10.size(2)).view(out1.shape[0],-1)

        out1=F.max_pool1d(out1,out1.size(2)).view(out1.shape[0],-1)
        out2=F.max_pool1d(out2,out2.size(2)).view(out1.shape[0],-1)
        out3=F.max_pool1d(out3,out3.size(2)).view(out1.shape[0],-1)
        out4=F.max_pool1d(out4,out4.size(2)).view(out1.shape[0],-1)
        out5=F.max_pool1d(out5,out5.size(2)).view(out1.shape[0],-1)
        out6=F.max_pool1d(out6,out6.size(2)).view(out1.shape[0],-1)
        out7=F.max_pool1d(out7,out7.size(2)).view(out1.shape[0],-1)
        out8=F.max_pool1d(out8,out8.size(2)).view(out1.shape[0],-1)
        out9=F.max_pool1d(out9,out9.size(2)).view(out1.shape[0],-1)
        out10=F.max_pool1d(out10,out10.size(2)).view(out1.shape[0],-1)


        out11=tr.cat([out1,out2,out3,out4,out5,out11,out12,out13,out14,out15],1)
        out12=tr.cat([out6,out7,out8,out9,out10,out16,out17,out18,out19,out20],1)
        delta1=tr.abs(out11-out12)
        delta2=(out11)*(out12)
    
        out=tr.cat([delta1,delta2],1)
        out=self.fc(out)
        
        if self.sta_feat is not None:
            idx=x[:,0,-1]
            #sta=self.sta_fc(self.sta_feat[idx])
            o1=tr.cat([out,self.sta_feat[idx]],1)
        else:
            o1=out
        o2=self.fc2(o1)
        o3=F.sigmoid(o2).view(-1)
        return o3
    
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

