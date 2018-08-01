import numpy as np
class param(object):
    pretrained_weight=np.load('data/wordvec4.npy')
    finetune=False
    vocab_size=pretrained_weight.shape[0]
    doc_length=18
    embed_dim=300
    num_class=2
    weight_decay=5e-4
    init_weight=True
    lr1=1e-3
    lr2=0
    epoch=100
    
    
class inception_hyparameter(param):
    kernel_num=[256,128,128,64,32]
    kmax=1
    incept_dim=100
    fc_hiddim=64
    
class DA_hyparameter(param):
    hidden_size=200
    num_layers=1
    useBi=2
    fc_hiddim=64
    compare_dim=400
    weight_decay=1e-4

class AR_hyparameter(param):
    hidden_size=256
    num_layers=1
    useBi=2
    da=400
    fc_hiddim=64

class RNN_hyparameter(param):
    hidden_size=256
    num_layers=1
    dropout=0.5
    useBi=2
    fc_hiddim=64
       
class CNN_hyparameter(param):  
    kernel_num=[256,128,128,64,32]
    kernel_size=[1,2,3,4,5]
    kmax=1
    fc_hiddim=50
    
if __name__ == '__main__':
    arg=CNN_hyparameter()
    print (arg.kmax)
        
        
        
