import numpy as np

class inception_hyparameter(object):
    pretrained_weight=np.load('data/wordvec4.npy')
    kernel_num=[256,128,128,64,32]
    incept_dim=100
    finetune=False
    vcab_size=pretrained_weight.shape[0]
    doc_length=18
    embed_dim=300
    num_class=2
    weight_decay=1e-5
    init_weight=True
    lr1=1e-3
    lr2=0
    epoch=100
    kmax=1


class SDA_hyparameter(object):
    pretrained_weight=np.load('data/wordvec4.npy')
    vocab_size=pretrained_weight.shape[0]
    finetune=False
    doc_length=18
    embed_dim=300
    num_class=2
    weight_decay=1e-5
    init_weight=True
    lr1=1e-3
    lr2=0
    input_size=300
    hidden_size=200
    num_layers=1
    dropout=0.5
    epoch=100
    useBi=2
    fc_hiddim=64
    compare_dim=800

    
    
    

class DA_hyparameter(object):
    pretrained_weight=np.load('data/wordvec4.npy')
    vocab_size=pretrained_weight.shape[0]
    finetune=False
    doc_length=18
    embed_dim=300
    num_class=2
    weight_decay=1e-5
    init_weight=True
    lr1=1e-3
    lr2=0
    input_size=300
    hidden_size=200
    num_layers=1
    dropout=0.5
    epoch=100
    useBi=2
    fc_hiddim=100
    compare_dim=400

    
    
    
    
    
class AR_hyparameter(object):
    pretrained_weight=np.load('data/wordvec4.npy')
    vocab_size=pretrained_weight.shape[0]
    finetune=False
    doc_length=18
    embed_dim=300
    num_class=2
    weight_decay=1e-5
    init_weight=True
    lr1=1e-3
    lr2=0
    input_size=300
    hidden_size=256
    num_layers=1
    dropout=0.5
    epoch=100
    useBi=2
    da=400
    r=5
    fc_hiddim=64

class RNN_hyparameter(object):
    pretrained_weight=np.load('data/wordvec4.npy')
    vcab_size=pretrained_weight.shape[0]
    finetune=False
    doc_length=18
    embed_dim=300
    num_class=2
    weight_decay=1e-5
    init_weight=True
    lr1=1e-3
    lr2=0
    input_size=300
    hidden_size=256
    num_layers=1
    dropout=0.5
    epoch=100
    kmax=1
    useBi=2
    
    
        
class CNN_hyparameter(object):  
    pretrained_weight=np.load('data/wordvec4.npy')
    kernel_num=[256,128,128,64,32]
    kernel_size=[1,2,3,4,5]
    finetune=False
    vocab_size=pretrained_weight.shape[0]
    doc_length=18
    embed_dim=300
    num_class=2
    weight_decay=1e-5
    init_weight=True
    lr1=1e-3
    lr2=0
    epoch=100
    kmax=1
    
class RCNN_hyparameter(object):
    pretrain_word_vec=np.load('data/wordvec4.npy')
    vcab_size=pretrain_word_vec.shape[0]
    pretrained_weight=pretrain_word_vec
    finetune=False
    doc_length=18
    embed_dim=300
    num_class=2
    weight_decay=0
    init_weight=True
    lr1=5e-3
    lr2=0
    input_size=300
    hidden_size=200
    
    num_layers=1
    dropout=0.5
    epoch=100
    kmax=1
    useBi=1
    kernel_size=2
    kernel_num=500
if __name__ == '__main__':
    arg=CNN_hyparameter()
    print (arg.kmax)
        
        
        
