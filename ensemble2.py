import numpy as np
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
labels=np.load('data/train_labels4.npy')
print (labels.shape)


on=0.247803738317757
stack_result=pd.DataFrame({'labels':labels})
seeds=[1000,2001,3000,4000,5000,5555,6666,7777,8888,9999]
models=['TextCNN','TextRNN','AttendRNN','ABCNN','CNN_inception','DA']
avgname=['TextCNN','TextRNN','at_RNN','at_CNN','inception','inter_RNN']
allavg=np.zeros(stack_result.shape[0])
llls=np.zeros(stack_result.shape[0])
for model,name2 in zip(models,avgname):
    avgpred=np.zeros(stack_result.shape[0])
    for runseed in seeds:
        np.random.seed(runseed)
        r1=(np.random.uniform(0,1,labels.shape[0])*5).astype(int)
        name=model+'_val_result_%d'%runseed
        tmp=np.loadtxt('stacking/'+name+'.txt')
        count=0
        for i in range(5):
            ls=((r1==i).sum())
            stack_result.loc[r1==i,name]=tmp[count:count+ls]
            count+=ls
        avgpred+=np.log(stack_result[name]/(1-stack_result[name]))
        
    avgpred/=len(seeds)
    allavg+=avgpred
    
    avgpred=1/(1+np.exp(-avgpred))
    
    stack_result[name2]=avgpred
    print (name2,log_loss(stack_result['labels'],stack_result[name2]),np.mean(stack_result[name2]))
    
    off=np.mean(stack_result[name2])
    pred=(on/off*stack_result[name2])/((on/off)*stack_result[name2]+((1-on)/(1-off)*(1-stack_result[name2])))
    print (name2,log_loss(stack_result['labels'],pred),np.mean(pred))
    llls+=np.log(pred/(1-pred))
    
allavg_ls=allavg/len(models)
allavg_ls=1/(1+np.exp(-allavg_ls))
print ('offline',log_loss(stack_result['labels'],allavg_ls))
llls2=llls/len(models)
llls2=1/(1+np.exp(-llls2))
print ('after leaky',log_loss(stack_result['labels'],llls2))
print ()



test_result=pd.DataFrame()
seeds=[1000,2001,3000,4000,5000,5555,6666,7777,8888,9999]
models=['TextCNN','TextRNN','AttendRNN','ABCNN','CNN_inception','DA']
avgname=['TextCNN','TextRNN','at_RNN','at_CNN','inception','inter_RNN']
avgname2=['TextCNN_withsta','TextRNN_withsta','at_RNN_withsta','at_CNN_withsta','inception_withsta','inter_RNN_withsta']
for model,name2 in zip(models,avgname):
    avgpred=np.zeros(10000)
    for runseed in seeds:
        name=model+'_pred_%d'%runseed
        test_result[name]=np.loadtxt('submit/'+name+'.txt')
        avgpred+=np.log(test_result[name]/(1-test_result[name]))
    avgpred/=len(seeds)
    avgpred=1/(1+np.exp(-avgpred))
    print (name2,np.mean(avgpred))
    test_result[name2]=avgpred



test_result[avgname].to_csv('cyx_testb.txt',sep='\t',index=False)
stack_result[avgname].to_csv('cyx_stacking.txt',sep='\t',index=False)
