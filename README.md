# Pytorch--Text matching

This is a code for text matching,all the Deep model is run on the platform of **pytorch**

It is a competition about [CIKM spanish text matching](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.7b7633afFVoLQR&raceId=231661)<br>

## Environment
python 3.6 or python 3.5<br>
pytorch 0.4.0 for cpu or gpu (This code main runs on cpu)<br>
tqdm<br>
scikit-learn<br>




## Code organization
`data` is used to save train,test,word-embeding vector or temporary file<br>
`model` is used to save all kinds of Deep models<br>
`stacking` is used to save predicted result on validation set of all kinds of models<br>
`data_propresse` is used to preprocess data<br>
`submit` is used to save submit file<br>

## Data preprocess
run the file `runme.py` in the dir `data_propresse`

## Train 
run the file `train.py` 

## stacking
run the file `stacking.py`
