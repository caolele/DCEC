# Tensorflow Implemantation of CAE and DCEC

Abbreviations:

- DCEC: Deep Clustering with Convolutional Autoencoders (Xifeng Guo, Xinwang Liu, En Zhu, Jianping Yin. 
Deep Clustering with Convolutional Autoencoders. ICONIP 2017.)

- CAE: Convolutional Autoencoder (initially evaluated in [this paper](https://arxiv.org/abs/1703.07980))

This repo is forked from [here](https://github.com/XifengGuo/DCEC), yet this implementation get rid of the dependency of Keras and scikit-learn.

## To Run Mnist Examples
Pre-request: Install Tensorflow 1.4.1
`pip install tensorflow==1.4.1`

### Train CAE
Run:   
`python CAE_TF.py --epoch=2 --ver=1 --bs=256 --lr=0.005 --dir=./dump`

### Train DCEC
Run: `python TF_DCEC.py` with default parameters:  
>'iterCAE':210000,  
>'iterDCEC':700000,  
>'updateIntervalDCEC':140,  
>'kmeansTrainSteps':1000,  
>'cluster':10,  
>'ver':1,  
>'bs':256,  
>'lr':0.001,  
>'dir':'./dump'  

Change the parameters by `python TF_DCEC.py --iterCAE=10000 --cluster=16` and so on ...

Example terminal printout:
```
Using params: {'iterCAE': 210000, 'iterDCEC': 700000, 'updateIntervalDCEC': 140, 'kmeansTrainSteps': 1000, 'cluster': 10, 'ver': 1, 'bs': 256, 'lr': 0.001, 'dir': './dump'}
Using TensorFlow backend.
MNIST: (70000, 28, 28, 1)
CAE Pre-training ...
[============================================================] 100.0% ... CAE_Loss=0.021142, Lr=0.000731, Epoch=1/3
[============================================================] 100.0% ... CAE_Loss=0.017476, Lr=0.000534, Epoch=2/3
[============================================================] 100.0% ... CAE_Loss=0.016387, Lr=0.000390, Epoch=3/3
Initializing and Training K-means ...
[============================================================] 100.0% ... Avg.Distance=274.427216, Step=1000
DCEC Fine-tuning ...
Epoch:0, Accuracy:0.622030, Scale_of_label_change:0.000000
[============================================================] 100.0% ... CAE_Loss=0.040661, KL_Loss=27.561775, Loss=2.796839, Lr=0.000285, Epoch=1/10
Epoch:1, Accuracy:0.633030, Scale_of_label_change:0.060857
[============================================================] 100.0% ... CAE_Loss=0.045316, KL_Loss=35.555901, Loss=3.600906, Lr=0.000208, Epoch=2/10
Epoch:2, Accuracy:0.660490, Scale_of_label_change:0.083857
[============================================================] 100.0% ... CAE_Loss=0.045954, KL_Loss=34.958328, Loss=3.541786, Lr=0.000152, Epoch=3/10
Epoch:3, Accuracy:0.667530, Scale_of_label_change:0.034700
[============================================================] 100.0% ... CAE_Loss=0.045734, KL_Loss=35.869713, Loss=3.632705, Lr=0.000111, Epoch=4/10
Epoch:4, Accuracy:0.670290, Scale_of_label_change:0.017114
[============================================================] 100.0% ... CAE_Loss=0.048054, KL_Loss=38.074066, Loss=3.855461, Lr=0.000081, Epoch=5/10
Epoch:5, Accuracy:0.670130, Scale_of_label_change:0.010229
[============================================================] 100.0% ... CAE_Loss=0.048234, KL_Loss=36.597973, Loss=3.708031, Lr=0.000059, Epoch=6/10
```


