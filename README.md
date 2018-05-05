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


