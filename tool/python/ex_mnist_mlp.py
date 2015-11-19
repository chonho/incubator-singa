#!/usr/bin/env python
from modelconf import *
from datasets import mnist 

X_train, X_test = mnist.load_data()

#m = Model('mlp')
m = Sequential('mlp')

par = Param(init='uniform', low=-0.05, high=0.05)
m.add(Dense(2500, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(2000, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(1500, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(1000, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(500, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(10, w_param=par, b_param=par, activation='softmax')) 

sgd = SGD(lr=0.001, lr_type='step')
topo = Cluster('examples/mnist')
m.compile(updater=sgd, cluster=topo)
m.fit(X_train, train_steps=1000, disp_freq=50)
m.evaluate(X_test, batch_size=100, test_steps=10)

#TODO---- classify/predict for new data
#result = m.predict(data_new, ...)
#-------

#print
#m.display()

