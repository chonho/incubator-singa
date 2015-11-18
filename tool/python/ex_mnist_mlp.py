#!/usr/bin/env python
from modelconf import *
from datasets import mnist 

#store = Store(path='examples/mnist/train_data.bin', backend='kvfile',
#              random_skip=5000, batchsize=64, shape=784,
#              std_value=127.5, mean_value=127.5)
#X_train = Data(load='recordinput', conf=store)
X_train = mnist.load_data()

#m = Model('mlp')
m = Sequential('mlp')

par = Param(init='uniform', low=-0.05, high=0.05)
m.add(Dense(2500, w_param=par, b_param=par)) 
m.add(Activation('tanh'))
m.add(Dense(2000, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(1500, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(1000, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(500, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(10, w_param=par, b_param=par, activation='softmax')) 

sgd = SGD(lr=0.001, lr_type='step')
topo = Cluster('examples/mnist')
m.compile(updater=sgd, cluster=topo)
m.fit(X_train, 1000, disp_freq=10)

#TODO classify for test
#result = m.evaluate(x_test, ...)
#acc, loss = m.evaluate(x_test, ...)

print
m.display()

