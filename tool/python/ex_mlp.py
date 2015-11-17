#!/usr/bin/env python
from modelconf import *

m = Model('mlp')

store = Store(path='examples/mnist/train_data.bin', backend='kvfile',
              random_skip=5000, batchsize=64, shape=784,
              std_value=127.5, mean_value=127.5)
m.add(Data(load='record', conf=store))

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
m.evaluate(Algorithm(kBP), train_steps=1000, disp_freq=10)

print
m.display()

