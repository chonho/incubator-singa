#!/usr/bin/env python
from model import *
from datasets import cifar10

X_train, X_test, workspace = cifar10.load_data()

m = Sequential('cnn', label=False)

parw = Param(init='gaussian', std=0.0001)
parb = Param(lr_scale=2, init='constant', value=0)
m.add(Convolution2D(32, 5, 1, 2, w_param=parw, b_param=parb))
m.add(MaxPooling2D(pool_size=(3,3), stride=2))
m.add(Activation('relu'))
m.add(LRN2D(3, alpha=0.00005, beta=0.75))

m.add(Convolution2D(32, 5, 1, 2, w_param=parw, b_param=parb))
m.add(Activation('relu'))
m.add(AvgPooling2D(pool_size=(3,3), stride=2))
m.add(LRN2D(3, alpha=0.00005, beta=0.75))

m.add(Convolution2D(64, 5, 1, 2, w_param=parw, b_param=parb.setval(lr_scale=1)))
m.add(Activation('relu'))
m.add(AvgPooling2D(pool_size=(3,3), stride=2))

parw.setval(wd_scale=250)
parb.setval(lr_scale=2, wd_scale=0)
m.add(Dense(10, w_param=parw, b_param=parb, activation='softmax'))

sgd = SGD(decay=0.004, lr_type='fixed', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))
topo = Cluster(workspace)
m.compile(optimizer=sgd, cluster=topo)
m.fit(X_train, train_steps=1000, disp_freq=30)
result = m.evaluate(X_test, test_steps=10, test_freq=300)

#print
#m.display()

