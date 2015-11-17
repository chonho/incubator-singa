#!/usr/bin/env python
from modelconf import *
from datasets import cifar10

#store = Store(path='examples/cifar10/cifar10_train_shard',
#              random_skip=5000, batchsize=64) 
#m.add(Data(load='sharddata', conf=store))
X_train = cifar10.load_data()

m = Sequential('cnn', label=True)

m.add(RGB('examples/cifar10/image_mean.bin'))

parw = Param(init='gaussian', std=0.0001)
parb = Param(lr_scale=2, init='constant', value=0)
m.add(Convolution(32, 5, 1, 2, w_param=parw, b_param=parb))
m.add(Pooling('max'))
m.add(Activation('relu'))
m.add(LRN2D(3, alpha=0.00005, beta=0.75))

m.add(Convolution(32, 5, 1, 2, w_param=parw, b_param=parb))
m.add(Activation('relu'))
m.add(Pooling('avg'))
m.add(LRN2D(3, alpha=0.00005, beta=0.75))

m.add(Convolution(64, 5, 1, 2, w_param=parw, b_param=parb.setval(lr_scale=1)))
m.add(Activation('relu'))
m.add(Pooling('avg'))

parw.setval(wd_scale=250)
parb.setval(lr_scale=2, wd_scale=0)
m.add(Dense(10, w_param=parw, b_param=parb, activation='softmax'))


sgd = SGD(weight_decay=0.004, lr_type='fixed', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))
topo = Cluster('examples/cifar10')
m.compile(updater=sgd, cluster=topo)
m.fit(X_train, train_steps=1000, disp_freq=30)

#TODO classify test data
#result = m.evaluate(x_test, ...)


print
m.display()

