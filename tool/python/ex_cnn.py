#!/usr/bin/env python
from modelconf import *
#from configure import *

m = Model('cnn', label=True)

store = Store(path='examples/cifar10/cifar10_train_shard',
              random_skip=5000, batchsize=64) 
m.add(Data(load='shard', conf=store))
m.add(RGB('examples/cifar10/image_mean.bin'))

parw = Param(init='gauss', std=0.0001)
parb = Param(lr_scale=2, init='const', value=0)
m.add(Convolution(32, 5, 1, 2, w_param=parw, b_param=parb))
m.add(Pooling('max'))
m.add(Activation('relu'))
m.add(Norm(3, alpha=0.00005, beta=0.75))

m.add(Convolution(32, 5, 1, 2, w_param=parw, b_param=parb))
m.add(Activation('relu'))
m.add(Pooling('avg'))
m.add(Norm(3, alpha=0.00005, beta=0.75))

m.add(Convolution(64, 5, 1, 2, w_param=parw, b_param=parb.setval(lr_scale=1)))
m.add(Activation('relu'))
m.add(Pooling('avg'))

parw.setval(wd_scale=250)
parb.setval(lr_scale=2, wd_scale=0)
m.add(Dense(10, w_param=parw, b_param=parb, activation='softmax'))


sgd = SGD(weight_decay=0.004, lr_type='fixed', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))
topo = Cluster('examples/cifar10')
m.compile(updater=sgd, cluster=topo)
m.evaluate(Algorithm(kBP), train_steps=1000, disp_freq=30)



#m.add(Dense(2500))
#m.add(Activation('tanh'))
##d = Dense(64, input_dim=20, init='uniform')
##m.add(d)

#m.add(Dense(16, init='uniform'))
#m.add(Dense(16, init='uniform', 
#            w_param=Param(low=-0.05, high=0.05),
#            b_param=Param(low=-0.05, high=0.05)))

#m.add(Activation('softmax'))
#m.add(Activation('softmax', 3))
#m.add(Dense(2, activation='softmax'))
#m.add(Dropout(0.5))

#sgd = SGD(lr=0.001, change='step')
#sgd = SGD(lr=0.001, change='step', momentum=0.1, weight_decay = 0.2, delta=0.3)
#m.compile(loss='mean_squared_error', optimizer=sgd)
#topo = Cluster(workspace='examples/mnist')
#m.compile(updater=sgd, cluster=topo)
#m.build()

#alg = Algorithm(kBP)
#m.evaluate(algorithm=alg, train_steps=1000)
#m.evaluate(alg, train_steps=1000)
#m.evaluate(Algorithm(kBP), train_steps=1000, disp_freq=10)
#m.evaluate(Algorithm(kCD, cd_k=1), train_steps=1000)

print
m.display()

