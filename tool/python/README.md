### Layer class (inherited)

* Data
* Dense
* Activation
* Convolution
* Pooling
* Norm
* Dropout
* RGB 

### Other classes

* Store
* Param
* SGD
* Cluster
* Algorithm

### Model class

* Model class has `jobconf` (JobProto) and `layers` (layer list)

Methods in Model class

* add
	* add Layer class into Model class

* compile	
	* set Updater and Cluster
	* build model, i.e., connect layers

* evaluate
	* set Algorithm (i.e., train_one_batch) and parameter values for training
	* [IN PROGRESS] run singa via a wrapper for Driver class


## MLP Example

An example (to generate job.conf for mnist)

```
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
```

Hidden layers can be written as
```
par = Param(init='uniform', low=-0.05, high=0.05)
for n in [2500, 2000, 1500, 1000, 500]:
  m.add(Dense(n, w_param=par, b_param=par, activation='tanh'))
m.add(Dense(10, w_param=par, b_param=par, activation='softmax'))
```

Alternative ways to write the following lines
```
m.add(Dense(2500, w_param=par, b_param=par))
m.add(Activation('tanh'))
```
```
m.add(Dense(2500, init='uniform', activation='softmax'))
```
```
m.add(Dense(2500, w_param=Param(init='uniform'), b_param=Param(init='gauss')))
```


## CNN Example

An example (to generate job.conf for cifar10)

```
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
```

