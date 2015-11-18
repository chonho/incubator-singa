### Layer class (inherited)

* Data
* Dense
* Activation
* Convolution
* Pooling
* LRN2D 
* Dropout
* RGB 

### Other classes

* Store
* Param
* SGD
* Cluster

### Model class

* Model class has `jobconf` (JobProto) and `layers` (layer list)

Methods in Model class

* add
	* add Layer class into Model class

* compile	
	* set Updater and Cluster
	* build model, i.e., connect layers

* fit 
	* set Train_one_batch and parameter values for training/testing/validation
	* [IN PROGRESS] run singa via a wrapper for Driver class
	* [IN PROGRESS] recieve training/testing/validation accuracy

* evaluate
	* [IN PROGRESS] run singa for tasks, e.g., classification/prediction


## MLP Example

An example (to generate job.conf for mnist)

```
X_train = mnist.load_data()

m = Sequential('mlp')  # inherited from Model 

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

#(TODO) classify for test
#result = m.evaluate(x_test, ...)
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
m.add(Dense(2500, w_param=Param(init='uniform'), b_param=Param(init='gaussian')))
```


## CNN Example

An example (to generate job.conf for cifar10)

```
X_train = cifar10.load_data()

m = Sequential('cnn', label=True)

m.add(RGB('examples/cifar10/image_mean.bin'))

parw = Param(init='gauss', std=0.0001)
parb = Param(lr_scale=2, init='const', value=0)
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
m.fit(X_train, 1000, disp_freq=30)
```

