#!/usr/bin/env python
from modelconf import *

def load_data(
         path = 'examples/cifar10/cifar10_train_shard',
         backend = 'kvfile',
         random = 5000,
         batchsize = 64,
         shape = 784,
         std = 127.5,
         mean = 127.5
      ):

  store = Store(path=path,
              random_skip=random, batchsize=batchsize) 

  data = Data(load='sharddata', conf=store)

  return data

