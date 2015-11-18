#!/usr/bin/env python
from modelconf import * 

def load_data(
         path = 'examples/mnist/train_data.bin',
         backend = 'kvfile',
         random = 5000,
         batchsize = 64,
         shape = 784,
         std = 127.5,
         mean = 127.5
      ):

  store = Store(path=path, backend=backend,
                random_skip=random, batchsize=batchsize, shape=shape,
                std_value=std, mean_value=mean)

  data = Data(load='recordinput', conf=store)

  return data

