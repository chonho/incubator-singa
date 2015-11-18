#!/usr/bin/env python
from modelconf import *

def load_data(
         path = 'examples/cifar10/train_data.bin',
         path_mean = 'examples/cifar10/image_mean.bin',
         backend = 'kvfile',
         batchsize = 64,
         random = 5000,
         shape = (3, 32, 32),
         std = 127.5,
         mean = 127.5
      ):

  store = Store(path=path, mean_file=path_mean, backend=backend,
              random_skip=random, batchsize=batchsize,
              shape=shape) 

  data = Data(load='recordinput', conf=store)

  return data

