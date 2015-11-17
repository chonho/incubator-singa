#!/usr/bin/env python
import os, sys, re, subprocess
from Utility import *
from MessageClass import *
from job_pb2 import *
from google.protobuf import text_format


#TODO auto-generate it in MessageClass.py ???
def enumInitMethod(key):
  if key == 'uniform': return kUniform
  if key == 'const': return kConstant
  if key == 'gauss': return kGaussian
  return ''

def enumChangeMethod(key):
  if key == 'fixed': return kFixed
  if key == 'inverse': return kInverse
  if key == 'exponential': return kExponential
  if key == 'linear': return kLinear
  if key == 'step': return kStep
  if key == 'fixedstep': return kFixedStep
  return ''
#------------------

class Model(object):

  def __init__(self, name='my model', label=False):
    self.jobconf = JobProto() 
    self.jobconf.name = name 
    self.layers = []
    self.label = label
    
  def add(self, layer):
    self.layers.append(layer)

  def compile(self, updater, cluster):
    setval(self.jobconf, updater=updater.proto)
    setval(self.jobconf, cluster=cluster.proto)
    self.build()

  def build(self):
    net = NetProto() 
    slyname = self.layers[0].layer.name
    for i in range(len(self.layers)):
      ly = net.layer.add()
      ly.CopyFrom(self.layers[i].layer)
      lastly = ly
      if i > 0:
        getattr(ly, 'srclayers').append(slyname)
        slyname = ly.name
      if hasattr(self.layers[i], 'mask'):
        mly = net.layer.add()
        mly.CopyFrom(self.layers[i].mask)
        getattr(mly, 'srclayers').append(slyname)
        slyname = mly.name
        lastly = mly

    if self.label == True:
      label_layer = Layer(name='label', type=kLabel)      
      ly = net.layer.add()
      ly.CopyFrom(label_layer.layer)
      getattr(ly, 'srclayers').append(self.layers[0].layer.name)
      getattr(lastly, 'srclayers').append(label_layer.layer.name)

    setval(self.jobconf, neuralnet=net)

  def evaluate(self, algorithm, train_steps=1000, **kargs):
    setval(self.jobconf, train_one_batch=algorithm.proto)
    setval(self.jobconf, train_steps=train_steps)
    setval(self.jobconf, **kargs)

    
  def display(self):
    print text_format.MessageToString(self.jobconf)



class Store(object):
  def __init__(self, **kwargs):
    self.proto = StoreProto()
    setval(self.proto, **kwargs)

class Algorithm(object):
  def __init__(self, type=kBP, **kwargs):
    alg = Message('Alg', alg=type, **kwargs).proto
    if type == kCD:
      setval(alg.cd_conf, **kwargs)
    self.proto = alg

class SGD(object):
  def __init__(self, lr=0.01, lr_type=None, lr_conf=None,
               step=[0], step_lr=[0.01],
               **kwargs):
    upd = Message('Updater', type=kSGD, **kwargs).proto

    assert lr_type != None, 'learning rate type should be specified'

    setval(upd.learning_rate, base_lr=lr) 
    if lr_type == 'step':
      cp = Message('Step', change_freq=60, gamma=0.997)
      setval(upd.learning_rate, type=kStep, step_conf=cp.proto) 
    elif lr_type == 'fixed':
      cp = Message('FixedStep', step=step, step_lr=step_lr)
      setval(upd.learning_rate, type=kFixedStep, fixedstep_conf=cp.proto) 
    elif lr_type == 'linear':
      cp = Message('Linear', change_freq=10, final_lr=0.1)
      setval(upd.learning_rate, type=kLinear, linear_conf=cp.proto) 
    self.proto = upd


class Cluster(object):
  def __init__(self, workspace=None, **kwargs):
    assert workspace != None, 'need to set workspace'
    self.proto = ClusterProto()
    self.proto.workspace = workspace 
    # default value
    self.proto.nworker_groups = 1
    self.proto.nserver_groups = 1
    self.proto.nworkers_per_group = 1
    self.proto.nservers_per_procs = 1
    setval(self.proto, **kwargs)

class Param(object):
  def __init__(self, **kwargs):
    self.param = ParamProto()
    if not 'name' in kwargs:
      setval(self.param, name=generateName('param', 1))
    if 'init' in kwargs:
      pg = Message('ParamGen', type=enumInitMethod(kwargs['init']))
      setval(self.param, init=pg.proto)
      del kwargs['init']
    if 'param_init' in kwargs:
      setval(self.param, init=kwargs['param_init'].proto)
      del kwargs['param_init']
    setval(self.param, **kwargs) 
    setval(self.param.init, **kwargs) 

  def setval(self, **kwargs):
    setval(self.param, **kwargs) 
    return self

class Layer(object):
  def __init__(self, **kwargs):
    self.layer = Message('Layer', **kwargs).proto
    # required
    if not 'name' in kwargs:
      setval(self.layer, name=generateName('layer', 1))
    # optional
    # srclayers are set in Model.build()

class Convolution(Layer):
  def __init__(self, num_filters, kernel, stride, pad, 
               w_param=None, b_param=None, activation=None):
    super(Convolution, self).__init__(name=generateName('conv',1), type=kCConvolution)
    self.layer.convolution_conf.num_filters = num_filters
    self.layer.convolution_conf.kernel = kernel 
    self.layer.convolution_conf.stride = stride 
    self.layer.convolution_conf.pad = pad 

    # param w  
    assert w_param != None, 'weight param should be specified'
    setval(w_param.param, name=generateName('w'))
    setval(self.layer, param=w_param.param)

    # param b  
    assert b_param != None, 'bias param should be specified'
    setval(b_param.param, name=generateName('b'))
    setval(self.layer, param=b_param.param)

    # following layers: e.g., activation, dropout, etc.
    if activation:
      self.mask = Activation(activation=activation).layer

class Pooling(Layer):
  def __init__(self, method, kernel=3, stride=2): 
    super(Pooling, self).__init__(name=generateName('pool'), type=kCPooling)
    if method == 'max':
      self.layer.pooling_conf.pool = PoolingProto().MAX 
    if method == 'avg':
      self.layer.pooling_conf.pool = PoolingProto().AVG 
    setval(self.layer.pooling_conf, kernel=kernel)
    setval(self.layer.pooling_conf, stride=stride)

class Norm(Layer):
  def __init__(self, size, **kwargs): 
    super(Norm, self).__init__(name=generateName('norm'), type=kLRN)
    # required
    self.layer.lrn_conf.local_size = size
    setval(self.layer.lrn_conf, **kwargs)

class Dense(Layer):
  def __init__(self, output_dim, activation=None, 
               init='uniform', w_param=None, b_param=None):
    super(Dense, self).__init__(type=kInnerProduct)
    self.layer.innerproduct_conf.num_output = output_dim   # required
    
    pg = Message('ParamGen', type=enumInitMethod(init))
    
    # param w  
    if w_param == None:
      w_param = Param(name=generateName('w'), init=pg)
    else:
      setval(w_param.param, name=generateName('w'))
      setval(w_param.param, init=pg.proto)
    setval(self.layer, param=w_param.param)

    # param b  
    if b_param == None:
      b_param = Param(name=generateName('b'), init=pg)
    else:
      setval(b_param.param, name=generateName('b'))
      setval(b_param.param, init=pg.proto)
    setval(self.layer, param=b_param.param)

    # following layers: e.g., activation, dropout, etc.
    if activation:
      self.mask = Activation(activation=activation).layer

class Activation(Layer):
  def __init__(self, activation, topk=1):
    self.name = activation 
    self.layer_type = kSTanh  
    if self.name == 'tanh':
      self.layer_type = kSTanh  
    if self.name == 'sigmoid':
      self.layer_type = kSigmoid  
    if self.name == 'softmax':
      self.layer_type = kSoftmaxLoss  
    if self.name == 'relu':
      self.layer_type = kReLU  
    super(Activation, self).__init__(name=generateName(self.name), type=self.layer_type)
    if self.name == 'softmax':
      self.layer.softmaxloss_conf.topk = topk 

class Dropout(Layer): 
  def __init__(self, ratio):
    self.name = 'dropout'
    self.layer_type = kDropout
    super(Dropout, self).__init__(name=generateName(self.name), type=self.layer_type)
    self.layer.dropout_conf.dropout_ratio = ratio

class Data(Layer):
  def __init__(self, load, train=True, conf=None, **kwargs):
    self.name = 'data'
    assert load != None, 'data type should be specified'
    if load == 'record':
      self.layer_type = kRecordInput
    if load == 'csv':
      self.layer_type = kCSVInput
    if load == 'shard':
      self.layer_type = kShardData
    super(Data, self).__init__(name=generateName(self.name), type=self.layer_type)
    setval(self.layer, exclude=kTest if train else kTrain)

    if conf == None:
      setval(self.layer.store_conf, **kwargs)
    else:
      setval(self.layer, store_conf=conf.proto)


class RGB(Layer):
  def __init__(self, meanfile=None, **kwargs):
    assert meanfile != None, 'meanfile should be specified'
    self.name = 'rgb'
    self.layer_type = kRGBImage
    super(RGB, self).__init__(name=generateName(self.name), type=self.layer_type)
    self.layer.rgbimage_conf.meanfile = meanfile
   

#TODO run singa training/testing via a wrapper for Driver
'''
def SingaRun():
  conf = '../../examples/mnist/job.conf'
  cmd = '../../bin/singa-run.sh ' \
      + '-conf %s ' % conf 
  print 'cmd:' + cmd
  subprocess.Popen(cmd)
'''

