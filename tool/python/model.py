#!/usr/bin/env python
import os, sys, re, subprocess
from utils.utility import * 
from utils.message import * 
from google.protobuf import text_format
sys.path.append(os.path.join(os.path.dirname(__file__), '../pb2'))
from job_pb2 import *

class Model(object):

  def __init__(self, name='my model', label=False):
    self.jobconf = Message('Job', name=name).proto 
    self.layers = []
    self.label = label
    
  def add(self, layer):
    self.layers.append(layer)

  def compile(self, optimizer=None, cluster=None, loss='categorical_crossentropy', topk=1):
    '''
    required
      optimizer = (Updater) // updater settings, e.g., SGD
      cluster   = (Cluster) // cluster settings
    optional
      loss      = (string)  // name of loss function type
      topk      = (int)     // the number of results considered to compute accuracy
    '''
    assert optimizer != None, 'optimizer (Updater) should be set'
    assert cluster != None, 'cluster (Cluster) should be set'  
    setval(self.jobconf, updater=optimizer.proto)
    setval(self.jobconf, cluster=cluster.proto)

    # take care of loss function layer
    assert loss != None, 'loss should be set'
    if hasattr(self.layers[len(self.layers)-1], 'mask'):
      ly = self.layers[len(self.layers)-1].mask
    else:
      ly = self.layers[len(self.layers)-1].layer

    if ly.type == enumLayerType('softmax'):
      # revise the last layer
      if loss == 'categorical_crossentropy':
        setval(ly, type=enumLayerType('softmaxloss'))
        setval(ly.softmaxloss_conf, topk=topk) 
      elif loss == 'mean_squared_error':
        setval(ly, type=enumLayerType('euclideanloss'))
    else:
      # add new layer
      if loss == 'categorical_crossentropy':
        self.add(Activation('softmaxloss', topk=topk))
      elif loss == 'mean_squared_error':
        self.add(Activation('euclideanloss'))


  def build(self):
    net = NetProto() 
    slyname = self.layers[0].layer.name
    for i in range(len(self.layers)):
      ly = net.layer.add()
      ly.CopyFrom(self.layers[i].layer)
      lastly = ly
      if self.layers[i].is_datalayer == True:
        continue
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
    else:
      getattr(lastly, 'srclayers').append(self.layers[0].layer.name)

    setval(self.jobconf, neuralnet=net)

  def fit(self, **kwargs):
    pass

  def evaluate(self, **kwargs):
    pass
    
  def display(self):
    print text_format.MessageToString(self.jobconf)


class Sequential(Model):
  def __init__(self, name='my model', label=False):
    super(Sequential, self).__init__(name=name, label=label)

  def exist_datalayer(self, phase):
    if self.layers[0].layer.include != enumPhase(phase): # train
      if self.layers[1].layer.include != enumPhase(phase): # test
        if self.layers[2].layer.include != enumPhase(phase): # valid
          return False
    return True
    
  def fit(self, data=None, train_steps=0, **fields):
    '''
    required
      data        = (Data)  // Data class object for training data
      train_steps = (int)   // the number of training, i.e., epoch
    optional
      **fields (KEY=VALUE)
        batch_size       = (int)    // batch size for training data
        disp_freq        = (int)    // frequency to display training info
        disp_after       = (int)    // display after this number 
        validate_data    = (Data)   // validation data, specified in load_data()
        validate_freq    = (int)    // frequency of validation
        validate_steps   = (int)    // total number of steps for validation
        validate_after   = (int)    // start validation after this number
        checkpoint_path  = (string) // path to checkpoint file
        checkpoint_freq  = (int)    // frequency for checkpoint
        checkpoint_after = (int)    // start checkpointing after this number
    '''
    assert data != None, 'Training data shold be set'
    assert train_steps != 0, 'Training steps shold be set'

    if 'batch_size' in fields:  # if new value is set, replace the batch_size
      setval(data.layer.store_conf, batchsize=fields['batch_size'])
   
    if self.exist_datalayer('train') == False: 
      self.layers.insert(0, data)

    if 'validate_data' in fields:
      self.layers.insert(1, fields['validate_data'])

    setval(self.jobconf, train_steps=train_steps)
    setval(self.jobconf, **fields)
    
    # set Train_one_batch component, using backprogapation
    setval(self.jobconf, train_one_batch=Algorithm(type=enumAlgType('bp')).proto)

  def evaluate(self, data=None, test_steps=10, **fields):
    '''
    required
      data       = (Data)  // Data class object for testing data
    optional
      test_steps = (int)   // the number of testing
      **fields (KEY=VALUE)
        batch_size   = (int)  // batch size for testing data
        test_freq    = (int)  // frequency of testing
        test_steps   = (int)  // total number of steps for testing 
        test_after   = (int)  // start testing after this number of steps 
    '''
    assert data != None, 'Testing data shold be set'

    if 'batch_size' in fields:  # if new value is set, replace the batch_size
      setval(data.layer.store_conf, batchsize=fields['batch_size'])

    if self.exist_datalayer('test') == False: 
      self.layers.insert(1, data)
    
    setval(self.jobconf, test_steps=test_steps)
    setval(self.jobconf, **fields)

    self.build()  # construct Nneuralnet Component

    with open('job.conf', 'w') as f:
      f.write(text_format.MessageToString(self.jobconf))

    #self.display()
    self.result = SingaRun()
    return self.result

class Store(object):
  def __init__(self, **kwargs):
    '''
    **kwargs
        path       = (string)  // path to dataset
        backend    = (string)  // 
        batch_size = (int)     // batch size of dataset
        shape      = (int)     // 

    '''
    self.proto = Message('Store', **kwargs).proto

class Algorithm(object):
  def __init__(self, type=enumAlgType('bp'), **kwargs):
    alg = Message('Alg', alg=type, **kwargs).proto
    if type == enumAlgType('cd'):
      setval(alg.cd_conf, **kwargs)
    self.proto = alg

class SGD(object):
  def __init__(self, lr=0.01, lr_type=None,
               decay=0, momentum=0,
               step=(0), step_lr=(0.01), **fields):
    '''
    required
      lr      = (float)  // base learning rate
      lr_type = (string) // type of learning rate
    optional
      **fields (KEY=VALUE)
        decay    = (float) // weight decay
        momentum = (float) // momentum

    '''
    assert lr
    assert lr_type != None, 'type of learning rate should be specified'

    upd = Message('Updater', type=kSGD, **fields).proto
    setval(upd.learning_rate, base_lr=lr) 
    if decay > 0:
      setval(upd, weight_decay=decay) 
    if momentum > 0:
      setval(upd, momentum=momentum) 

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
  def __init__(self, workspace=None,
               nworker_groups=1, nserver_groups=1,
               nworkers_per_group=1, nservers_per_group=1,
               nworkers_per_procs=1, nservers_per_procs=1,
               **fields):
    '''
    required
      workspace = (string) // workspace path
    optional
      nworker_groups     = (int)
      nserver_groups     = (int)
      nworkers_per_group = (int)
      nservers_per_group = (int)
      nworkers_per_procs = (int)
      nservers_per_procs = (int)
      **fields
        server_worker_separate = (bool)
    '''
    assert workspace != None, 'need to set workspace'
    self.proto = Message('Cluster', workspace=workspace).proto
    # optional
    self.proto.nworker_groups = nworker_groups 
    self.proto.nserver_groups = nserver_groups 
    self.proto.nworkers_per_group = nworkers_per_group 
    self.proto.nservers_per_group = nservers_per_group 
    self.proto.nworkers_per_procs = nworkers_per_procs 
    self.proto.nservers_per_procs = nservers_per_procs 
    # other fields
    setval(self.proto, **fields)

#TODO make this internally used
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

    # srclayers are set in Model.build()
    self.is_datalayer = False 

class Data(Layer):
  def __init__(self, load, phase='train', conf=None, **kwargs):
    assert load != None, 'data type should be specified'
    self.layer_type = enumLayerType(load)
    super(Data, self).__init__(name=generateName('data'), type=self.layer_type)

    # include/exclude
    setval(self.layer, include=enumPhase(phase))
    #setval(self.layer, exclude=kTest if phase=='train' else kTrain)

    if conf == None:
      setval(self.layer.store_conf, **kwargs)
    else:
      setval(self.layer, store_conf=conf.proto)
    self.is_datalayer = True

class Convolution2D(Layer):
  def __init__(self, nb_filter=0, kernel=0, stride=1, pad=0,
               w_param=None, b_param=None, activation=None, **kwargs):
    '''
    required
      nb_filter = (int)  // the number of filters
      kernel    = (int)  // the size of filter
    optional
      stride    = (int)  // the size of stride
      pad       = (int)  // the size of padding
    '''
    assert nb_filter > 0 and kernel > 0, 'should be set as positive int'
    super(Convolution2D, self).__init__(name=generateName('conv',1), type=kCConvolution)
    fields = {'num_filters' : nb_filter,
              'kernel' : kernel,
              'stride' : stride,
              'pad' : pad}
    setval(self.layer.convolution_conf, **fields)

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

class MaxPooling2D(Layer):
  def __init__(self, pool_size=None, stride=1, ignore_border=True, **kwargs): 
    '''
    required
      pool_size     = (int|tuple) // the size for pooling
    optional
      stride        = (int)       // the size of striding
      ignore_border = (bool)      // flag for padding
      **kwargs                    // fields for Layer class
    '''
    assert pool_size != None, 'pool_size is required'
    if type(pool_size) == int:
      pool_size = (pool_size, pool_size)
    assert type(pool_size) == tuple and  \
           pool_size[0] == pool_size[1], 'pool size should be square in Singa'
    super(MaxPooling2D, self).__init__(name=generateName('pool'), type=kCPooling, **kwargs)
    fields = {'pool' : PoolingProto().MAX,
              'kernel' : pool_size[0],
              'stride' : stride,
              'pad' : 0 if ignore_border else 1}
    setval(self.layer.pooling_conf, **fields)

class AvgPooling2D(Layer):
  def __init__(self, pool_size=None, stride=1, ignore_border=True, **kwargs): 
    '''
    required
      pool_size     = (int|tuple) // size for pooling
    optional
      stride        = (int)       // size of striding
      ignore_border = (bool)      // flag for padding
      **kwargs                    // fields for Layer class
    '''
    assert pool_size != None, 'pool_size is required'
    if type(pool_size) == int:
      pool_size = (pool_size, pool_size)
    assert type(pool_size) == tuple and  \
           pool_size[0] == pool_size[1], 'pool size should be square in Singa'
    super(AvgPooling2D, self).__init__(name=generateName('pool'), type=kCPooling, **kwargs)
    self.layer.pooling_conf.pool = PoolingProto().AVG 
    fields = {'pool' : PoolingProto().AVG,
              'kernel' : pool_size[0],
              'stride' : stride,
              'pad' : 0 if ignore_border else 1}
    setval(self.layer.pooling_conf, **fields)

class LRN2D(Layer):
  def __init__(self, size=0, alpha=1e-4, k=1, beta=0.75, **kwargs):
    super(LRN2D, self).__init__(name=generateName('norm'), type=kLRN)
    # required
    assert size != 0, 'local size should be set'
    self.layer.lrn_conf.local_size = size 
    setval(self.layer.lrn_conf, alpha=alpha, knorm=k, beta=beta, **kwargs)


class Dense(Layer):
  def __init__(self, output_dim=0, activation=None, 
               init='uniform', w_param=None, b_param=None, input_dim=None,
               **kwargs):
    '''
    required
      output_dim = (int)
    optional
      activation = (string)

    '''
    assert output_dim > 0, 'output_dim should be set'
    super(Dense, self).__init__(type=kInnerProduct, **kwargs)
    self.layer.innerproduct_conf.num_output = output_dim   # required
    
    pg = Message('ParamGen', type=enumInitMethod(init))
    
    # param w  
    if w_param == None:
      w_param = Param(name=generateName('w'), init=pg)
    else:
      setval(w_param.param, name=generateName('w'))
    setval(self.layer, param=w_param.param)

    # param b  
    if b_param == None:
      b_param = Param(name=generateName('b'), init=pg)
    else:
      setval(b_param.param, name=generateName('b'))
    setval(self.layer, param=b_param.param)

    # following layers: e.g., activation, dropout, etc.
    if activation:
      self.mask = Activation(activation=activation).layer

class Activation(Layer):
  def __init__(self, activation='stanh', topk=1):
    self.name = activation 
    #TODO better way?
    if activation == 'tanh': activation = 'stanh'
    self.layer_type = enumLayerType(activation)  
    super(Activation, self).__init__(name=generateName(self.name), type=self.layer_type)
    if activation == 'softmaxloss':
      self.layer.softmaxloss_conf.topk = topk

class Dropout(Layer): 
  def __init__(self, ratio=0.5):
    self.name = 'dropout'
    self.layer_type = kDropout
    super(Dropout, self).__init__(name=generateName(self.name), type=self.layer_type)
    self.layer.dropout_conf.dropout_ratio = ratio


class RGB(Layer):
  def __init__(self, meanfile=None, **kwargs):
    assert meanfile != None, 'meanfile should be specified'
    self.name = 'rgb'
    self.layer_type = kRGBImage
    super(RGB, self).__init__(name=generateName(self.name), type=self.layer_type)
    self.layer.rgbimage_conf.meanfile = meanfile
   

#TODO run singa training/testing via a wrapper for Driver
def SingaRun():
  SINGAROOT = '../../'
  conf = 'job.conf'
  cmd = '../../bin/singa-run.sh ' \
      + '-conf %s ' % conf

  procs = subprocess.Popen(cmd.strip().split(' '), stdout = subprocess.PIPE, stderr = subprocess.STDOUT)

  resultDic = {} 
  output = iter(procs.stdout.readline, '')
  for line in output:
    print line[:-1]
    line = re.findall(r'[\w|*.*]+', line)
    if 'accuracy' in line:
      step = line[line.index('step')+1]
      acc  = line[line.index('accuracy')+1]
      loss = line[line.index('loss')+1]
      #print 'Step: ', line[idx_step+1], 'Acc: ', acc, 'Loss: ', loss 
      resultDic.setdefault(step,{})['acc'] = acc 
      resultDic.setdefault(step,{})['loss'] = loss 

  #TODO better format to store??
  return resultDic

