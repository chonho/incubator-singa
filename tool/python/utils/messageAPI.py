#!/usr/bin/env python 
from utility import * 
from job_pb2 import * 

class Message(object): 
  def __init__(self, protoname, **kwargs): 
    if protoname == 'Job': 
      self.proto = JobProto()
    if protoname == 'Alg': 
      self.proto = AlgProto()
    if protoname == 'Net': 
      self.proto = NetProto()
    if protoname == 'Updater': 
      self.proto = UpdaterProto()
    if protoname == 'Cluster': 
      self.proto = ClusterProto()
    if protoname == 'CD': 
      self.proto = CDProto()
    if protoname == 'Layer': 
      self.proto = LayerProto()
    if protoname == 'Param': 
      self.proto = ParamProto()
    if protoname == 'LRGen': 
      self.proto = LRGenProto()
    if protoname == 'ParamGen': 
      self.proto = ParamGenProto()
    if protoname == 'RGBImage': 
      self.proto = RGBImageProto()
    if protoname == 'Prefetch': 
      self.proto = PrefetchProto()
    if protoname == 'Split': 
      self.proto = SplitProto()
    if protoname == 'Store': 
      self.proto = StoreProto()
    if protoname == 'SoftmaxLoss': 
      self.proto = SoftmaxLossProto()
    if protoname == 'ArgSort': 
      self.proto = ArgSortProto()
    if protoname == 'Convolution': 
      self.proto = ConvolutionProto()
    if protoname == 'Concate': 
      self.proto = ConcateProto()
    if protoname == 'Data': 
      self.proto = DataProto()
    if protoname == 'Mnist': 
      self.proto = MnistProto()
    if protoname == 'Dropout': 
      self.proto = DropoutProto()
    if protoname == 'RBM': 
      self.proto = RBMProto()
    if protoname == 'InnerProduct': 
      self.proto = InnerProductProto()
    if protoname == 'LRN': 
      self.proto = LRNProto()
    if protoname == 'Pooling': 
      self.proto = PoolingProto()
    if protoname == 'Slice': 
      self.proto = SliceProto()
    if protoname == 'ReLU': 
      self.proto = ReLUProto()
    if protoname == 'RMSProp': 
      self.proto = RMSPropProto()
    if protoname == 'FixedStep': 
      self.proto = FixedStepProto()
    if protoname == 'Step': 
      self.proto = StepProto()
    if protoname == 'Linear': 
      self.proto = LinearProto()
    if protoname == 'Exponential': 
      self.proto = ExponentialProto()
    if protoname == 'InverseT': 
      self.proto = InverseTProto()
    if protoname == 'Inverse': 
      self.proto = InverseProto()
    if protoname == 'Uniform': 
      self.proto = UniformProto()
    if protoname == 'Gaussian': 
      self.proto = GaussianProto()
    setval(self.proto, **kwargs)

def enumParamType(key): 
  if key == 'param': return kParam 
  if key == 'user': return kUser 
  return '' 

def enumAlgType(key): 
  if key == 'bp': return kBP 
  if key == 'cd': return kCD 
  if key == 'useralg': return kUserAlg 
  return '' 

def enumInitMethod(key): 
  if key == 'constant': return kConstant 
  if key == 'gaussian': return kGaussian 
  if key == 'uniform': return kUniform 
  if key == 'gaussiansqrtfanin': return kGaussianSqrtFanIn 
  if key == 'uniformsqrtfanin': return kUniformSqrtFanIn 
  if key == 'uniformsqrtfaninout': return kUniformSqrtFanInOut 
  if key == 'userinit': return kUserInit 
  return '' 

def enumUpdaterType(key): 
  if key == 'sgd': return kSGD 
  if key == 'adagrad': return kAdaGrad 
  if key == 'rmsprop': return kRMSProp 
  if key == 'nesterov': return kNesterov 
  if key == 'userupdater': return kUserUpdater 
  return '' 

def enumChangeMethod(key): 
  if key == 'fixed': return kFixed 
  if key == 'inverset': return kInverseT 
  if key == 'inverse': return kInverse 
  if key == 'exponential': return kExponential 
  if key == 'linear': return kLinear 
  if key == 'step': return kStep 
  if key == 'fixedstep': return kFixedStep 
  if key == 'userchange': return kUserChange 
  return '' 

def enumLayerType(key): 
  if key == 'recordinput': return kRecordInput 
  if key == 'csvinput': return kCSVInput 
  if key == 'csvoutput': return kCSVOutput 
  if key == 'recordoutput': return kRecordOutput 
  if key == 'imagepreprocess': return kImagePreprocess 
  if key == 'prefetch': return kPrefetch 
  if key == 'lmdbdata': return kLMDBData 
  if key == 'sharddata': return kShardData 
  if key == 'label': return kLabel 
  if key == 'mnist': return kMnist 
  if key == 'rgbimage': return kRGBImage 
  if key == 'argsort': return kArgSort 
  if key == 'convolution': return kConvolution 
  if key == 'cconvolution': return kCConvolution 
  if key == 'cpooling': return kCPooling 
  if key == 'dropout': return kDropout 
  if key == 'innerproduct': return kInnerProduct 
  if key == 'lrn': return kLRN 
  if key == 'pooling': return kPooling 
  if key == 'relu': return kReLU 
  if key == 'rbmvis': return kRBMVis 
  if key == 'rbmhid': return kRBMHid 
  if key == 'sigmoid': return kSigmoid 
  if key == 'stanh': return kSTanh 
  if key == 'softmax': return kSoftmax 
  if key == 'softmaxloss': return kSoftmaxLoss 
  if key == 'euclideanloss': return kEuclideanLoss 
  if key == 'bridgedst': return kBridgeDst 
  if key == 'bridgesrc': return kBridgeSrc 
  if key == 'concate': return kConcate 
  if key == 'slice': return kSlice 
  if key == 'split': return kSplit 
  if key == 'userlayer': return kUserLayer 
  return '' 

def enumPhase(key): 
  if key == 'unknown': return kUnknown 
  if key == 'train': return kTrain 
  if key == 'val': return kVal 
  if key == 'test': return kTest 
  if key == 'positive': return kPositive 
  if key == 'negative': return kNegative 
  if key == 'forward': return kForward 
  if key == 'backward': return kBackward 
  if key == 'loss': return kLoss 
  if key == 'deploy': return kDeploy 
  return '' 

def enumPoolMethod(key): 
  if key == 'ax': return MAX 
  if key == 'vg': return AVG 
  return '' 

def enumPartitionType(key): 
  if key == 'datapartition': return kDataPartition 
  if key == 'layerpartition': return kLayerPartition 
  if key == 'none': return kNone 
  return '' 

