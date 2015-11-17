from Utility import * 
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

