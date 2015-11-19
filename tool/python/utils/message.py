#!/usr/bin/env python 
import sys, os 
from utility import * 
sys.path.append(os.path.join(os.path.dirname(__file__),'../../pb2')) 
import job_pb2  

class Message(object):
  def __init__(self,protoname,**kwargs):
    if hasattr(job_pb2,protoname+"Proto"):
      class_ = getattr(job_pb2,protoname+"Proto")
      self.proto = class_()
    else:
      raise Exception('invalid protoname')
    setval(self.proto, **kwargs)

enumDict_=dict()
for enumtype in job_pb2.DESCRIPTOR.enum_types_by_name:
  tempDict=enumDict_[enumtype]=dict()
  for name in getattr(job_pb2,enumtype).DESCRIPTOR.values_by_name: 
    tempDict[name[1:].lower()]=getattr(job_pb2,name)

current_module = sys.modules[__name__]

def make_function(enumtype):
  def _function(key):
    return enumDict_[enumtype][key]
  return _function

for enumtype in job_pb2.DESCRIPTOR.enum_types_by_name:
  setattr(current_module,"enum"+enumtype,make_function(enumtype))

