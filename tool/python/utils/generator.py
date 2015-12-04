#!/usr/bin/env python
import os, sys, re
from google.protobuf import text_format

COMPONENTS = ['neuralnet', 'train_one_batch', 'updater', 'cluster']

type_primitive = ['int32', 'string', 'float']

msgList = []
protoDic = {}
enumList = []
enumDic = {}

def generateMessageAPI():
  out = ''
  with open("messageAPI.py", "w") as f:

    out = '#!/usr/bin/env python \n'
    out += 'import sys, os \n'
    out += 'from utility import * \n\n'
    out += 'sys.path.append(os.path.join(os.path.dirname(__file__),\'../../pb2\')) \n'
    out += 'from job_pb2 import * \n\n'

    out += 'class Message(object): \n' 
    out += '  def __init__(self, protoname, **kwargs): \n'  

    for proto in msgList:
      name = proto[:-5]
      out += '    if protoname == \'%s\': \n' % name 
      out += '      self.proto = %s()\n' % proto  

      #out += 'class %s(object):\n' % name  
      #out += '  def __init__(self, **kwargs):\n'  
      #out += '    self.proto = %s()\n' % proto  
      #out += '    setval(self.proto, **kwargs)\n\n'

    out += '    setval(self.proto, **kwargs)\n\n'

    f.write(out)
  
def generateEnumMethod():
  out = ''
  with open("messageAPI.py", "a") as f:

    for name, vals in enumDic.items():
      out += 'def enum%s(key): \n' % name 
      for val in vals:
        key = val[1:].lower()
        out += '  if key == \'%s\': return %s \n' % (key, val)
      out += '  return \'\' \n\n'

    f.write(out)

def isPrimitive(typename):
  if typename in type_primitive:
    return True
  return False

def readJobProto():
  lnum = 0
  temp = ''  
  enumKey = '' 
  enumFlag = False
  with open("../../../src/proto/job.proto", "r") as f:
    for line in f:
      lnum += 1      
      words = re.split(r'\s*|\n*', line.strip())

      if words[0] == "//" or words[0] == "}" or words[0] == "":
        enumFlag == False
        continue

      if words[0] == "message":
        #print words[1]
        msgList.append(words[1])
        temp = words[1]
        continue

      if words[0] == "required" or words[0] == "repeated" \
         or words[0] == "optional":
        protoDic.setdefault(temp,{})[words[2]] = words[1]
        continue

      if words[0] == "enum":
        enumList.append(words[1])
        enumDic[words[1]] = [] 
        enumKey = words[1]
        enumFlag = True
        continue
      
      if enumFlag == True:
        enumDic[enumKey].append(words[0])

if __name__ == "__main__":

  # read job.proto, store field and value in protoDic
  readJobProto()
  # generate api for message
  generateMessageAPI()
  generateEnumMethod()

  #for comp in COMPONENTS:

#  for k, v in protoDic['JobProto'].items():
#    print k, v
  
  #print isPrimitive(protoDic['LRNProto']['alpha'])
  #print isPrimitive(protoDic['NetProto']['layer'])
  #generateAPI('LRNProto')

