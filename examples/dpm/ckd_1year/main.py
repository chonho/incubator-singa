#!/usr/bin/env python
from datetime import datetime
import os, sys, re, time
import csv

posix_now = time.time()
d = datetime.fromtimestamp(posix_now)

emr = {}
codeid = {}


#---------------------------
def read_emr():
  for key, val in csv.reader(open("emr.csv")):
    emr[key] = val

def write_emr():
  w = csv.writer(open("emr.csv", "w"))
  for key, val in emr.items():
    w.writerow([key, val])

def print_emr():
  for key, val in emr.items():
    print key
    for v in val:
      print v['symbol'], v['dtime'], v['code'], v['labtest']

#---------------------------
def read_codeid():
  for key, val in csv.reader(open("codeid.csv")):
    codeid[key] = int(val)

def write_codeid():
  w = csv.writer(open("codeid.csv", "w"))
  for key, val in sorted(codeid.items(), key=lambda x:x[1]):
    w.writerow([key, val])

def print_codeid():
  for key, val in codeid.items():
    print key, val
#---------------------------

def getCodeNameByID(cid):
  return codeid.keys()[codeid.values().index(cid)]

def countCode():
  return sum(1 for i in codeid.keys())

def countPatients():
  return sum(1 for i in emr.keys())


DEMO = ('PTDOBMM','PTDOBYY','USERDATE','PTEDUCAT','PTGENDER')

patientDict = {}
demoDict = {}
dynamicDict = {}

def is_numeric(strnum):
  strnum = strnum.replace('.','')
  strnum = strnum.replace('-','')
  strnum = strnum.replace('e+','')
  return strnum.isdigit()

def generate_codeid():

  f = open('test_remove_empty.csv', 'r')
  reader = csv.reader(f)
  header = next(reader)

  for line in reader:
    if not line[2] in DEMO:
      if not line[2] in codeid:
        codeid[line[2]] = len(codeid)

fvalDict = {}
featureDict = {}
labelDict = {} 

def read_dataset(op):
  '''
  read data
  '''
  #wf = csv.writer(open("dynamic.csv", "w"))
  #wf = open("dynamic.csv", "w")
  #wf1 = open("dpm_test"+str(arg2), "w")
  #wf2 = open("dpm_train"+str(arg2), "w")
  wf1 = open("dpm_test", "w")
  wf2 = open("dpm_train", "w")
  rf = open('test_remove_empty.csv', 'r')
  reader = csv.reader(rf)
  header = next(reader)


  if op == 1:

    pid = 2
    et = 'bl' 
    et_prev = 'bl' 
    f_cnt = 0

    existMMSCORE = 0

    for line in reader:

      if 'MMSCORE' in line[2]: 
        existMMSCORE = 1
        #print int(line[0]), line[1], line[2], line[3]

      if pid != int(line[0]):
        field = {
           'age' : int(demoDict[DEMO[2]][:4]) - int(demoDict[DEMO[1]]),
           'education': demoDict[DEMO[3]],
           'gender': demoDict[DEMO[4]],
           'f_cnt': f_cnt 
        }
        patientDict[pid] = field 
        f_cnt = 0
        pid = int(line[0])
  
      if 'None' in line or is_numeric(line[3]) == False:
        # TODO take care of missing values or categorical value
        continue
   
      elif line[1] == 'sc' and line[2] in DEMO:
        # store demographic information
        demoDict[line[2]] = line[3]
  
      else:
        if "Willebrand Factor" in line[2]:
          line[3] = line[3][:-1] # remove ^M

        if is_numeric(line[3]) and not line[2] in DEMO:
          cid = codeid[line[2]]
          if not cid in fvalDict:
            fvalDict[cid] = [float(line[3])]
          else:
            fvalDict[cid].append(float(line[3]))


      # time changes
      if not et == line[1]:
        
        if not (et in ['sc', 'nv', 'f', '']): 
          if gettime(et) >= arg1:
            if existMMSCORE == 1:
              dt = gettime(et)-gettime(et_prev)
              if not pid in labelDict.keys():
                labelDict[pid] = [dt]
              else:
                labelDict[pid].append(dt)
          else:
            f_cnt += 1
            et_prev = et

        pid = int(line[0])
        et  = line[1]
  
        existMMSCORE = 0

    # compute mu, sigma for features   
    for key, val in fvalDict.items():
      mu, sigma, nval = normalize(val)
      fvalDict[key] = (mu, sigma, nval)


  # write dynamic dataset
  if op == 2:
 
    pid = 2
    et = 'bl' 
    et_prev = 'bl' 
    dt = 0
    f_idx = []
    f_val = []
    out = ''
    out2 = ''
    nb_sample = 0

    for line in reader:

      if pid == int(line[0]):

        if not et == line[1]:
          featureDict['idx'] = f_idx 
          featureDict['val'] = f_val 
          dynamicDict[(pid, et)] = featureDict
 
          if not (et in ['sc', 'nv', 'f', '']): 
            dt = []
            if pid in labelDict.keys():
              dt = labelDict[pid]
              #dt = len(labelDict[pid])
            if not gettime(et_prev) == -1:
              if gettime(et) < arg1:
                if nb_sample in dataRange:
                  out = out + prepare_dynamic(pid, gettime(et)-gettime(et_prev), f_idx, f_val, dt)
                else:
                  out2 = out2 + prepare_dynamic(pid, gettime(et)-gettime(et_prev), f_idx, f_val, dt)
              else:
                if 41 in f_idx: # use data with mmscore value
                  if nb_sample in dataRange:
                    out = out + prepare_dynamic(pid, -1, f_idx, f_val, dt)
                  else:
                    out2 = out2 + prepare_dynamic(pid, -1, f_idx, f_val, dt)
                  nb_sample += 1
                '''
                if 41 in f_idx:
                  print pid, i, dt, int(f_val[f_idx.index(41)])
                else:
                  print pid, i, dt, 'none'
                i += 1
                '''
 
          pid = int(line[0])
          et_prev = et
          et  = line[1]

          f_idx = []
          f_val = []

        if line[2] in DEMO:
          continue

        if 'None' in line or is_numeric(line[3]) == False:
          continue # ignore
  
        else:
          cid = codeid[line[2]]
          mu = fvalDict[cid][0]
          sigma = fvalDict[cid][1]

          f_idx.append(cid)
          if gettime(et) >= arg1 and cid == 41:
            f_val.append(float(line[3]))
            continue
          else:
            if sigma != 0:
              f_val.append((float(line[3])-mu)/sigma)
            else:
              f_val.append(mu)

      else:
        # replicate data
        if pid in labelDict.keys(): 
          for i in range(len(labelDict[pid])):
            wf1.write(out)
            wf2.write(out2)
        
        out = ''
        out2 = ''
        pid = int(line[0])
        et = 'bl' 
        et_prev = 'bl'

    print '# of samples ', nb_sample

def gettime(et):
  if et == 'bl':
    return 0
  elif 'm' in et:
    return int(et[1:])
  else:
    return -1

def normalize(values):
  mean = sum(values)/len(values)
  sqrs = [(elmt-mean)**2 for elmt in values]
  import math
  std  = math.sqrt(sum(sqrs)/len(sqrs))
  if not std == 0:
    norm = [(elmt-mean)/std for elmt in values]
  else:
    norm = [(elmt-mean) for elmt in values]
  return mean, std, norm

def write_demo():
  w = csv.writer(open("demo.csv", "w"))
  for key, val in demoDict.items():
    w.writerow([key, val])


def prepare_dynamic(pid, lt, idx, val, dt):
  ret = ''
  ret = ret + '{0}\n'.format( pid )
  if pid in patientDict.keys():
    a = patientDict[pid]['age']
    e = patientDict[pid]['education']
    g = patientDict[pid]['gender']
    c = patientDict[pid]['f_cnt']
    ret = ret + '{0},{1},{2},{3}\n'.format(a, e, g, c)
  else:
    ret = ret + '0\n'
  ret = ret + '{0},{1},'.format(lt, len(dt))
  ret = ret + ','.join([str(i) for i in dt]) + '\n'
  ret = ret + ','.join([str(i) for i in idx]) + '\n'
  ret = ret + ','.join([str(i) for i in val]) + '\n'
  return ret

def datasetIndex( did, ratio ):
  total_sample = 558
  start = (did - 1) * int(total_sample*ratio) 
  end = start + int(total_sample*ratio) - 1
  if end >= total_sample: end = total_sample
  return start, end  

#main
if len(sys.argv) > 3:
  arg1 = int(sys.argv[1]) # cut point
  arg2 = int(sys.argv[2]) # index (range id) of test data
  arg3 = float(sys.argv[3]) # % of test data
else:
  arg1 = 28 
  arg2 = 1
  arg3 = 0.1 

s, e = datasetIndex(arg2, arg3)
dataRange = range(s, e+1)

generate_codeid()
write_codeid()
print '# of codeid: ', countCode()
print 'MMSCORE: ', codeid['MMSCORE']


#for key, val in codeid.items(): print key, val
read_dataset(1)
#for key, val in fvalDict.items(): print key, val[0], val[1]
#for key, val in labelDict.items(): print key, val
read_dataset(2)
#for key, val in dynamicDict.items(): print key, val
#for key, val in patientDict.items(): print key, val
 
#write_demo()
#write_dynamic()


#print '# of patients: ', countPatients()
#print '# of codeid: ', countCode()

