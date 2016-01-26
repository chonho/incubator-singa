/*
 * This file include code from rnnlmlib-0.4 whose licence is as follows:
Copyright (c) 2010-2012 Tomas Mikolov
Copyright (c) 2013 Cantab Research Ltd
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

//
// This code creates DataShard for RNNLM dataset.
// The RNNLM dataset could be downloaded at
//    http://www.rnnlm.org/
//
// Usage:
//    create_shard.bin -train [train_file] -valid [valid_file]
//                     -test [test_file] -class_size [# of classes]

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>

#include "singa/io/store.h"
#include "singa/utils/common.h"
#include "singa/proto/common.pb.h"
#include "./dpm.pb.h"

#define MAX_STRING 100
#define BUFFER_LEN 32
#define NL_STRING  "</s>"

using std::string;
using std::max;
using std::min;

struct vocab_word {
  int cn;
  char word[MAX_STRING];
  int class_index;
};

struct vocab_word *vocab;
int vocab_max_size;
int vocab_size;
int *vocab_hash;
int vocab_hash_size;
int debug_mode;
int old_classes;
int *class_start;
int *class_end;
int class_size;
int cut_point;

char dpm_file[MAX_STRING];
char train_file[MAX_STRING];
char valid_file[MAX_STRING];
char test_file[MAX_STRING];

int valid_mode;
int test_mode;

unsigned int getWordHash(char *word) {
  unsigned int hash, a;

  hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 237 + word[a];
  hash = hash % vocab_hash_size;

  return hash;
}

int searchVocab(char *word) {
  int a;
  unsigned int hash;

  hash = getWordHash(word);

  if (vocab_hash[hash] == -1) return -1;
  if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];

  for (a = 0; a < vocab_size; a++) {   // search in vocabulary
    if (!strcmp(word, vocab[a].word)) {
      vocab_hash[hash] = a;
      return a;
    }
  }

  return -1;   // return OOV if not found
}

int addCharToVocab(char *word) {
  unsigned int hash;

  snprintf(vocab[vocab_size].word, strlen(word)+1, "%s", word);
  vocab[vocab_size].cn = 0;
  vocab_size++;

  if (vocab_size + 2 >= vocab_max_size) {   // reallocate memory if needed
    vocab_max_size += 100;
    vocab = (struct vocab_word *) realloc(
        vocab,
        vocab_max_size * sizeof(struct vocab_word));
  }

  hash = getWordHash(word);
  vocab_hash[hash] = vocab_size - 1;

  return vocab_size - 1;
}

char readChar(char *ch, FILE *fin) {
  int a = 0, c;
  char flag = '0';

  while (!feof(fin)) {
    c = fgetc(fin);

    if (c == 13) continue;

    if ((c == 'N') || (c == 't') || (c == 'c') ||
        (c == ' ') || (c == '\t') || (c == '\n')) {

      if ((c == 'N') || (c == 't') || (c == 'c')) { 
        flag = c;
      }

      if (a > 0) {
        if (c == '\n') ungetc(c, fin);
        if (c == 'c') ungetc(c, fin);  // deal with the last char
        break;
      }

      if (c == '\n') {
        snprintf(ch, strlen(NL_STRING) + 1,
            "%s", const_cast<char *>(NL_STRING));
        return flag;
      } else {
        continue;
      }
    }

    ch[a] = static_cast<char>(c);
    a++;

    if (a >= MAX_STRING) {
      // printf("Too long word found!\n");   //truncate too long words
      a--;
    }
  }
  ch[a] = 0;
  return flag;
}

int readWord(char *word, FILE *fin) {
  int a = 0, ret = 0, ch;

  while (!feof(fin)) {
    ch = fgetc(fin);

    if (ch == 13) continue;

    if ((ch == ',') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') {
          //ungetc(ch, fin);
          ret = 1;
        }
        break;
      }

      if (ch == '\n') {
        return 1;  // 1 for end of line
      } else {
        continue;
      }
    }

    word[a] = static_cast<char>(ch);
    a++;

    if (a >= MAX_STRING) {
      // printf("Too long word found!\n");   //truncate too long words
      a--;
    }
  }
  word[a] = 0;

  return ret;
}

void sortVocab() {
  int a, b, max;
  vocab_word swap;

  for (a = 1; a < vocab_size; a++) {
    max = a;
    for (b = a + 1; b < vocab_size; b++)
      if (vocab[max].cn < vocab[b].cn) max = b;

    swap = vocab[max];
    vocab[max] = vocab[a];
    vocab[a] = swap;
  }
}

int learnVocabFromEmrFile() {
  char ch[MAX_STRING];
  char chflag, wflag;
  FILE *fin;
  int a, i, wcnt;

  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

  fin = fopen(dpm_file, "rb");

  vocab_size = 0;

  addCharToVocab(const_cast<char *>(NL_STRING));

  wcnt = 0;
  wflag = '0';

  while (1) {
    chflag = readChar(ch, fin);  // chflag: N, c, t
    if (chflag == 'c' && chflag != wflag) wcnt++;
    wflag = chflag;

    if (feof(fin)) break;

    i = searchVocab(ch);
    if (i == -1) {
      a = addCharToVocab(ch);
      vocab[a].cn = 1;
    } else {
      vocab[i].cn++;
    }
  }

  sortVocab();

  if (debug_mode > 0) {
    printf("Vocab size (# of chars): %d\n", vocab_size);
    printf("Words in dpm file: %d\n", wcnt);
  }

  fclose(fin);
  return 0;
}

int splitClasses() {
  double df, dd;
  int i, a, b;

  df = 0;
  dd = 0;
  a = 0;
  b = 0;

  class_start = reinterpret_cast<int *>(calloc(class_size, sizeof(int)));
  memset(class_start, 0x7f, sizeof(int) * class_size);
  class_end = reinterpret_cast<int *>(calloc(class_size, sizeof(int)));
  memset(class_end, 0, sizeof(int) * class_size);

  if (old_classes) {    // old classes
    for (i = 0; i < vocab_size; i++)
      b += vocab[i].cn;
    for (i = 0; i < vocab_size; i++) {
      df += vocab[i].cn / static_cast<double>(b);
      if (df > 1) df = 1;
      if (df > (a + 1) / static_cast<double>(class_size)) {
        vocab[i].class_index = a;
        if (a < class_size - 1) a++;
      } else {
        vocab[i].class_index = a;
      }
    }
  } else {            // new classes
    for (i = 0; i < vocab_size; i++)
      b += vocab[i].cn;
    for (i = 0; i < vocab_size; i++)
      dd += sqrt(vocab[i].cn / static_cast<double>(b));
    for (i = 0; i < vocab_size; i++) {
      df += sqrt(vocab[i].cn / static_cast<double>(b)) / dd;
      if (df > 1) df = 1;
      if (df > (a + 1) / static_cast<double>(class_size)) {
        vocab[i].class_index = a;
        if (a < class_size - 1) a++;
      } else {
        vocab[i].class_index = a;
      }
    }
  }

  // after dividing classes, update class start and class end information
  for (i = 0; i < vocab_size; i++)  {
    a = vocab[i].class_index;
    class_start[a] = min(i, class_start[a]);
    class_end[a] = max(i + 1, class_end[a]);
  }
  return 0;
}

int init_class() {
  // debug_mode = 1;
  vocab_max_size = 100;  // largest length value for each word
  vocab_size = 0;
  vocab = (struct vocab_word *) calloc(vocab_max_size,
      sizeof(struct vocab_word));
  vocab_hash_size = 100000000;
  vocab_hash = reinterpret_cast<int *>(calloc(vocab_hash_size, sizeof(int)));
  old_classes = 1;

  // generate vocab list from emr_file
  learnVocabFromEmrFile();

  // split classes
  //splitClasses();

  return 0;
}


int create_data(const char *input_file, const char *output, int ROUND) {

  auto* store = singa::io::OpenStore("kvfile", output, singa::io::kCreate);
  DynamicRecord dynamicRecord;
  OutTimeRecord outtimeRecord;

  FILE *fin;
  int a, i, label;
  fin = fopen(input_file, "rb");

  int line = 0;
  int wflag = 0, rflag = 0, eofflag = 0;
  int reccnt = 0, ltcnt=0, democnt = 0, otimecnt = 0;
  int csv_num_line = 5; // # of lines for a patinet in dynamic.csv


  char key[BUFFER_LEN];
  char chstr[MAX_STRING];
  string value;

  int pid=0, pid_prev=0;
  int age, edu, gen, scnt=0, lt=0, label_len=0, dt=0;
  int nb_data=0, nb_label=0;
  std::vector<int> f_idx;
  std::vector<float> f_val;
  std::vector<int> dt_label;

  while (1) {

    wflag = readWord(chstr, fin);

    if (feof(fin)) {
      dynamicRecord.set_patient_id( -1 );
      int length = snprintf(key, BUFFER_LEN, "%05d", reccnt++);
      dynamicRecord.SerializeToString(&value);
      store->Write(string(key, length), value);
      dynamicRecord.Clear();
      break; 
    }
    
    if (line % csv_num_line == 0) {
      //printf("pid, %d, %s\n", wflag, chstr);
      pid_prev = pid;
      pid = atoi(chstr);
    }
    if (line % csv_num_line == 1) {
      //printf("demo, %d, %s\n", wflag, chstr);
      if (democnt == 0) age = atoi(chstr);
      else if (democnt == 1) edu = atoi(chstr);
      else if (democnt == 2) gen = atoi(chstr);
      else if (democnt == 3) scnt = atoi(chstr);
      if (democnt++ == 3) democnt = 0;
    }
    if (line % csv_num_line == 2) {
      //printf("label, %d, %s\n", pid, chstr);
      if (ltcnt == 0) {
          lt = atoi(chstr);
          if (lt == -1) nb_label++;
      }
      else if (ltcnt == 1) {
          label_len = atoi(chstr);
      }
      else {
          dt_label.push_back(atoi(chstr)); 
      } 
      if (ltcnt++ == 1+label_len) ltcnt = 0;
    }
    if (line % csv_num_line == 3) {
      f_idx.push_back(atoi(chstr));
    }
    if (line % csv_num_line == 4) {
      f_val.push_back(atof(chstr));
      rflag = 1;
    }

    // wflag 1: end of line
    if (wflag == 1) {
      line += 1;
      if (rflag == 1) rflag = 2;
    }

    // for 1 record (tuple)
    if (rflag == 2) {

      // patients with invalid data (need to check)
      if (pid == 285 || pid == 289
          || pid == 344 || pid == 377 || pid == 695
          || pid == 892 || pid == 995 || pid == 1074 
          || pid == 1092 || pid == 1211 
          || pid == 1082  
      ) {

        f_idx.clear();
        f_val.clear();
        dt_label.clear();
        rflag = 0;
        continue;
      }

      if (ROUND == 1 && lt != -1) {

           if ( lt == 0 && reccnt > 0) {
              // additional record for separation
              dynamicRecord.set_patient_id( -1 );
              int length = snprintf(key, BUFFER_LEN, "%05d", reccnt++);
              dynamicRecord.SerializeToString(&value);
              store->Write(string(key, length), value);
              dynamicRecord.Clear();
           }

           int i = 0;
           float fval = 0.0;
      
           //if (reccnt > 1375 && reccnt < 1495)
           //printf("DR: %d %d\n", pid, lt);
      
           dynamicRecord.set_patient_id( pid );
           dynamicRecord.set_lap_time( lt );
           dynamicRecord.set_age( age );
           dynamicRecord.set_education( edu );
           dynamicRecord.set_gender( gen );
           dynamicRecord.set_nb_sample( scnt );
           for(int k=0; k<f_idx.size(); k++) {
              dynamicRecord.add_observed_idx(f_idx.at(k));
              dynamicRecord.add_feature_value(f_val.at(k));
           }

           int length = snprintf(key, BUFFER_LEN, "%05d", reccnt++);
           dynamicRecord.SerializeToString(&value);
           store->Write(string(key, length), value);
      
           dynamicRecord.Clear();
   
      }

      else if (ROUND == 2 && lt == -1) {

           if(nb_data++ < nb_label) {
               float fval = 0.0;
               
               for (int i=0; i<f_idx.size(); i++) {
                 if (f_idx.at(i) == 41) // 41: MMSCORE
                   fval = f_val.at(i);
               }

               //printf("OT %d: %d %d\n", pid, dt_label.size(), nb_label);

               outtimeRecord.set_patient_id( pid );
               outtimeRecord.set_delta_time( dt_label.at(nb_label-1) );
               outtimeRecord.set_mmscore( fval );
       
               int length = snprintf(key, BUFFER_LEN, "%05d", otimecnt++);
               outtimeRecord.SerializeToString(&value);
               store->Write(string(key, length), value);

               outtimeRecord.Clear();
           }
      }
      else if (ROUND == 2 && lt == 0) {
           nb_label = 0;
           if (pid != pid_prev) nb_data = 0;
      }

      f_idx.clear();
      f_val.clear();
      dt_label.clear();
      rflag = 0;
    } 

  }

  fclose(fin);
  store->Flush();
  delete store;
  return 0;
}


int argPos(char *str, int argc, char **argv) {
  int a;

  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a]))
      return a;

  return -1;
}

int main(int argc, char **argv) {
  int i;
  FILE *f;

  // set debug mode
  i = argPos(const_cast<char *>("-debug"), argc, argv);
  if (i > 0) {
    debug_mode = 1;
    if (debug_mode > 0)
      printf("debug mode: %d\n", debug_mode);
  }

  // search for train file
  i = argPos(const_cast<char *>("-dpm"), argc, argv);
  if (i > 0) {
    if (i + 1 == argc) {
      printf("ERROR: dpm data file not specified!\n");
      return 0;
    }

    snprintf(dpm_file, strlen(argv[i + 1])+1, "%s", argv[i + 1]);
  }

  // search for train file
  i = argPos(const_cast<char *>("-train"), argc, argv);
  if (i > 0) {
    if (i + 1 == argc) {
      printf("ERROR: training data file not specified!\n");
      return 0;
    }

    snprintf(train_file, strlen(argv[i + 1])+1, "%s", argv[i + 1]);

    if (debug_mode > 0)
      printf("train file: %s\n", train_file);

    f = fopen(train_file, "rb");
    if (f == NULL) {
      printf("ERROR: training data file not found!\n");
      return 0;
    }
    fclose(f);
  } else {
    printf("ERROR: training data must be set.\n");
  }

  // search for valid file
  i = argPos(const_cast<char *>("-valid"), argc, argv);
  if (i > 0) {
    if (i + 1 == argc) {
      printf("ERROR: validating data file not specified!\n");
      return 0;
    }

    snprintf(valid_file, strlen(argv[i + 1])+1, "%s", argv[i + 1]);

    if (debug_mode > 0)
      printf("valid file: %s\n", valid_file);

    f = fopen(valid_file, "rb");
    if (f == NULL) {
      printf("ERROR: validating data file not found!\n");
      return 0;
    }
    fclose(f);
    valid_mode = 1;
  }

  // search for test file
  i = argPos(const_cast<char *>("-test"), argc, argv);
  if (i > 0) {
    if (i + 1 == argc) {
      printf("ERROR: testing data file not specified!\n");
      return 0;
    }

    snprintf(test_file, strlen(argv[i + 1])+1, "%s", argv[i + 1]);

    if (debug_mode > 0)
      printf("test file: %s\n", test_file);

    f = fopen(test_file, "rb");
    if (f == NULL) {
      printf("ERROR: testing data file not found!\n");
      return 0;
    }
    fclose(f);
    test_mode = 1;
  }

  i = argPos(const_cast<char *>("-cut_point"), argc, argv);
  if (i > 0) {
    if (i + 1 == argc) {
      printf("ERROR: cut point not specified!\n");
      return 0;
    }

    cut_point = atoi(argv[i + 1]);

    if (debug_mode > 0)
      printf("cut point: %d\n", cut_point);
  }
  if (cut_point <= 0) {
    printf("ERROR: no or invalid cut point received!\n");
    return 0;
  }

  //init_class();

  create_data(train_file, "train_input_data.bin", 1);
  create_data(train_file, "train_label_data.bin", 2);
  //if (valid_mode) create_data(valid_file, "valid_data.bin");
  if (test_mode) {
     create_data(test_file, "test_input_data.bin", 1);
     create_data(test_file, "test_label_data.bin", 2);
  }

  return 0;
}
