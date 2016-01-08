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
#include "./mscnnlm.h"

#include <string>
#include <algorithm>
#include <glog/logging.h>
#include "mshadow/tensor.h"
#include "mshadow/tensor_expr.h"
#include "mshadow/cxxnet_op.h"
#include "./mscnnlm.pb.h"

namespace mscnnlm {
using std::vector;
using std::string;

using namespace mshadow;
using mshadow::cpu;
using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Tensor;

inline Tensor<cpu, 2> RTensor2(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 2> tensor(blob->mutable_cpu_data(),
      Shape2(shape[0], blob->count() / shape[0]));
  return tensor;
}

inline Tensor<cpu, 1> RTensor1(Blob<float>* blob) {
  Tensor<cpu, 1> tensor(blob->mutable_cpu_data(), Shape1(blob->count()));
  return tensor;
}


/*******DataLayer**************/
DataLayer::~DataLayer() {
  if (store_ != nullptr)
    delete store_;
}

void DataLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers) {
  LOG(ERROR) << "Setup @ Data";
  MSCNNLayer::Setup(conf, srclayers);
  string key;
  max_window_ = conf.GetExtension(data_conf).max_window();
  max_word_len_ = conf.GetExtension(data_conf).max_word_len(); // max num chars in a word
  max_num_word_ = conf.GetExtension(data_conf).max_num_word();
  int row = max_num_word_ * max_word_len_;
  int col = 4 + max_word_len_; // 4: label, word_idx, word_length, delta_time
  data_.Reshape(vector<int>{row, col});
  window_ = 0;
  LOG(ERROR) << "row, col: " << row << ", " << col;
}

void SetInst(int k, const WordCharRecord& ch, Blob<float>* to, int max_wlen) {
  float* dptr = to->mutable_cpu_data() + k * (4 + max_wlen);
  dptr[0] = static_cast<float>(ch.label());
  dptr[1] = static_cast<float>(ch.word_index());
  dptr[2] = static_cast<float>(ch.word_length());
  dptr[3] = static_cast<float>(ch.delta_time());
  for (int i=0; i<ch.word_length(); i++)
    dptr[4+i] = static_cast<float>(ch.char_index(i));
}

void ShiftInst(int from, int to,  Blob<float>* data) {
  const float* f = data->cpu_data() + from * 3;
  float* t = data->mutable_cpu_data() + to * 3;
  // hard code the feature dim to be 5;
  t[0] = f[0]; t[1] = f[1]; t[2] = f[2]; t[3] = f[3];
}

void DataLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  string key, value;
  WordCharRecord ch;
  LOG(ERROR) << "Comp @ Data -----";
  if (store_ == nullptr) {
    store_ = singa::io::OpenStore(
        layer_conf_.GetExtension(data_conf).backend(),
        layer_conf_.GetExtension(data_conf).path(),
        singa::io::kRead);
    //store_->Read(&key, &value);
    //ch.ParseFromString(value);
    //SetInst(0, ch, &data_);
  }
  //ShiftInst(window_, 0, &data_);
  window_ = max_num_word_;
  for (int i = 0; i < max_window_; i++) {
    if (!store_->Read(&key, &value)) {
      store_->SeekToFirst();
      CHECK(store_->Read(&key, &value));
    }
    ch.ParseFromString(value);
    SetInst(i, ch, &data_, max_word_len_);

    if (ch.word_index() == -1) {
      window_ = i;
      break;
    }
  }
}


/*******EmbeddingLayer**************/
EmbeddingLayer::~EmbeddingLayer() {
  delete embed_;
}

void EmbeddingLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  LOG(ERROR) << "Setup @ Embed";
  MSCNNLayer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  int max_window = srclayers[0]->data(this).shape()[0]; // # of unique chars in dataset (vocab_size)
  word_dim_ = conf.GetExtension(embedding_conf).word_dim();
  data_.Reshape(vector<int>{max_window, word_dim_});
  grad_.ReshapeLike(data_);
  vocab_size_ = conf.GetExtension(embedding_conf).vocab_size();
  embed_ = Param::Create(conf.param(0));
  embed_->Setup(vector<int>{vocab_size_, word_dim_});
  LOG(ERROR) << "data row, col: " << max_window << ", " << word_dim_;
  LOG(ERROR) << "embd row, col: " << vocab_size_ << ", " << word_dim_;
}

void EmbeddingLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[0]);
  LOG(ERROR) << "Comp @ Embed";
  window_ = datalayer->window(); // <-- # of words in patient
  auto chars = RTensor2(&data_);
  auto embed = RTensor2(embed_->mutable_data());

  const float* idxptr = datalayer->data(this).cpu_data();
  int i = 0;
  int shift = datalayer->data(this).shape()[1];
  for (int k=0; k < window_; k++) {
    int wlen = static_cast<int>(idxptr[k*shift+2]);
    for(int c=0; c<wlen; c++)
    {
      int char_idx = static_cast<int>(idxptr[k*shift+4+c]);  // "4": start position of char index
      CHECK_GE(char_idx, 0);
      CHECK_LT(char_idx, vocab_size_);
      Copy(chars[i++], embed[char_idx]);
    }
  }
}

void EmbeddingLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  auto grad = RTensor2(&grad_);
  auto gembed = RTensor2(embed_->mutable_grad());
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[0]);
  gembed = 0;
  const float* idxptr = datalayer->data(this).cpu_data();
  int i = 0;
  int shift = datalayer->data(this).shape()[1];
  for (int k = 0; k < window_; k++) {
    int wlen = static_cast<int>(idxptr[k*shift+2]);
    for(int c=0; c<wlen; c++)
    {
      int char_idx = static_cast<int>(idxptr[k*shift+4+c]);  // "4": start position of char index
      Copy(gembed[char_idx], grad[i++]);
    }
  }
}

/*************Concat Layer***************/
int ConcatLayer::Binomial(int n, int k) {
  if (k < 0 || k > n)
    return 0;
  if (k > n - k)
    k = n - k;
  int cb = 1;
  for (int i = 1; i <= k; i++) {
    cb *= (n - (k -i));
    cb /= i;
  }
  return cb;
}

void ConcatLayer::Combinations(int n, int k) {
  int c = Binomial(n, k);
  int *ind =new int[k];
  for (int i = 0; i < k; i++)
    ind[i] = i;
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < k; j++)
      concat_index_[i][j] = ind[j];
    int x = k - 1;
    bool loop;
    do {
      loop = false;
      ind[x] = ind[x] + 1;
      if (ind[x] > n - (k - x)) {
        x--;
        loop = (x >= 0);
      }
      else {
        for (int x1 = x + 1; x1 < k; x1++)
          ind[x1] = ind[x1 - 1] + 1;
      }
    } while(loop);
  }
  delete[] ind;
}

void ConcatLayer::SetIndex(const vector<Layer*>& srclayers) {
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[1]);
  window_ = datalayer->window();    // #words in a patient
  const float* idxptr = datalayer->data(this).cpu_data();
  int shift = datalayer->data(this).shape()[1];
  for (int i = 0; i < window_; i++) {
    int wlen = static_cast<int>(idxptr[i * shift + 2]);
    word_index_[i] = wlen;
    for (int j = 0; j < wlen; j++) {
      int c = static_cast<int>(idxptr[i * shift + 4 + j]);
      char_index_[i][j] = c;
    }
  }
}

ConcatLayer::~ConcatLayer() {
    delete[] word_index_;
    for (int i = 0; i < max_num_word_; i++)
      delete[] char_index_[i];
    delete[] char_index_;
    int b = Binomial(max_word_len_, kernel_);
    for (int i = 0; i < b; i++)
      delete[] concat_index_[i];
    delete[] concat_index_;
}

void ConcatLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  LOG(ERROR) << "Setup @ Concat";
  CHECK_EQ(srclayers.size(), 2);
  max_word_len_ = conf.GetExtension(data_conf).max_word_len();
  max_num_word_ = conf.GetExtension(data_conf).max_num_word();
  word_dim_ = conf.GetExtension(embedding_conf).word_dim();
  kernel_ = conf.GetExtension(concat_conf).kernel();
  int cols = kernel_ * word_dim_;
  int bino = Binomial(max_word_len_, kernel_);
  int rows = max_num_word_ * bino;
  data_.Reshape(vector<int>{rows, cols});
  grad_.ReshapeLike(data_);
  word_index_ = new int[max_num_word_];
  char_index_ = new int*[max_num_word_];
  for (int i = 0; i < max_num_word_; i++)
    char_index_[i] = new int[max_word_len_];
  SetIndex(srclayers);
  // for each word, a concat_index_ indicates all the combinations
  concat_index_ = new int*[bino];
  for (int i = 0; i < max_num_word_; i++)
    concat_index_[i] = new int[kernel_];
}

void ConcatLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  auto emlayer = dynamic_cast<EmbeddingLayer*>(srclayers[0]);
  window_ = emlayer->window();      // #words in a patient
  auto src_ptr = emlayer->data(this).cpu_data();
  float* dst_ptr = data_.mutable_cpu_data();
  data_.SetValue(0);

  for (int w = 0; w < window_; w++) {
    Combinations(word_index_[w], kernel_);
    int b = Binomial(word_index_[w], kernel_);
    int max = Binomial(max_word_len_, kernel_);
    for (int r = 0; r < b; r++)
      for (int c = 0; c < kernel_; c++)
        memcpy(dst_ptr + ((w * max + r) * kernel_ + c) * word_dim_,
         src_ptr + concat_index_[r][c] * word_dim_, word_dim_);
  }
}

void ConcatLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  auto grad = Tensor2(&grad_);
  // initialize gsrc
  srclayers[0]->mutable_grad(this)->SetValue(0);
  auto gsrc = Tensor2(srclayers[0]->mutable_grad(this));
  auto emlayer = dynamic_cast<EmbeddingLayer*>(srclayers[0]);
  window_ = emlayer->window();      // #words in a patient

  for (int w = 0; w < window_; w++) {
    Combinations(word_index_[w], kernel_);
    int b = Binomial(word_index_[w], kernel_);
    int max = Binomial(max_word_len_, kernel_);
    for (int r = 0; r < b; r++)
      for (int c = 0; c < kernel_; c++)
        for (int i = 0; i < word_dim_; i++)
          gsrc[concat_index_[r][c]][i] += grad[w * max + r][c * word_dim_ + i];
  }
}

/*********PoolingOverTime Layer*********/
void PoolingOverTime::SetIndex(const vector<Layer*>& srclayers) {
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[1]);
  window_ = datalayer->window();    // #words in a patient
  const float* idxptr = datalayer->data(this).cpu_data();
  int shift = datalayer->data(this).shape()[1];
  for (int i = 0; i < window_; i++) {
    int wlen = static_cast<int>(idxptr[i * shift + 2]);
    word_index_[i] = wlen;
  }
}

int PoolingOverTime::Binomial(int n, int k) {
  if (k < 0 || k > n)
    return 0;
  if (k > n - k)
    k = n - k;
  int cb = 1;
  for (int i = 1; i <= k; i++) {
    cb *= (n - (k -i));
    cb /= i;
  }
  return cb;
}

PoolingOverTime::~PoolingOverTime() {
  delete[] word_index_;
  for (int i = 0; i < max_num_word_; i++)
    delete[] max_index_[i];
  delete[] max_index_;
}

void PoolingOverTime::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  LOG(ERROR) << "Setup @ Pooling";
  CHECK_EQ(srclayers.size(), 2);
  MSCNNLayer::Setup(conf, srclayers);
  const auto& convlayer = srclayers[0]->data(this);
  max_num_word_ = conf.GetExtension(data_conf).max_num_word();
  max_word_len_ = conf.GetExtension(data_conf).max_word_len();
  kernel_ = conf.GetExtension(concat_conf).kernel();
  batchsize_ = convlayer.shape()[0];
  vdim_ = convlayer.count() / batchsize_;
  // will append time from data layer
  data_.Reshape(vector<int>{max_num_word_, vdim_ + 1});
  grad_.ReshapeLike(data_);
  int *word_index_ = new int[max_num_word_];
  SetIndex(srclayers);
  max_index_ = new int*[max_num_word_];
  for (int i = 0; i < max_num_word_; i++)
    max_index_[i] = new int[vdim_];
}

void PoolingOverTime::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  LOG(ERROR) << "CompFeat @ Pooling";
  auto data = Tensor2(&data_);
  auto src = Tensor2(srclayers[0]->mutable_data(this));
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[1]);
  window_ = datalayer->window();
  int max = Binomial(max_word_len_, kernel_);
  //LOG(ERROR) << "window_ :" << window_;
  for (int w = 0; w < window_; w++) {
    for (int c = 0; c < vdim_; c++) {
      data[w][c] = src[w * max][c];
      max_index_[w][c] = w * max;
    }
    int b = Binomial(word_index_[w], kernel_);
    for (int r = 0; r < b; r++)
      for (int c = 0; c < vdim_; c++)
        if (src[w * max + r][c] > data[w][c]) {
          data[w][c] = src[w * max + r][c];
          max_index_[w][c] = w * max + b;
        }
  }
  // append time from data layer
  const float* idxptr = datalayer->data(this).cpu_data();
  int shift = datalayer->data(this).shape()[1];
  for (int i = 0; i < window_; i++) {
    int delta_time = static_cast<int>(idxptr[i * shift + 3]);
    data[i][vdim_] = delta_time;
  }
}

void PoolingOverTime::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  auto grad = Tensor2(&grad_);
  srclayers[0]->mutable_grad(this)->SetValue(0);
  auto gsrc = Tensor2(srclayers[0]->mutable_grad(this));
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[1]);
  window_ = datalayer->window();
  for (int w = 0; w < window_; w++)
    for (int c = 0; c < vdim_; c++)
      gsrc[max_index_[w][c]][c] = grad[w][c];
}


/*******WordPoolingLayer*********/
WordPoolingLayer::~WordPoolingLayer() {
  delete[] index_;
}

void WordPoolingLayer::Setup(const LayerProto& conf,
  const vector<Layer*>& srclayers) {
  LOG(ERROR) << "Setup @ WordPoolingLayer";
  CHECK_EQ(srclayers.size(), 1);
  MSCNNLayer::Setup(conf, srclayers);
  const auto& src = srclayers[0]->data(this);
  batchsize_ = src.shape()[0];
  vdim_ = src.count() / batchsize_;
  data_.Reshape(vector<int>{vdim_});
  grad_.ReshapeLike(data_);
  index_ = new int[vdim_];
}

void WordPoolingLayer::ComputeFeature(int flag,
  const vector<Layer*>& srclayers) {
  auto data = Tensor1(&data_);
  auto src = Tensor2(srclayers[0]->mutable_data(this));
  for (int i = 0; i < vdim_; i++) {
    data[i] = src[0][i];
    index_[i] = 0;
  }
  for (int i = 0; i < vdim_; i++)
    for (int j = 1; j < batchsize_; j++)
      if (data[i] < src[j][i]) {
        data[i] = src[j][i];
        index_[i] = j;
      }
}

void WordPoolingLayer::ComputeGradient(int flag,
  const vector<Layer*>& srclayers) {
  auto grad = Tensor1(&grad_);
  srclayers[0]->mutable_grad(this)->SetValue(0);
  auto gsrc = Tensor2(srclayers[0]->mutable_grad(this));
  for (int i = 0; i < vdim_; i++)
    gsrc[index_[i]][i] = grad[i];
}

/*********** Implementation for LossLayer **********/
LossLayer::~LossLayer() {
  delete word_weight_;
  delete class_weight_;
}

void LossLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers) {
  MSCNNLayer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 2);
  const auto& src = srclayers[0]->data(this);
  int max_window = src.shape()[0];
  int vdim = src.count() / max_window;   // Dimension of input
  int vocab_size = conf.GetExtension(loss_conf).vocab_size();
  int nclass = conf.GetExtension(loss_conf).nclass();
  word_weight_ = Param::Create(conf.param(0));
  word_weight_->Setup(vector<int>{vocab_size, vdim});
  class_weight_ = Param::Create(conf.param(1));
  class_weight_->Setup(vector<int>{nclass, vdim});

  pword_.resize(max_window);
  pclass_.Reshape(vector<int>{max_window, nclass});
}

void LossLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  window_ = dynamic_cast<MSCNNLayer*>(srclayers[0])->window();
  auto pclass = RTensor2(&pclass_);
  auto src = RTensor2(srclayers[0]->mutable_data(this));
  auto word_weight = RTensor2(word_weight_->mutable_data());
  auto class_weight = RTensor2(class_weight_->mutable_data());
  const float * label = srclayers[1]->data(this).cpu_data();

  float loss = 0.f, ppl = 0.f;
  for (int t = 0; t < window_; t++) {
    // label is the next word
    int start = static_cast<int>(label[(t + 1) * 4 + 2]);
    int end = static_cast<int>(label[(t + 1) * 4 + 3]);

    auto wordWeight = word_weight.Slice(start, end);
    CHECK_GT(end, start);
    pword_[t].Reshape(std::vector<int>{end-start});
    auto pword = RTensor1(&pword_[t]);
    pword = dot(src[t], wordWeight.T());
    Softmax(pword, pword);

    pclass[t] = dot(src[t], class_weight.T());
    Softmax(pclass[t], pclass[t]);

    int wid = static_cast<int>(label[(t + 1) * 4 + 0]);
    int cid = static_cast<int>(label[(t + 1) * 4 + 1]);
    CHECK_GT(end, wid);
    CHECK_GE(wid, start);
    loss_ += -log(std::max(pword[wid - start] * pclass[t][cid], FLT_MIN));
    ppl_ += log10(std::max(pword[wid - start] * pclass[t][cid], FLT_MIN));
  }
  num_ += window_;
}

void LossLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  auto pclass = RTensor2(&pclass_);
  auto src = RTensor2(srclayers[0]->mutable_data(this));
  auto gsrc = RTensor2(srclayers[0]->mutable_grad(this));
  auto word_weight = RTensor2(word_weight_->mutable_data());
  auto gword_weight = RTensor2(word_weight_->mutable_grad());
  auto class_weight = RTensor2(class_weight_->mutable_data());
  auto gclass_weight = RTensor2(class_weight_->mutable_grad());
  const float * label = srclayers[1]->data(this).cpu_data();
  gclass_weight = 0;
  gword_weight = 0;
  for (int t = 0; t < window_; t++) {
    int start = static_cast<int>(label[(t + 1) * 4 + 2]);
    int end = static_cast<int>(label[(t + 1) * 4 + 3]);
    int wid = static_cast<int>(label[(t + 1) * 4 + 0]);
    int cid = static_cast<int>(label[(t + 1) * 4 + 1]);
    auto pword = RTensor1(&pword_[t]);
    CHECK_GT(end, wid);
    CHECK_GE(wid, start);

    // gL/gclass_act
    pclass[t][cid] -= 1.0;
    // gL/gword_act
    pword[wid - start] -= 1.0;

    // gL/gword_weight
    gword_weight.Slice(start, end) += dot(pword.FlatTo2D().T(),
                                          src[t].FlatTo2D());
    // gL/gclass_weight
    gclass_weight += dot(pclass[t].FlatTo2D().T(),
                         src[t].FlatTo2D());

    gsrc[t] = dot(pword, word_weight.Slice(start, end));
    gsrc[t] += dot(pclass[t], class_weight);
  }
}

const std::string LossLayer::ToString(bool debug, int flag) {
  float loss = loss_ / num_;
  float ppl = exp10(- ppl_ / num_);
  loss_ = 0;
  num_ = 0;
  ppl_ = 0;
  return "loss = " + std::to_string(loss) + ", ppl = " + std::to_string(ppl);
}
}   // end of namespace mscnnlm
