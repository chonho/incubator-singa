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
  num_feature_ = 5;
  data_.Reshape(vector<int>{max_window_ + 1, num_feature_});
  window_ = 0;
}

void SetInst(int k, const WordCharRecord& ch, Blob<float>* to) {
  float* dptr = to->mutable_cpu_data() + k * 5;
  dptr[0] = static_cast<float>(ch.label());
  dptr[1] = static_cast<float>(ch.word_index());
  dptr[2] = static_cast<float>(ch.word_length());
  dptr[3] = static_cast<float>(ch.delta_time());
  dptr[4] = static_cast<float>(ch.char_index());
}

void ShiftInst(int from, int to,  Blob<float>* data) {
  const float* f = data->cpu_data() + from * 5;
  float* t = data->mutable_cpu_data() + to * 5;
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
  window_ = max_window_;
  for (int i = 0; i < max_window_; i++) {
  //for (int i = 1; i <= max_window_; i++) {
    if (!store_->Read(&key, &value)) {
      store_->SeekToFirst();
      CHECK(store_->Read(&key, &value));
    }
    ch.ParseFromString(value);
    SetInst(i, ch, &data_);

    LOG(ERROR) << ch.label() << " " << ch.word_index() << " "
               << ch.word_length() << " " << ch.delta_time() << " "
               << ch.char_index();

    if (ch.word_length() == i+1) {
      window_ = ch.word_length();
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
  // max_window is the number of characters in each word
  int max_window = srclayers[0]->data(this).shape()[0];
  // word_dim_ is the embedded feature length
  word_dim_ = conf.GetExtension(embedding_conf).word_dim();
  // the data should be shaped as a 3D matrix
  data_.Reshape(vector<int>{max_window, word_dim_});
  grad_.ReshapeLike(data_);
  vocab_size_ = conf.GetExtension(embedding_conf).vocab_size();
  embed_ = Param::Create(conf.param(0));
  embed_->Setup(vector<int>{vocab_size_, word_dim_});
}

void EmbeddingLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[0]);
  LOG(ERROR) << "Comp @ Embed";
  window_ = datalayer->window();
  auto words = RTensor2(&data_);
  auto embed = RTensor2(embed_->mutable_data());

  const float* idxptr = datalayer->data(this).cpu_data();
  for (int t = 0; t < window_; t++) {
    int cidx = static_cast<int>(idxptr[t * 5 + 4]);  // 5: num_feature, 4: pos of char index
    LOG(ERROR) << "char_index: " << cidx;
    CHECK_GE(cidx, 0);
    CHECK_LT(cidx, vocab_size_);
    Copy(words[t], embed[cidx]);
  }
}

void EmbeddingLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  LOG(ERROR) << "CompGrad @ Embed";
  auto grad = RTensor2(&grad_);
  auto gembed = RTensor2(embed_->mutable_grad());
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[0]);
  gembed = 0;
  const float* idxptr = datalayer->data(this).cpu_data();
  for (int t = 0; t < window_; t++) {
    int idx = static_cast<int>(idxptr[t * 5 + 4]);  // same as above
    Copy(gembed[idx], grad[t]);
  }
}

/**************Xiangrui***************/

// Our convolution layer should be here
// should be similar to InerProdcut layer
MSInnerProductLayer::~MSInnerProductLayer() {
  delete weight_;
  delete bias_;
  for (int i = 0; i < rows_; i++)
    delete[] index_[i];
  delete[] index_;
}

void MSInnerProductLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  LOG(ERROR) << "Setup @ Multi-Scale Convolution";
  CHECK_EQ(srclayers.size(), 1);
  const auto& src = srclayers[0]->data(this);
  batchsize_ = src.shape()[0];
  vdim_ = src.count() / batchsize_;
  // get conf parameters
  hdim_ = conf.GetExtension(msip_conf).num_output();
  kernel_ = conf.GetExtension(msip_conf).kernel();
  transpose_ = conf.GetExtension(msip_conf).transpose();
  N = 0;

  if (partition_dim() > 0)
    hdim_ /= srclayers.at(0)->num_partitions();
  InitCombination();
  rows_ = Combination(batchsize_, kernel_);
  data_.Reshape(vector<int>{rows_, hdim_});
  grad_.ReshapeLike(data_);
  weight_ = Param::Create(conf.param(0));
  bias_ = Param::Create(conf.param(1));
  index_ = new int*[rows_];
  for (int i = 0; i < rows_; i++)
    index_[i] = new int[kernel_];
  // The length of input vector is kernel_ * vacab_size_
  if (transpose_)
    weight_->Setup(vector<int>{kernel_ * vdim_, hdim_});
  else
    weight_->Setup(vector<int>{hdim_, kernel_ * vdim_});
  //bias_->Setup(vector<int>{hdim_});
  bias_->Setup(vector<int>{hdim_});
}

// C(n, k) = C(n-1, k-1) + C(n-1, k)
//const int MAXN = 200;
//int C[MAXN + 1][MAXN + 1];
void MSInnerProductLayer::InitCombination()
{
  int i, j;
  for (i = 0; i <= MAXN; i++) {
    C[0][i] = 0;
    C[i][0] = 1;
  }
  for (i = 1; i <= MAXN; i++)
    for (j = 1; j <= MAXN; j++)
      C[i][j] = C[i - 1][j] + C[i - 1][j - 1];
}

int MSInnerProductLayer::Combination(int n, int k)
{
  return C[n][k];
}

// k rows in src, copy to one row in dst.
// indices are in s
// vdim: length of row in src
void MSInnerProductLayer::Copy(float* dst, float* src, int* s, int k, int vdim) {
  float* t_src;
  float* t_dst = dst;
  for (int i = 0; i < k; i++) {
    t_src = src + s[i] * vdim;
    memcpy(t_dst, t_src, vdim);
    //for (int j = 0; j < vdim; j++)
    //  *(dst + j) = *(t_src + j);
    t_dst += vdim;
  }
}

//int N = 0;
void MSInnerProductLayer::KSubset(int n, int *s, int sindex, int index, int k,
    float* dst, float* src, int vdim) {
  if (index > n)
    return;
  if (k == 0){
    // N: row number in dst
    // k: kernel
    float* dn = dst + k * vdim * N;
    memcpy(index_[N], s, k);
    Copy(dn, src, s, k, vdim);
    N++;
    return;
  }
  s[sindex] = index;
  KSubset(n, s, sindex+1, index+1, k-1, dst, src, vdim);
  KSubset(n, s, sindex, index+1, k, dst, src, vdim);
}

Tensor<cpu, 2> MSInnerProductLayer::Concatenation(const vector<Layer*>& srclayers) {
  //InitCombination();
  auto Embeddinglayer = dynamic_cast<EmbeddingLayer*>(srclayers[0]);
  LOG(ERROR) << "Concatenation";
  window_ = Embeddinglayer->window();
  //int rows = Combination(batchsize_, kernel_);
  Blob<float>* concatenated = new Blob<float>(rows_, kernel_ * vdim_);
  concatenated->SetValue(0);
  auto src = srclayers[0]->mutable_data(this)->mutable_cpu_data();
  float* a_ptr = concatenated->mutable_cpu_data();
  int* s = new int[kernel_];
  KSubset(window_, s, 0, 0, kernel_, a_ptr, src, vdim_);
  return Tensor2(concatenated);
}

void MSInnerProductLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  LOG(ERROR) << "CompFeat @ Multi-Scale Convolution";
  auto data = Tensor2(&data_);
  auto src = Concatenation(srclayers);//Tensor2(srclayers[0]->mutable_data(this));
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());

  LOG(ERROR) << "concatenation ok!";
  LOG(ERROR) << "s_shape[0]: " << src.shape[0];
  LOG(ERROR) << "s_shape[1]: " << src.shape[1];
  LOG(ERROR) << "w_shape[0]: " << weight.shape[0];
  LOG(ERROR) << "w_shape[1]: " << weight.shape[1];
  LOG(ERROR) << "d_shape[0]: " << data.shape[0];
  LOG(ERROR) << "d_shape[1]: " << data.shape[1];
  if (transpose_)
    data = dot(src, weight);
  else
    data = dot(src, weight.T());
  LOG(ERROR) << "dot ok!";
  // repmat: repeat bias vector into rows rows
  //int rows = Combination(batchsize_, kernel_);
  data += expr::repmat(bias, rows_);
  LOG(ERROR) << "plus ok!";
}

void MSInnerProductLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  LOG(ERROR) << "CompGrad @ Multi-Scale Convolution";
  auto src = Concatenation(srclayers);//Tensor2(srclayers[0]->mutable_data(this));
  auto grad = Tensor2(&grad_);
  auto weight = Tensor2(weight_->mutable_data());
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());

  gbias = expr::sum_rows(grad);
  if (transpose_)
    gweight = dot(src.T(), grad);
  else
    gweight = dot(grad.T(), src);
  LOG(ERROR) << "gweight dot OK!";
  if (srclayers[0]->mutable_grad(this) != nullptr) {
    auto gsrc = Tensor2(srclayers[0]->mutable_grad(this));
    //auto tmp = Tensor2(&Blob<float>(rows_, kernel_ * vdim_));
    TensorContainer<cpu, 2> tmp(Shape2(rows_, kernel_ * vdim_));
    tmp = dot(grad, weight);
    LOG(ERROR) << "tmp dot OK!";
    int i, j, k;
    for (i = 0; i < rows_; i++) {
      for (j = 0; j < kernel_; j++)
        for (k = 0; k < vdim_; k++)
          gsrc[index_[i][j]][k] += tmp[i][j * vdim_ + k];
    }
  }
}


/*********PoolingOverTime Layer*********/
PoolingOverTime::~PoolingOverTime() {
  delete[] index_;
}

void PoolingOverTime::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  LOG(ERROR) << "Setup @ Pooling";
  MSCNNLayer::Setup(conf, srclayers);
  const auto& src = srclayers[0]->data(this);
  CHECK_EQ(srclayers.size(), 1);
  //PoolingOverTimeProto pool_conf = conf.pot_conf();
  //pool_ = pool_conf.pool();
  batchsize_ = src.shape()[0];
  vdim_ = src.count() / batchsize_;
  data_.Reshape(vector<int>{vdim_});
  grad_.ReshapeLike(data_);
  index_ = new int[vdim_];
}

void PoolingOverTime::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  LOG(ERROR) << "CompFeat @ Pooling";
  auto data = Tensor1(&data_);
  auto src = Tensor2(srclayers[0]->mutable_data(this));
  auto msip = dynamic_cast<MSInnerProductLayer*>(srclayers[0]);
  window_ = msip->window();
  LOG(ERROR) << "window_ :" << window_;
  int i, j;
  for (i = 0; i < vdim_; i++) {
    data[i] = src[0][i];
    index_[i] = 0;
  }
  for (i = 0; i < vdim_; i++)
    for (j = 1; j < window_; j++)
      if (src[j][i] > data[i]) {
        data[i] = src[j][i];
        index_[i] = j;
      }
}

void PoolingOverTime::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  auto grad = Tensor1(&grad_);
  srclayers[0]->mutable_grad(this)->SetValue(0);
  auto gsrc = Tensor2(srclayers[0]->mutable_grad(this));
  int i;
  for (i = 0; i < vdim_; i++)
    gsrc[index_[i]][i] = grad[i];
}

OneDimConvLayer::~OneDimConvLayer()
{
  delete[] weight;
  delete[] bias;
}

void OneDimConvLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 2);
  Layer::Setup(conf, srclayers);

  OneDimConvProto conv_conf = conf.Conv_conf();
  kernel_ = conv_conf.kernel();
  CHECK_NE(kernel_, 0);
  pad_ = conv_conf.pad();
  stride_ = conv_conf.stride();
  num_filters_ = conv_conf.num_filters();

  if (partition_dim() > 0)
    num_filters_ /= srclayers.at(0)->num_partitions();
  const vector<int>& srcshape = srclayers[0]->data(this).shape();
  batchsize_ = srcshape[0];     //how many words do we have? not this number
  int dim = srcshape.size();
  CHECK_GT(dim, 1);
  vdim_ = srcshape[0] + 1;          // the length of pooling vector for each words.
  conv_width_ = (vdim_ + 2 * pad_ - kernel_) / stride_ + 1;
  weight_width_ = kernel_ * vdim_;
  vector<int> shape{batchsize_, num_filters_, conv_width_};
  data.Reshape(shape);
  grad.ReshapeLike(data);
  weight_ = Param::Create(conf.param(0));
  weight_->Setup(vector<int>{num_filters_, weight_width_});
  bias_ = Param::Create(conf.param(1));
  bias_->Setup(vector<int>{num_filters});
}
/***********HiddenLayer**********/
HiddenLayer::~HiddenLayer() {
  delete weight_;
}

void HiddenLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  MSCNNLayer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  const auto& innerproductData = srclayers[0]->data(this);
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(srclayers[0]->grad(this));
  int word_dim = data_.shape()[1];
  weight_ = Param::Create(conf.param(0));
  weight_->Setup(std::vector<int>{word_dim, word_dim});
}

// hid[t] = sigmoid(hid[t-1] * W + src[t])
void HiddenLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  window_ = dynamic_cast<MSCNNLayer*>(srclayers[0])->window();
  auto data = RTensor2(&data_);
  auto src = RTensor2(srclayers[0]->mutable_data(this));
  auto weight = RTensor2(weight_->mutable_data());
  for (int t = 0; t < window_; t++) {  // Skip the 1st component
    if (t == 0) {
      data[t] = expr::F<op::sigmoid>(src[t]);
    } else {
      data[t] = dot(data[t - 1], weight);
      data[t] += src[t];
      data[t] = expr::F<op::sigmoid>(data[t]);
    }
  }
}

void HiddenLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  auto data = RTensor2(&data_);
  auto grad = RTensor2(&grad_);
  auto weight = RTensor2(weight_->mutable_data());
  auto gweight = RTensor2(weight_->mutable_grad());
  auto gsrc = RTensor2(srclayers[0]->mutable_grad(this));
  gweight = 0;
  TensorContainer<cpu, 1> tmp(Shape1(data_.shape()[1]));
  // Check!!
  for (int t = window_ - 1; t >= 0; t--) {
    if (t < window_ - 1) {
      tmp = dot(grad[t + 1], weight.T());
      grad[t] += tmp;
    }
    grad[t] = expr::F<op::sigmoid_grad>(data[t])* grad[t];
  }
  gweight = dot(data.Slice(0, window_-1).T(), grad.Slice(1, window_));
  Copy(gsrc, grad);
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
