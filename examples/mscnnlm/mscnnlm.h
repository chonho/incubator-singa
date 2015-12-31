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

#ifndef EXAMPLES_MSCNNLM_MSCNNLM_H_
#define EXAMPLES_MSCNNLM_MSCNNLM_H_

#include <string>
#include <vector>
#include "singa/singa.h"
#include "./mscnnlm.pb.h"

namespace mscnnlm {
using std::vector;
using singa::LayerProto;
using singa::Layer;
using singa::Param;
using singa::Blob;
using singa::Metric;

using namespace mshadow;
using mshadow::Tensor;
/**
 * Base RNN layer. May make it a base layer of SINGA.
 */
class MSCNNLayer : virtual public singa::Layer {
 public:
  /**
   * The recurrent layers may be unrolled different times for different
   * iterations, depending on the applications. For example, the ending word
   * of a sentence may stop the unrolling; unrolling also stops when the max
   * window size is reached. Every layer must reset window_ in its
   * ComputeFeature function.
   *
   * @return the effective BPTT length, which is <= max_window.
   */
  inline int window() { return window_; }

 protected:
  //!< effect window size for BPTT
  int window_;
};

/**
 * Input layer that get read records from data shard
 */
class DataLayer : public MSCNNLayer, public singa::InputLayer {
 public:
  ~DataLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  int max_window() const {
    return max_window_;
  }

 private:
  int num_feature_;
  int max_window_;
  singa::io::Store* store_ = nullptr;
};


/**
 * LabelLayer that read records_[1] to records_[window_] from DataLayer to
 * offer label information
class LabelLayer : public RNNLayer {
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override {}
};
 */


/**
 * Word embedding layer that get one row from the embedding matrix for each
 * word based on the word index
 */
class EmbeddingLayer : public MSCNNLayer {
 public:
  ~EmbeddingLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{embed_};
    return params;
  }


 private:
  int word_dim_;
  int vocab_size_;
  //!< word embedding matrix of size vocab_size_ x word_dim_
  Param* embed_;
};

/**
 * Convolution layer that get one word from the lower layer(data/embedding layer)
 * for each word based on its index
 */
class MSInnerProductLayer : public MSCNNLayer {
 public:
  ~MSInnerProductLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  void InitCombination();
  int Combination(int n, int k);
  void Copy(float* dst, float* src, int* s, int k, int vdim);
  void KSubset(int n, int* s, int sindxe, int index, int k,
    float* dst, float* src, int vdim);
  Tensor<cpu, 2> Concatenation(const vector<Layer*>& srclayers);

 private:
  int batchsize_;
  // total number of unique characters(events), set manually now.
  int vacab_size_;
  // number of concatenate vector
  int kernel_;
  // hdim_: length of result vector
  int vdim_, hdim_;
  // combination number
  int rows_;
  bool transpose_;
  Param *weight_, *bias_;
  // other parameters
  int N;
  const static int MAXN = 200;
  int C[MAXN + 1][MAXN + 1];
  int **index_;
};

class PoolingOverTime : public MSCNNLayer {
 public:
  ~PoolingOverTime();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
 private:
  int batchsize_;
  int vdim_;
  int *index_;
  //PoolingOverTimeProto_PoolMethod pool_;
};

/**
 * hid[t] = sigmoid(hid[t-1] * W + src[t])
 */
class HiddenLayer : public MSCNNLayer {
 public:
  ~HiddenLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_};
    return params;
  }


 private:
  Param* weight_;
};

/**
 * p(word at t+1 is from class c) = softmax(src[t]*Wc)[c]
 * p(w|c) = softmax(src[t]*Ww[Start(c):End(c)])
 * p(word at t+1 is w)=p(word at t+1 is from class c)*p(w|c)
 */
class LossLayer : public MSCNNLayer {
 public:
  ~LossLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

  const std::string ToString(bool debug, int flag) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{word_weight_, class_weight_};
    return params;
  }

 private:
  std::vector<Blob<float>> pword_;
  Blob<float> pclass_;
  Param* word_weight_, *class_weight_;
  float loss_, ppl_;
  int num_;
};
}  // namespace mscnnlm
#endif  // EXAMPLES_MSCNNLM_MSCNNLM_H_
