#ifndef SINGA_NEURALNET_LAYER_H_
#define SINGA_NEURALNET_LAYER_H_

#include <lmdb.h>

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <memory>
#include <chrono>
#include <random>

#include "proto/job.pb.h"
#include "utils/data_shard.h"
#include "neuralnet/base_layer.h"

/**
 * \file this file includes the declarations neuron layer classes that conduct
 * the transformation of features.
 */
namespace singa {

/**
 * Convolution layer.
 */
class ConvolutionLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase) override;
  const vector<Param*> GetParams() const override {
    vector<Param*> params{weight_, bias_};
    return params;
  }
  ConnectionType src_neuron_connection(int k) const  override {
    // CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }
  ~ConvolutionLayer();

 protected:
  int kernel_, pad_,  stride_;
  int batchsize_,  channels_, height_, width_;
  int col_height_, col_width_, conv_height_, conv_width_, num_filters_;
  Param* weight_, *bias_;
  Blob<float> col_data_, col_grad_;
};

class DropoutLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase) override;

 protected:
  // drop probability
  float pdrop_;
  /* record which neuron is dropped, required for back propagating gradients,
   * if mask[i]=0, then the i-th neuron is dropped.
   */
  Blob<float> mask_;
};
/**
  * RBM visible layer
  */
class RBMVisLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto,
      int npartitions) override;
  virtual bool is_vislayer() const {
    return true;
  }

  void ComputeFeature(Phase phase,
     Metric *perf) override;
  void ComputeGradient(Phase phase) override;
  virtual void ComputeLoss(Metric* perf);
  virtual Blob<float>* mutable_data(const Layer* from, Phase phase) {
    if (phase == kPositive) {
      return &data_;
    } else {
       return &vis_sample_;
    }
  }
  virtual const Blob<float>& data(const Layer* from, Phase phase) const {
    if (phase == kPositive) {
      return data_;
    } else {
       return vis_sample_;
    }
  }
  // virtual void ToProto(LayerProto *layer_proto, bool copyData);
  const vector<Param*> GetParams() const override {
    vector<Param*> params{weight_, bias_};
    return params;
  }
  ~RBMVisLayer();


 private:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int batchsize_;
  // batchsize of negative phase
  int neg_batchsize_;
  bool is_first_iteration_vis_;
  float scale_;
  // srclayer index
  int data_idx_;
  int hid_idx_;
  Param* weight_, *bias_;
  // data to store sampling result
  Blob<float> vis_sample_;
  // in order to implement Persistent Contrastive Divergence,
};
/**
  * RBM hidden layer
  */
class RBMHidLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto,
      int npartitions) override;
  virtual bool is_hidlayer() const {
    return true;
  }

  void ComputeFeature(Phase phase,
     Metric *perf) override;
  void ComputeGradient(Phase phase) override;
  virtual Blob<float>* mutable_data(const Layer* from, Phase phase) {
    if (phase == kPositive)
      return &data_;
    else
      return &hid_sample_;
  }
  virtual const Blob<float>& data(const Layer* from, Phase phase) const {
    if (phase == kPositive)
      return data_;
    else
      return hid_sample_;
  }
  const vector<Param*> GetParams() const override {
    vector<Param*> params{weight_, bias_};
    return params;
  }
  ~RBMHidLayer();

 private:
  //! dimension of the hidden layer
  int hdim_;
  int vdim_;  // dimension of visible layer
  int batchsize_;
  // batchsize of negative phase
  int neg_batchsize_;
  float scale_;
  Blob<float> hid_sample_;
  Param* weight_, *bias_;
};
/**
  * fully connected layer
  */
class InnerProductLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase) override;


  ConnectionType src_neuron_connection(int k) const override {
    // CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }
  const vector<Param*> GetParams() const override {
    vector<Param*> params{weight_, bias_};
    return params;
  }
  ~InnerProductLayer();

 private:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int batchsize_;
  Param* weight_, *bias_;
};



/***********  Implementing layers used in RNNLM application ***********/
/**
  * 1-RnnlmComputationLayer, this layer will receive the ground truth and then compute the SUM(log(probability))
  */
class RnnlmComputationLayer: public Layer {
public:
    using Layer::ComputeFeature;
    using Layer::ComputeGradient;

    void Setup(const LayerProto& proto, int npartitions) override;//need to change the row, column order
    void ComputeFeature(Phase phase, Metric *perf) override;
    void ComputeGradient(Phase phase) override;

    ConnectionType src_neuron_connection(int k) const override {
        // CHECK_LT(k, srclayers_.size());
        return kOneToAll;
    }

    const vector<Param*> GetParams() const override {
        vector<Param*> params{weight_};
        return params;
    }
    ~RnnlmComputationLayer();


private:
    //! dimension of the hidden layer
    int hdim_;//dimension of output
    //! dimension of the visible layer
    int vdim_;//dimension of input
    //int batchsize_;
    int windowsize_; // Use windowsize_ to represent different timestamps
    Param* weight_; // Delete the parameter "bias" as there is no need to use “bias” according to the paper
    float sum_;
    int classsize_;
    int vocabsize_;
};


/**
  * 2-RnnlmSigmoidLayer, this layer will make use of information from previous time stamp
  */
class RnnlmSigmoidLayer: public Layer {
public:
    using Layer::ComputeFeature;
    using Layer::ComputeGradient;

    void Setup(const LayerProto& proto, int npartitions) override;//need to change the row, column order
    void ComputeFeature(Phase phase, Metric *perf) override;
    void ComputeGradient(Phase phase) override;

    ConnectionType src_neuron_connection(int k) const override {
        // CHECK_LT(k, srclayers_.size());
        return kOneToAll;
    }

    const vector<Param*> GetParams() const override {
        vector<Param*> params{weight_};
        return params;
    }
    ~RnnlmSigmoidLayer();


private:
    //! dimension of the hidden layer
    int hdim_;//dimension of output
    //! dimension of the visible layer
    int vdim_;//dimension of input
    //int batchsize_;
    int windowsize_; // Use windowsize_ to represent different timestamps
    Param* weight_; // The weight matrix between s(t-1) and s(t)
};


/**
  * 3-RnnlmInnerproductLayer, this layer is very similar to ordinary InnerproductLayer except that the consideration for window_size
  */
class RnnlmInnerproductLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase) override;


  ConnectionType src_neuron_connection(int k) const override {
    // CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }
  const vector<Param*> GetParams() const override {
    vector<Param*> params{weight_};
    return params;
  }
  ~RnnlmInnerproductLayer();

 private:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int windowsize_;  //The only difference from ordinary InnerProductLayer
  Param* weight_;
};

/**
  * 4-RnnlmWordinputLayer, this layer is in charge of managing the input words, for each inout word index, output the |V| vector
  * Note: the value of |V| is user-defined
  */
class RnnlmWordinputLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase) override;


  ConnectionType src_neuron_connection(int k) const override {  //TODO to check whether use this or not in this layer?
    // CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }
  const vector<Param*> GetParams() const override {
    vector<Param*> params{weight_};
    return params;
  }
  ~RnnlmWordinputLayer();

 private:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int windowsize_;
  int vocabsize_;   //The size of vocabulary
  Param* weight_;   //The weight matrix here is U, (vocab_size, |v|)
};

/**
  * 5-RnnlmWordparserLayer
  */
class RnnlmWordparserLayer: public ParserLayer {
 public:
  using ParserLayer::ParseRecords;
  //using Layer::partition_dim;   //to check later? using Layer:: or ParserLayer:: ?

  void Setup(const LayerProto& proto, int npartitions) override;
  void ParseRecords(Phase phase, const vector<Record>& records, Blob<float>* blob) override;
  int partition_dim() const override{   //Need to return 0
    return 0;
  }

  int vocabsize() const {
    return vocabsize_;
  }

  ~RnnlmWordparserLayer();

 private:
  int windowsize_;
  int vocabsize_;
};

/**
  * 6-RnnlmClassparserLayer
  */
class RnnlmClassparserLayer: public ParserLayer {
 public:
  using ParserLayer::ParseRecords;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ParseRecords(Phase phase, const vector<Record>& records, Blob<float>* blob) override;

  int partition_dim() const override{   //Need to return 0
    return 0;
  }

  int classsize() const {
    return classsize_;
  }

  int vocabsize() const {
    return vocabsize_;
  }

  ~RnnlmClassparserLayer();
 private:
  int windowsize_;
  int vocabsize_;
  int classsize_;
};

/**
  * 7-RnnlmDataLayer
  */
class RnnlmDataLayer: public DataLayer{
 public:
  using Layer::ComputeFeature;
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  //~RnnlmDataLayer();    //No need to write destructor function explicitly
  int windowsize() const {
    return windowsize_;
  }

  int classsize() const {
    return classsize_;
  }

  int vocabsize() const {
    return vocabsize_;
  }

  Blob<int> * classinfo(){
    return & classinfo_;
  }

 private:
  int windowsize_;  //there is a field "batchsize_" as a member of DataLayer
  shared_ptr<DataShard> classshard_;
  shared_ptr<DataShard> wordshard_;
  int classsize_;
  int vocabsize_;
  Blob<int> classinfo_; //For each class index, info: start vocab_index, end vocab_index
  vector<Record> classrecords_; //used for processing class records; The member "records_" in DataLayer is specified for use of WordRecords here
};

/********** end line for RNNLM example **********/

class LabelLayer: public ParserLayer {
 public:
  using ParserLayer::ParseRecords;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ParseRecords(Phase phase, const vector<Record>& records,
      Blob<float>* blob) override;
};

class LRNLayer: public Layer {
/**
 * Local Response Normalization edge
 * b_i=a_i/x_i^beta
 * x_i=knorm+alpha*\sum_{j=max(0,i-n/2}^{min(N,i+n/2}(a_j)^2
 * n is size of local response area.
 * a_i, the activation (after ReLU) of a neuron convolved with the i-th kernel.
 * b_i, the neuron after normalization, N is the total num of kernels
 */
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase) override;

 protected:
  //! shape of the bottom layer feature
  int batchsize_, channels_, height_, width_;
  //! size local response (neighbor) area
  int lsize_;
  //! hyper-parameter
  float alpha_, beta_, knorm_;
  Blob<float> norm_;
};

class MnistLayer: public ParserLayer {
 public:
  using ParserLayer::ParseRecords;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ParseRecords(Phase phase, const vector<Record>& records,
      Blob<float>* blob) override;

 protected:
  // height and width of the image after deformation
  // kernel size for elastic distortion
  // n^2 images are processed as a batch for elastic distortion
  // conv height and conv width
  // gauss kernel values, displacements, column image and tmp buffer
  // float* gauss_, *displacementx_, *displacementy_, *colimg_, *tmpimg_;
  float  gamma_, beta_, sigma_, kernel_, alpha_, norm_a_, norm_b_;
  int resize_, elastic_freq_;
};

class PoolingLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase) override;

 protected:
  int kernel_, pad_, stride_;
  int batchsize_, channels_, height_, width_, pooled_height_, pooled_width_;
  PoolingProto_PoolMethod pool_;
};

class ReLULayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions = 1) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase) override;
};


class SoftmaxLossLayer: public LossLayer {
  /*
   * connected from the label layer and the last fc layer
   */
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase) override;

  /**
   * softmax is not recommendeded for partition because it requires the whole
   * src layer for normalization.
   */
  int partition_dim() const override {
    CHECK_LE(layer_proto_.partition_dim(), 1);
    return layer_proto_.partition_dim();
  }
  ConnectionType src_neuron_connection(int k) const override {
    // CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }

 private:
  int batchsize_;
  int dim_;
  float scale_;
  int topk_;
};

class RGBImageLayer: public ParserLayer {
 public:
  using ParserLayer::ParseRecords;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ParseRecords(Phase phase, const vector<Record>& records,
      Blob<float>* blob) override;

 private:
  float scale_;
  int cropsize_;
  bool mirror_;
  Blob<float> mean_;
};

class ShardDataLayer: public DataLayer{
 public:
  using Layer::ComputeFeature;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
 private:
  shared_ptr<DataShard> shard_;
};

/**
 * This layer apply Tan function to neuron activations.
 * f(x)=A tanh(Bx)
 * f'(x)=B/A (A*A-f(x)*f(x))
 */
class TanhLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase) override;

 private:
  float outer_scale_, inner_scale_;
};


}  // namespace singa

#endif  // SINGA_NEURALNET_LAYER_H_
