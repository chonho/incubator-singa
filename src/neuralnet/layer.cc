#include <glog/logging.h>
#include <memory>
#include <algorithm>
#include "mshadow/tensor.h"
#include "mshadow/cxxnet_op.h"
#include "neuralnet/layer.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "../../include/neuralnet/layer.h"

using namespace mshadow;
using namespace mshadow::expr;

namespace singa {
inline Tensor<cpu, 4> Tensor4(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 4> tensor(blob->mutable_cpu_data(),
      Shape4(shape[0], shape[1], shape[2], shape[3]));
  return tensor;
}

inline Tensor<cpu, 3> Tensor3(Blob<float>* blob){
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 3> tensor(blob->mutable_cpu_data(),
      Shape3(shape[0], shape[1], blob->count() / shape[0] / shape[1]));
  return tensor;
}
inline Tensor<cpu, 2> Tensor2(Blob<float>* blob){
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 2> tensor(blob->mutable_cpu_data(),
      Shape2(shape[0], blob->count() / shape[0]));
  return tensor;
}
inline Tensor<cpu, 1> Tensor1(Blob<float>* blob){
  Tensor<cpu, 1> tensor(blob->mutable_cpu_data(), Shape1(blob->count()));
  return tensor;
}

/************ Implementation for ConvProductLayer*************************/
ConvolutionLayer::~ConvolutionLayer() {
  delete weight_;
  delete bias_;
}
void ConvolutionLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  ConvolutionProto conv_conf=proto.convolution_conf();
  kernel_=conv_conf.kernel();
  CHECK_GT(kernel_, 0) << "Filter size cannot be zero.";
  pad_=conv_conf.pad();
  stride_=conv_conf.stride();
  num_filters_=conv_conf.num_filters();
  if(partition_dim() > 0)
    num_filters_ /= npartitions;

  const vector<int>& srcshape=srclayers_[0]->data(this).shape();
  int dim=srcshape.size();
  CHECK_GT(dim, 2);
  width_=srcshape[dim-1];
  height_=srcshape[dim-2];
  if(dim>3)
    channels_=srcshape[dim-3];
  else if(dim>2)
    channels_=1;
  batchsize_=srcshape[0];
  conv_height_=(height_ + 2 * pad_ - kernel_) / stride_ + 1;
  conv_width_= (width_ + 2 * pad_ - kernel_) / stride_ + 1;
  col_height_=channels_*kernel_*kernel_;
  col_width_=conv_height_*conv_width_;
  vector<int> shape{batchsize_, num_filters_, conv_height_, conv_width_};
  data_.Reshape(shape);
  grad_.Reshape(shape);
  col_data_.Reshape(vector<int>{col_height_, col_width_});
  col_grad_.Reshape(vector<int>{col_height_, col_width_});

  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  weight_ = factory->Create("Param");
  weight_->Setup(proto.param(0), vector<int>{num_filters_, col_height_});
  bias_ = factory->Create("Param");
  bias_->Setup(proto.param(1), vector<int>{num_filters_});
}

void ConvolutionLayer::ComputeFeature(Phase phase, Metric* perf){
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor3(&data_);
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());

  for(int n=0;n<batchsize_;n++){
    if(pad_>0)
      col=unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col=unpack_patch2col(src[n], kernel_, stride_);
    data[n]=dot(weight, col);
  }
  data+=broadcast<1>(bias, data.shape);
}

void ConvolutionLayer::ComputeGradient(Phase phase) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());

  auto grad = Tensor3(&grad_);
  auto gcol = Tensor2(&col_grad_);
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());

  Blob<float>* gsrcblob=srclayers_[0]->mutable_grad(this);
  Tensor<cpu, 4> gsrc(nullptr, Shape4(batchsize_, channels_, height_, width_));
  if(gsrcblob!=nullptr)
    gsrc.dptr=gsrcblob->mutable_cpu_data();
  gbias=sumall_except_dim<1>(grad);

  gweight = 0.0f;
  Shape<3> padshp(gsrc.shape.SubShape());
  padshp[0] += 2 * pad_;
  padshp[1] += 2 * pad_;
  Shape<2> imgshp = Shape2(height_, width_);
  for(int n=0;n<batchsize_;n++){
    if(pad_>0)
      col=unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col=unpack_patch2col(src[n], kernel_, stride_);
    gweight += dot(grad[n], col.T());

    if(gsrcblob!=nullptr){
      gcol = dot(weight.T(), grad[n]);
      gsrc[n] = crop(pack_col2patch(gcol, padshp, kernel_, stride_), imgshp);
    }
  }
}

/****************** Implementation for DropoutLayer ***********************/
void DropoutLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(*srclayers_[0]->mutable_grad(this));
  mask_.Reshape(srclayers_[0]->data(this).shape());
  pdrop_ = proto.dropout_conf().dropout_ratio();
}

void DropoutLayer::ComputeFeature(Phase phase, Metric* perf) {
  // check training
  if(phase != kTrain){//!training){
    data_.CopyFrom(srclayers_[0]->data(this));
    return;
  }
  float pkeep=1-pdrop_;
  auto mask = Tensor1(&mask_);
  mask = F<op::threshold>(TSingleton<Random<cpu>>::Instance()\
      ->uniform(mask.shape), pkeep ) * (1.0f/pkeep);
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data = src * mask;
}

void DropoutLayer::ComputeGradient(Phase phase)  {
  auto mask = Tensor1(&mask_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc = grad * mask;
}
/**************** Implementation for RBMVisLayer********************/
RBMVisLayer::~RBMVisLayer() {
  delete weight_;
  delete bias_;
}
void RBMVisLayer::Setup(const LayerProto& proto,
      int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 2);
  // hid_idx_: index indicating which srclayer is is hidden layer
  // data_idx_: index indicating which srclayer is data layer
  for (unsigned int i = 0; i < srclayers_.size(); i++)
    for (unsigned int j = 0; j < (srclayers_[i]-> dstlayers()).size(); j++)
      if (strcmp(((srclayers_[i]->dstlayers()).at(j)->name().c_str()),
        (this->name()).c_str()) == 0)
        hid_idx_ = i;
  for (unsigned int i = 0; i < srclayers_.size(); i++)
    if (i != static_cast<unsigned int>(hid_idx_) )
      data_idx_ = i;
  const auto& src = srclayers_[data_idx_]->data(this);
  is_first_iteration_vis_ = true;
  batchsize_ = src.shape()[0];
  neg_batchsize_ = batchsize_;
  /*gibbs sampling size and input have the same size*/
  vdim_ = src.count()/batchsize_;
  hdim_ = proto.rbmvis_conf().num_output();
  data_.Reshape(vector<int>{batchsize_, vdim_});  // this is visible dimension
  vis_sample_.Reshape(vector<int>{neg_batchsize_, vdim_});
  Factory<Param>* factory = Singleton<Factory<Param>>::Instance();
  weight_ = factory->Create("Param");
  bias_ = factory->Create("Param");
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});
  bias_->Setup(proto.param(1), vector<int>{vdim_});
}

void RBMVisLayer::ComputeFeature(Phase phase, Metric* perf) {
  if (phase == kPositive) { /*positive phase*/
    auto data = Tensor2(&data_);
    CHECK_EQ(srclayers_[data_idx_]->data(this).count(), batchsize_*vdim_);
    auto src = Tensor2(srclayers_[data_idx_]->mutable_data(this));
    Copy(data, src);
  } else if (phase == kNegative) {   /*negative phase*/
      if (is_first_iteration_vis_) {
        CHECK_EQ(srclayers_[data_idx_]->data(this).count(), batchsize_*vdim_);
        auto src = Tensor2(srclayers_[data_idx_]->mutable_data(this));
        auto vis_sample = Tensor2(&vis_sample_);
        Copy(vis_sample, src);
        is_first_iteration_vis_ = false;
      } else {
          auto hid_sample =
                Tensor2(srclayers_[hid_idx_]->mutable_data(this, kNegative));
          // fetch sampling results from hidden layer
          auto vis_sample = Tensor2(&vis_sample_);
          auto weight = Tensor2(weight_->mutable_data());
          auto bias = Tensor1(bias_->mutable_data());
          vis_sample = dot(hid_sample, weight.T());
          vis_sample+=repmat(bias, neg_batchsize_);
          vis_sample = F<op::sigmoid>(vis_sample);
          TSingleton<Random<cpu>>::Instance()->SampleBinary(vis_sample);
        }
    }
}

void RBMVisLayer::ComputeGradient(Phase phase) {
  auto data = Tensor2(&data_);
  auto hid_data = Tensor2(srclayers_[hid_idx_]->mutable_data(this, kPositive));
  auto vis_sample = Tensor2(&vis_sample_);
  auto hid_sample =
       Tensor2(srclayers_[hid_idx_]->mutable_data(this, kNegative));
  // fetch sampling results from hidden layer
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());
  gbias = sum_rows(vis_sample);
  gbias -= sum_rows(data);
  gweight = dot(vis_sample.T(), hid_sample);
  gweight -= dot(data.T(), hid_data);
  gbias*=(1.0f)/(1.0f*batchsize_);
  gweight*=(1.0f)/(1.0f*batchsize_);
}

void RBMVisLayer::ComputeLoss(Metric* perf) {
  float loss = (0.0f);
  CHECK_EQ(srclayers_[data_idx_]->data(this).count(), batchsize_*vdim_);
  auto src = Tensor2(srclayers_[data_idx_]->mutable_data(this));
  auto hid_data = Tensor2(srclayers_[hid_idx_]->mutable_data(this, kPositive));
  // gibbs using u
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());
  Tensor<cpu, 2> reconstruct(Shape2(batchsize_, vdim_)); /*reconstruct error*/
  AllocSpace(reconstruct);
  reconstruct = dot(hid_data, weight.T());
  reconstruct+=repmat(bias, batchsize_);
  reconstruct = F<op::sigmoid>(reconstruct);
  float *src_dptr = src.dptr;
  float *reconstruct_dptr = reconstruct.dptr;
  for (int i = 0; i < vdim_*batchsize_; i++)
    loss += -(src_dptr[i]*log(reconstruct_dptr[i])
            +(1-src_dptr[i])*log(1-reconstruct_dptr[i]));
  loss/=batchsize_;
  FreeSpace(reconstruct);
  perf->Reset();
  perf->Add("reconstruct_error", loss);
}
/**************** Implementation for RBMHidLayer********************/
RBMHidLayer::~RBMHidLayer() {
  delete weight_;
  delete bias_;
}
void RBMHidLayer::Setup(const LayerProto& proto,
      int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src_data = srclayers_[0]->data(this, kPositive);
  const auto& src_sample = srclayers_[0]->data(this, kNegative);
  scale_ = static_cast<float> (1.0f);
  batchsize_ = src_data.shape()[0];
  neg_batchsize_ = src_sample.shape()[0];
  vdim_ = src_data.count()/batchsize_;
  hdim_ = proto.rbmhid_conf().hid_dim();
  data_.Reshape(vector<int>{batchsize_, hdim_});
  hid_sample_.Reshape(vector<int>{neg_batchsize_, hdim_});
  Factory<Param>* factory = Singleton<Factory<Param>>::Instance();
  bias_ = factory->Create("Param");
  weight_ = factory->Create("Param");
  bias_->Setup(proto.param(1), vector<int>{hdim_});
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});
}

void RBMHidLayer::ComputeFeature(Phase phase, Metric* perf) {
  if (phase == kPositive) {  /*postive phase*/
    auto data = Tensor2(&data_);
    CHECK_EQ(srclayers_[0]->data(this, kPositive).count(), batchsize_*vdim_);
    auto src = Tensor2(srclayers_[0]->mutable_data(this, kPositive));
    auto weight = Tensor2(weight_->mutable_data());
    auto bias = Tensor1(bias_->mutable_data());
    data = dot(src, weight);
    data += repmat(bias, batchsize_);
    data = F<op::sigmoid>(data);
  } else if (phase == kNegative) {   /*negative phase*/
      CHECK_EQ(srclayers_[0]->data(this, kNegative).count(),
         neg_batchsize_*vdim_);
      auto src_sample = Tensor2(srclayers_[0]->mutable_data(this, kNegative));
      auto hid_sample = Tensor2(&hid_sample_);
      auto bias = Tensor1(bias_->mutable_data());
      auto weight = Tensor2(weight_->mutable_data());
      hid_sample = dot(src_sample, weight);
      hid_sample += repmat(bias, neg_batchsize_);
      hid_sample = F<op::sigmoid>(hid_sample);
      TSingleton<Random<cpu>>::Instance()->SampleBinary(hid_sample);
    } else if (phase == kLoss) {   /*test phase*/
       auto data = Tensor2(&data_);  // data: sigmoid(Wv+b)
       TSingleton<Random<cpu>>::Instance()->SampleBinary(data);
      }
}
void RBMHidLayer::ComputeGradient(Phase phase) {
  auto data = Tensor2(&data_);
  auto hid_sample = Tensor2(&hid_sample_);
  auto gbias = Tensor1(bias_->mutable_grad());
  gbias = sum_rows(hid_sample);
  gbias -= sum_rows(data);
  gbias *= scale_/(1.0f*batchsize_);
}
/*********** Implementation for InnerProductLayer**********/
InnerProductLayer::~InnerProductLayer() {
  delete weight_;
  delete bias_;
}
void InnerProductLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src=srclayers_[0]->data(this);
  batchsize_=src.shape()[0];
  vdim_=src.count()/batchsize_;
  hdim_=proto.innerproduct_conf().num_output();
  if(partition_dim()>0)
    hdim_ /= npartitions;
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  weight_ = factory->Create("Param");
  bias_ = factory->Create("Param");
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});
  bias_->Setup(proto.param(1), vector<int>{hdim_});
}

void InnerProductLayer::ComputeFeature(Phase phase, Metric* perf) {
  auto data = Tensor2(&data_);
  auto src = Tensor2(srclayers_[0]->mutable_data(this));
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());
  data=dot(src, weight);
  // repmat: repeat bias vector into batchsize rows
  data+=repmat(bias, batchsize_);
}

void InnerProductLayer::ComputeGradient(Phase phas) {
  auto src = Tensor2(srclayers_[0]->mutable_data(this));
  auto grad = Tensor2(&grad_);
  auto weight = Tensor2(weight_->mutable_data());
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());

  gbias=sum_rows(grad);
  gweight=dot(src.T(), grad);
  if(srclayers_[0]->mutable_grad(this)!=nullptr){
    auto gsrc = Tensor2(srclayers_[0]->mutable_grad(this));
    gsrc=dot(grad, weight.T());
  }
}



/***********  Implementing layers used in RNNLM application ***********/
/*********** 1-Implementation for RnnlmComputationLayer **********/
RnnlmComputationLayer::~RnnlmComputationLayer() {
    delete weight_;
}

void RnnlmComputationLayer::Setup(const LayerProto& proto, int npartitions) {
    Layer::Setup(proto, npartitions);
    CHECK_EQ(srclayers_.size(), 2); //RnnlmComputationLayer has 2 src layers, 1st one: SigmoidLayer, 2nd one: ClassParser
    const auto& sigmoidData = srclayers_[0]->data(this);
    //const auto& labelData = srclayers_[1]->data(this);  //The order of src layers are due to conf order; labelData has the shape (windowsize_,4)
    windowsize_= sigmoidData.shape()[0];
    vdim_ = sigmoidData.count()/windowsize_;   //e.g, 30; dimension of input
    classsize_ = static_cast<RnnlmClassparserLayer*>(srclayers_[1])->classsize(); //10 here, use type casting
    vocabsize_ = static_cast<RnnlmClassparserLayer*>(srclayers_[1])->vocabsize(); //10000 here, use type casting
    hdim_ = classsize_ + vocabsize_; //e.g, 10010 if VocabSize=10000, ClassSize=10; TODO implement getVocabSize() and getClassSize() on LabelLayer
    data_.Reshape(vector<int>{windowsize_, hdim_});
    grad_.ReshapeLike(data_);
    Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
    weight_ = factory->Create("Param");
    weight_->Setup(proto.param(0), vector<int>{hdim_, vdim_});  // (10010, 30)Need to transpose the original weight matrix for the slicing part
    //sum_ = 0.0; // Initialize the accuracy value; used to store the sum of log(pi), i.e., sum of log(p1 * p2)
}

void RnnlmComputationLayer::ComputeFeature(Phase phase, Metric* perf) {
    auto data = Tensor2(&data_);    //(window_size, 10010)
    auto sigmoidData = Tensor2(srclayers_[0]->mutable_data(this));  //mutable_data means data with a variable length
    const float * label = srclayers_[1]->data(this).cpu_data(); //The shape should be (windowsize_,4); a quadruple: start and end vocabulary index for the ground truth class; then the word index in vocabulary; then finally the class for the input word
    auto weight = Tensor2(weight_->mutable_data());

    auto weightPart1 = weight.Slice(0, classsize_);  //slice is [) form (10, 30), the slicing operation is by row
    auto weightPart2 = weight.Slice(classsize_, classsize_ + vocabsize_);  //(10000, 30), the slicing operation is by row

    //Compute y1(t), y2(t), then copy to data of RnnlmComputationLayer
    for(int t = 0; t < windowsize_; t++){
        int startVocabIndex = static_cast<int>(label[t * 4 + 0]);
        int endVocabIndex = static_cast<int>(label[t * 4 + 1]);
        //int wordIndex = static_cast<int>(label[t * 4 + 2]); //ground truth word index
        //int classIndex = static_cast<int>(label[t * 4 + 3]);    //ground truth class index

        auto weightPart2Slice = weightPart2.Slice(startVocabIndex, endVocabIndex + 1);  //closed [start, end]
        Tensor<cpu, 1> y1(data.dptr + hdim_ * t, Shape1(classsize_));    //hdim_ = classsize_ + vocabsize_
        y1 = dot(sigmoidData[t], weightPart1.T());  //TODO kaiping (ddim, ldim, rdim) = (1, 1, 2)
        Tensor<cpu, 1> y2(data.dptr + hdim_ * t + classsize_ + startVocabIndex, Shape1(endVocabIndex - startVocabIndex + 1));
        y2 = dot(sigmoidData[t], weightPart2Slice.T()); // Directly modify the value of "data" - TODO kaiping (ddim, ldim, rdim) = (1, 1, 2)
    }

    float sum = 0.0;
    float ppl = 0.0;
    //Compute p1(t), p2(t) using the computed value of y1 and y2 and then copy to the "data" of ComputationLayer; Additionally compute the sum_ value
    for(int t = 0; t < windowsize_; t++){
        int startVocabIndex = static_cast<int>(label[t * 4 + 0]);
        int endVocabIndex = static_cast<int>(label[t * 4 + 1]);
        int wordIndex = static_cast<int>(label[t * 4 + 2]); //ground truth word index
        int classIndex = static_cast<int>(label[t * 4 + 3]);    //ground truth class index

        Tensor<cpu, 1> p1(nullptr, Shape1(classsize_));
        AllocSpace(p1); //Allocate the space for p1; after allocating the space, must free the space
        Tensor<cpu, 1> p2(nullptr, Shape1(endVocabIndex - startVocabIndex + 1));
        AllocSpace(p2); //Allocate the space for p2

        Tensor<cpu, 1> tmp1(data.dptr + hdim_ * t, Shape1(classsize_));
        Tensor<cpu, 1> tmp2(data.dptr + hdim_ * t + classsize_ + startVocabIndex, Shape1(endVocabIndex - startVocabIndex + 1));

        Softmax(p1, tmp1);
        Softmax(p2, tmp2);   //In Softmax(), tmp1 and tmp2 are not changed

        //Then copy p1[t] and p2[t] to "data"
        memcpy(data[t].dptr, p1.dptr, sizeof(float) * classsize_);
        memcpy(data[t].dptr + classsize_ + startVocabIndex, p2.dptr, sizeof(float) * (endVocabIndex - startVocabIndex + 1));

        //For each word respectively, add a term in the sum_
        //sum_ += log(p1[classIndex] * p2[wordIndex - startVocabIndex]);
        sum += log(std::max(p1[classIndex] * p2[wordIndex - startVocabIndex], FLT_MIN));
        LOG(ERROR) << "SUM: " << sum;
        FreeSpace(p1);
        FreeSpace(p2);
    }
    LOG(ERROR) << "SUM: " << sum;
    ppl = exp(-(1.0 / windowsize_) * sum); //per word
    LOG(ERROR) << "PPL: " << ppl;
    //perf->Reset();
    perf->Add("sum value", sum);
    perf->Add("ppl value", ppl);
    LOG(ERROR) << perf->ToLogString();
    LOG(ERROR) << "----------This is computation layer----------";
    LOG(ERROR) << "Forward Phase: Sum value of data: " << data_.asum_data();
    LOG(ERROR) << "Forward Phase: Sum value of grad: " << grad_.asum_data();
}

void RnnlmComputationLayer::ComputeGradient(Phase phase){
    LOG(ERROR) << "***********************************************";
    //auto data = Tensor2(&data_);    //(win_size, 10010)
    Blob<float> *data_dptr = &data_; //(win_size, 10010)
    float *data_dptr_tmp = data_dptr->mutable_cpu_data();
    //auto grad = Tensor2(&grad_);    //(win_size, 10010)
    Blob<float> *grad_ptr = &grad_;  //(win_size, 10010)
    float *grad_ptr_tmp = grad_ptr->mutable_cpu_data();
    //auto src = Tensor2(srclayers_[0]->mutable_data(this));
    Blob<float> *src_ptr = srclayers_[0]->mutable_data(this);
    float *src_ptr_tmp = src_ptr->mutable_cpu_data();
    const float * label = srclayers_[1]->data(this).cpu_data();   //offer the ground truth info

    auto gweight = Tensor2(weight_->mutable_grad());    //the gradient for the parameter: weight matrix
    auto gweightPart1 = gweight.Slice(0, classsize_);  //slice is [) form (10, 30), the slicing operation is by row
    auto gweightPart2 = gweight.Slice(classsize_, classsize_ + vocabsize_);  //(10000, 30), the slicing operation is by row

    auto weight = Tensor2(weight_->mutable_data());
    auto weightPart1 = weight.Slice(0, classsize_);  //(10, 30), the slicing operation is by row
    auto weightPart2 = weight.Slice(classsize_, classsize_ + vocabsize_);  //(10000, 30), the slicing operation is by row

    if(srclayers_[0]->mutable_grad(this) != nullptr) {
        auto gsrc = Tensor2(srclayers_[0]->mutable_grad(this)); //(10,30), i.e., (window_size, 30)


        memset(gweight.dptr, 0, sizeof(float) * gweight.shape[0] *
                                                gweight.shape[1]);   //Need initialization before aggregate updates in all timestamps

        for (int t = 0; t < windowsize_; t++) {
            //Obtain ground truth information
            int startVocabIndex = static_cast<int>(label[t * 4 + 0]);
            int endVocabIndex = static_cast<int>(label[t * 4 + 1]);
            int wordIndex = static_cast<int>(label[t * 4 + 2]); //ground truth word index
            int classIndex = static_cast<int>(label[t * 4 + 3]);    //ground truth class index

            auto gweightPart2Slice = gweightPart2.Slice(startVocabIndex, endVocabIndex +
                                                                         1);    //e.g, (150, 30), set # of words in ground truth class is 150
            auto weightPart2Slice = weightPart2.Slice(startVocabIndex, endVocabIndex +
                                                                       1);    //e.g, (150, 30), set # of words in ground truth class is 150

            //Compute the gradient for the current layer
            //To check later: can compute values for one t and then back propagate the error/gradient?
            for (int i = 0; i < classsize_; i++) {  //TODO kaiping change or not?
                //grad[t][i] = 0 - data[t][i];
                grad_ptr_tmp[t * hdim_ + i] = 0 - data_dptr_tmp[t * hdim_ + i];

            }
            //grad[t][classIndex] = 1 - data[t][classIndex];  //Compute ground truth for the class
            grad_ptr_tmp[t * hdim_ + classIndex] = 1 - data_dptr_tmp[t * hdim_ + classIndex];   //Compute ground truth for the class

            for (int j = classsize_; j < classsize_ + vocabsize_; j++) {    //TODO kaiping: to change later, first initialize to 0 and then loop in [startVocabIndex, endVocabIndex]
                if (j >= (classsize_ + startVocabIndex) && j <= (classsize_ + endVocabIndex)) {
                    //grad[t][j] = 0 - data[t][j];
                    grad_ptr_tmp[t * hdim_ + j] = 0 - data_dptr_tmp[t * hdim_ + j];
                }
                else {
                    //grad[t][j] = 0;
                    grad_ptr_tmp[t * hdim_ + j] = 0;
                }
                //grad[t][classsize_ + wordIndex] = 1 - data[t][classsize_ + wordIndex];  //Compute ground truth for the word
                grad_ptr_tmp[t * hdim_ + classsize_ + wordIndex] = 1 - data_dptr_tmp[t * hdim_ + classsize_ + wordIndex];   //Compute ground truth for the word
            }

            //Compute the gradient for the weight matrix, the loop is for various timestamps T
            //Tensor <cpu, 2> gradPart1(grad[t].dptr, Shape2(classsize_, 1));   //(10,1)
            Tensor <cpu, 2> gradPart1(grad_ptr_tmp + t * hdim_, Shape2(classsize_, 1));   //(10,1)
            //Tensor <cpu, 2> src_t(src[t].dptr, Shape2(1, vdim_));   //(1, 30)
            Tensor <cpu, 2> src_t(src_ptr_tmp + t * vdim_, Shape2(1, vdim_));   //(1, 30)
            gweightPart1 += dot(gradPart1, src_t);  //aggregate all updates for this weight matrix together    //TODO kaiping (ddim, ldim, rdim) = (2, 2, 1) -> (2, 2, 2)
            //Tensor <cpu, 2> gradPart2Slice(grad[t].dptr + classsize_ + startVocabIndex, Shape2(endVocabIndex - startVocabIndex + 1, 1));
            Tensor <cpu, 2> gradPart2Slice(grad_ptr_tmp + t * hdim_ + classsize_ + startVocabIndex, Shape2(endVocabIndex - startVocabIndex + 1, 1));
            gweightPart2Slice += dot(gradPart2Slice, src_t);    //TODO kaiping (ddim, ldim, rdim) = (2, 2, 1) -> (2, 2, 2)

            //Compute the gradient for the src layer, the loop is for various timestamps T; actually another part of gsrc will be added in RnnSigmoidLayer
            //Tensor <cpu, 1> gradPart1ForSrc(grad[t].dptr, Shape1(classsize_));   //(1,10)
            Tensor <cpu, 1> gradPart1ForSrc(grad_ptr_tmp + t * hdim_, Shape1(classsize_));   //(1,10)
            //Tensor <cpu, 1> gradPart2SliceForSrc(grad[t].dptr + classsize_ + startVocabIndex, Shape1(endVocabIndex - startVocabIndex + 1));  //(1,150)
            Tensor <cpu, 1> gradPart2SliceForSrc(grad_ptr_tmp + t * hdim_ + classsize_ + startVocabIndex, Shape1(endVocabIndex - startVocabIndex + 1));  //(1,150)
            //gsrc[t] = dot(gradPart1ForSrc, weightPart1) + dot(gradPart2SliceForSrc, weightPart2Slice);
            gsrc[t] = dot(gradPart1ForSrc, weightPart1);    //TODO kaiping (ddim, ldim, rdim) = (1, 1, 2)
            gsrc[t] += dot(gradPart2SliceForSrc, weightPart2Slice); //TODO kaiping (ddim, ldim, rdim) = (1, 1, 2)
        }
    }
    LOG(ERROR) << "----------This is computation layer----------";
    LOG(ERROR) << "Backward Phase: Sum value of weight data: " << weight_->mutable_data()->asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of weight grad: " << weight_->mutable_grad()->asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of data: " << data_.asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of grad: " << grad_.asum_data();
}


/*********** 2-Implementation for RnnlmSigmoidLayer **********/
//This layer is like a combination of InnerProductLayer and ActivationLayer
RnnlmSigmoidLayer::~RnnlmSigmoidLayer() {
    delete weight_;
}

void RnnlmSigmoidLayer::Setup(const LayerProto& proto, int npartitions) {
    Layer::Setup(proto, npartitions);
    CHECK_EQ(srclayers_.size(), 1); //RnnlmSigmoidLayer has 1 src layers: RnnlmInnerproductLayer
    const auto& innerproductData = srclayers_[0]->data(this);
    windowsize_= innerproductData.shape()[0];
    vdim_ = innerproductData.count()/windowsize_;   //e.g, 30; dimension of input
    hdim_ = vdim_;  //e.g, 30; dimension of output
    data_.ReshapeLike(srclayers_[0]->data(this));
    grad_.ReshapeLike(srclayers_[0]->grad(this));
    Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
    weight_ = factory->Create("Param");
    weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});  // (30, 30) weight matrix between s(t-1) and s(t)
}

void RnnlmSigmoidLayer::ComputeFeature(Phase phase, Metric* perf) {
    auto data = Tensor2(&data_);
    auto src = Tensor2(srclayers_[0]->mutable_data(this)); //Shape for src is (window_size, 30)
    auto weight = Tensor2(weight_->mutable_data());
    //First compute the s(t-1) * W part, then add the sigmoid part of input
    for(int t = 0; t < windowsize_; t++){   //Skip the 1st component
        if(t == 0){
            data[t] = F<op::sigmoid>(src[t]);
        }
        else{
            //data[t] = dot(data[t - 1], weight) + F<op::sigmoid>(src[t]);
            data[t] = dot(data[t - 1], weight); //TODO kaiping (ddim, ldim, rdim) = (1, 1, 2)
            data[t] += F<op::sigmoid>(src[t]);
        }
    }
    LOG(ERROR) << "----------This is sigmoid layer----------";
    LOG(ERROR) << "Forward Phase: Sum value of data: " << data_.asum_data();
    LOG(ERROR) << "Forward Phase: Sum value of grad: " << grad_.asum_data();
}

void RnnlmSigmoidLayer::ComputeGradient(Phase phase){
    auto data = Tensor2(&data_);    //(win_size, 30)
    auto grad = Tensor2(&grad_);    //(win_size, 30)
    auto weight = Tensor2(weight_->mutable_data()); //(30,30)
    auto gweight = Tensor2(weight_->mutable_grad());    //the gradient for the parameter: weight matrix
    if(srclayers_[0]->mutable_grad(this) != nullptr) {
        auto gsrc = Tensor2(srclayers_[0]->mutable_grad(this)); //(10,30), i.e., (window_size, 30)
        memset(gweight.dptr, 0, sizeof(float) * gweight.shape[0] *
                                                gweight.shape[1]);   //Need initialization before aggregate updates in all timestamps
        //1-Update the gradient for the current layer, add a new term
        for (int t = windowsize_ - 2; t >= 0; t--) {   //grad[windowsize_ - 1] does not have this term
            grad[t] += dot(grad[t + 1], weight);    //TODO kaiping (ddim, ldim, rdim) = (1, 1, 2)
        }

        //2-Compute the gradient for the weight matrix; 3-Compute the gradient for src layer; the loop is for various timestamps T
        for (int t = 0; t < windowsize_; t++) {
            if (t == 0) {
                gsrc[t] = F<op::sigmoid_grad>(data[t]) *
                          grad[t];  //?here F<op::sigmoid_grad>(data) is a scalar value; make sure to use the final value of grad(t)
            }
            else {
                Tensor <cpu, 2> data_t_minus1(data[t - 1].dptr, Shape2(hdim_, 1));   //(30, 1)
                Tensor <cpu, 2> grad_t_trans(grad[t].dptr, Shape2(1, hdim_));   //(1,30)
                gweight += dot(data_t_minus1, grad_t_trans);   //TODO kaiping (ddim, ldim, rdim) = (2, 2, 1) -> (2, 2, 2)
                gsrc[t] = F<op::sigmoid_grad>(data[t]) * grad[t];  //TODO here F<op::sigmoid_grad>(data) is a scalar value
            }
        }
    }
    LOG(ERROR) << "----------This is sigmoid layer----------";
    LOG(ERROR) << "Backward Phase: Sum value of weight data: " << weight_->mutable_data()->asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of weight grad: " << weight_->mutable_grad()->asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of data: " << data_.asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of grad: " << grad_.asum_data();
}


/*********** 3-Implementation for RnnlmInnerproductLayer **********/
//The only difference between this layer type and ordinary InnerProductLayer is the consideration of window_size, i.e., time
RnnlmInnerproductLayer::~RnnlmInnerproductLayer() {
  delete weight_;
}

void RnnlmInnerproductLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src=srclayers_[0]->data(this);
  windowsize_=src.shape()[0];
  vdim_=src.count()/windowsize_; //dimension of input, i.e., |V|
  //hdim_=proto.rnnlminnerproduct_conf().num_output();
  hdim_=proto.GetExtension(rnnlminnerproduct_conf).num_output();
  data_.Reshape(vector<int>{windowsize_, hdim_});   //(win_size,30)
  grad_.ReshapeLike(data_);
  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  weight_ = factory->Create("Param");
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});    //(|V|,30)
}

void RnnlmInnerproductLayer::ComputeFeature(Phase phase, Metric* perf) {
  auto data = Tensor2(&data_);  //(window_size, 30)
  auto src = Tensor2(srclayers_[0]->mutable_data(this));    //(window_size, |V|)
  auto weight = Tensor2(weight_->mutable_data());           //(|V|, 30)
    data = dot(src, weight);
    LOG(ERROR) << "----------This is inner product layer----------";
    LOG(ERROR) << "Backward Phase: Sum value of data: " << data_.asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of grad: " << grad_.asum_data();
}

void RnnlmInnerproductLayer::ComputeGradient(Phase phas) {
    auto grad = Tensor2(&grad_);    //(win_size, 30)
    auto weight = Tensor2(weight_->mutable_data()); //(|V|,30)
    auto gweight = Tensor2(weight_->mutable_grad());    //the gradient for the parameter: weight matrix
    //auto src = Tensor2(srclayers_[0]->mutable_data(this)); //Shape for src is (window_size, 30)
    Blob<float> *src_ptr = srclayers_[0]->mutable_data(this);
    float *src_ptr_tmp = src_ptr->mutable_cpu_data();
    if(srclayers_[0]->mutable_grad(this) != nullptr) {   //Why have to check this?
        auto gsrc = Tensor2(srclayers_[0]->mutable_grad(this)); //(10,|V|), i.e., (window_size, |V|)

        memset(gweight.dptr, 0, sizeof(float) * gweight.shape[0] *
                                                gweight.shape[1]);   //Need initialization before aggregate updates in all timestamps

        //2-Compute the gradient for the weight matrix; 3-Compute the gradient for src layer;
        for (int t = 0; t < windowsize_; t++) {
            //Tensor <cpu, 2> src_t_trans(src[t].dptr, Shape2(vdim_, 1));   //(|V|,1)
            Tensor <cpu, 2> src_t_trans(src_ptr_tmp + t * hdim_, Shape2(vdim_, 1));   //(|V|,1)
            Tensor <cpu, 2> grad_t(grad[t].dptr, Shape2(1, hdim_));   //(1,30)
            gweight += dot(src_t_trans, grad_t);    //TODO kaiping (ddim, ldim, rdim) = (2, 2, 1) -> (2, 2, 2)
            //gsrc[t] = dot(grad[t], weight.T());     //TODO kaiping (ddim, ldim, rdim) = (1, 1, 2)
        }
        gsrc = dot(grad, weight.T());
    }
    LOG(ERROR) << "----------This is inner product layer----------";
    LOG(ERROR) << "Backward Phase: Sum value of weight data: " << weight_->mutable_data()->asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of weight grad: " << weight_->mutable_grad()->asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of data: " << data_.asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of grad: " << grad_.asum_data();
}


/*********** 4-Implementation for RnnlmWordinputLayer **********/
RnnlmWordinputLayer::~RnnlmWordinputLayer() {
  delete weight_;
}

void RnnlmWordinputLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src=srclayers_[0]->data(this);
  windowsize_=src.shape()[0];
  vdim_=src.count()/windowsize_; //dimension of input, i.e., 1
    CHECK_EQ(vdim_, 1);
  //hdim_=proto.rnnlmwordinput_conf().word_length(); // i.e., |V|
  hdim_=proto.GetExtension(rnnlmwordinput_conf).word_length(); // i.e., |V|
  data_.Reshape(vector<int>{windowsize_, hdim_});
  grad_.ReshapeLike(data_);
  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  weight_ = factory->Create("Param");
  vocabsize_ = static_cast<RnnlmWordparserLayer*>(srclayers_[0])->vocabsize();   //use type casting static_cast or dynamic_cast?
  weight_->Setup(proto.param(0), vector<int>{vocabsize_, hdim_});
}

void RnnlmWordinputLayer::ComputeFeature(Phase phase, Metric* perf) {
    Blob<float> *data_ptr = &data_;
    float *data_ptr_tmp = data_ptr->mutable_cpu_data();
    Blob<float> *src_ptr = srclayers_[0]->mutable_data(this);
    float *src_ptr_tmp = src_ptr->mutable_cpu_data();
    Blob<float> *weight_ptr = weight_->mutable_data();
    float *weight_ptr_tmp = weight_ptr->mutable_cpu_data();
  for(int t = 0; t < windowsize_; t++){ //Then src[t] is the t'th input word index
    //data[t] = weight[src[t]];
      //Check whether src_ptr_tmp[t] is in the range [0, vocabsize_ - 1]
      CHECK_GE(src_ptr_tmp[t],0);
      CHECK_LT(src_ptr_tmp[t], vocabsize_);
      memcpy(data_ptr_tmp + hdim_ * t, weight_ptr_tmp + hdim_ * static_cast<int>(src_ptr_tmp[t]), sizeof(float) * hdim_);
  }
    LOG(ERROR) << "----------This is wordinput layer----------";
    LOG(ERROR) << "Forward Phase: Sum value of data: " << data_.asum_data();
    LOG(ERROR) << "Forward Phase: Sum value of grad: " << grad_.asum_data();
}

void RnnlmWordinputLayer::ComputeGradient(Phase phas) {
    Blob<float> *gweight_ptr = weight_->mutable_grad();    //the gradient for the parameter: weight matrix
    float *gweight_ptr_tmp = gweight_ptr->mutable_cpu_data();
    Blob<float> *grad_ptr = &grad_;    //(win_size, |V|)
    float *grad_ptr_tmp = grad_ptr->mutable_cpu_data();
    Blob<float> *src_ptr = srclayers_[0]->mutable_data(this);
    float *src_ptr_tmp = src_ptr->mutable_cpu_data();

    memset(gweight_ptr_tmp, 0, sizeof(float) * vocabsize_ * hdim_);   //Need initialization before aggregate updates in all timestamps

   //Update the weight matrix here
   for(int t = 0; t < windowsize_; t++){
    //gweight[src[t]] = grad[t];
       memcpy(gweight_ptr_tmp + hdim_ * static_cast<int>(src_ptr_tmp[t]), grad_ptr_tmp + hdim_ * t, sizeof(float) * hdim_);
   }
    LOG(ERROR) << "----------This is wordinput layer----------";
    LOG(ERROR) << "Backward Phase: Sum value of weight data: " << weight_->mutable_data()->asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of weight grad: " << weight_->mutable_grad()->asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of data: " << data_.asum_data();
    LOG(ERROR) << "Backward Phase: Sum value of grad: " << grad_.asum_data();
}

/*********** 5-Implementation for RnnlmWordparserLayer **********/
void RnnlmWordparserLayer::Setup(const LayerProto& proto, int npartitions){
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  windowsize_ = static_cast<RnnlmDataLayer*>(srclayers_[0])->windowsize();
  vocabsize_ = static_cast<RnnlmDataLayer*>(srclayers_[0])->vocabsize();
  data_.Reshape(vector<int>{windowsize_});  //Can use 1-dimension
}
void RnnlmWordparserLayer::ParseRecords(Phase phase, const vector<Record>& records, Blob<float>* blob){
    //LOG(ERROR) << "This is word parser layer";
    float *data_dptr = data_.mutable_cpu_data();
    singa::SingleWordRecord swr;
    for(int i = 0; i < records.size() - 1; i++){//The first windowsize_ records in input "windowsize_ + 1" records
        //data_[i] = records[i].word_record().word_index();
        //data_dptr[i] = records[i].word_record().word_index();
	swr = records[i].GetExtension(singleword); // clee
	data_dptr[i] = swr.word_index();
    }
    LOG(ERROR) << "This is word parser layer";
}

/*********** 6-Implementation for RnnlmClassparserLayer **********/
void RnnlmClassparserLayer::Setup(const LayerProto& proto, int npartitions){
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  windowsize_ = static_cast<RnnlmDataLayer*>(srclayers_[0])->windowsize();
  vocabsize_ = static_cast<RnnlmDataLayer*>(srclayers_[0])->vocabsize();
  classsize_ = static_cast<RnnlmDataLayer*>(srclayers_[0])->classsize();
  data_.Reshape(vector<int>{windowsize_, 4});
}
void RnnlmClassparserLayer::ParseRecords(Phase phase, const vector<Record>& records, Blob<float>* blob){
    float *data_dptr = data_.mutable_cpu_data();
    //Blob<int> *class_info_ptr = (static_cast<RnnlmDataLayer*>(srclayers_[0])->classinfo());
    int *class_info_ptr_tmp = (static_cast<RnnlmDataLayer*>(srclayers_[0])->classinfo())->mutable_cpu_data();
    singa::SingleWordRecord swr;
    for(int i = 1; i < records.size(); i++){//The last windowsize_ records in input "windowsize_ + 1" records
	//swr = records[i].GetExtension(singa::SingleWordRecord::singleword); // clee
	swr = records[i].GetExtension(singleword); // clee
        int tmp_class_idx = swr.class_index();
        //data_[i][0] = (*(static_cast<RnnlmDataLayer*>(srclayers_[0])->classinfo()))[tmp_class_idx][0];
        data_dptr[4 * (i - 1) + 0] = class_info_ptr_tmp[2 * tmp_class_idx + 0];
        //data_[i][1] = (*(static_cast<RnnlmDataLayer*>(srclayers_[0])->classinfo()))[tmp_class_idx][1];
        data_dptr[4 * (i - 1) + 1] = class_info_ptr_tmp[2 * tmp_class_idx + 1];
        //data_[i][2] = records[i].word_record().word_index();
        data_dptr[4 * (i - 1) + 2] = swr.word_index(); // clee
        //data_[i][3] = tmp_class_idx;
        data_dptr[4 * (i - 1) + 3] = tmp_class_idx;
        //LOG(ERROR) << "Test class parser information: (start, end, word_idx, end_idx): ";
        //LOG(ERROR) << "( " << data_dptr[4 * (i - 1) + 0] << " , " << data_dptr[4 * (i - 1) + 1] << " , " << data_dptr[4 * (i - 1) + 2] << " , " << data_dptr[4 * (i - 1) + 3] << " )";
    }
    LOG(ERROR) << "This is class parser layer";
}

/*********** 7-Implementation for RnnlmDataLayer **********/
void RnnlmDataLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  classshard_ = std::make_shared<DataShard>(
		//proto.rnnlmdata_conf().class_path(),
		proto.GetExtension(rnnlmdata_conf).class_path(),
		//proto.GetExtension(RnnlmDataProto::rnnlmdata_conf).class_path(),
		DataShard::kRead);
  wordshard_ = std::make_shared<DataShard>(
		//proto.rnnlmdata_conf().word_path(),
		proto.GetExtension(rnnlmdata_conf).word_path(),
		//proto.GetExtension(RnnlmDataProto::rnnlmdata_conf).word_path(),
		DataShard::kRead);
  string class_key, word_key;
  windowsize_ = proto.GetExtension(rnnlmdata_conf).window_size();
  //windowsize_ = proto.GetExtension(RnnlmDataProto::rnnlmdata_conf).window_size();
  records_.resize(windowsize_ + 1);
  classsize_ = classshard_->Count(); //First read through class_shard and obtain values for class_size and vocab_size
        LOG(ERROR) << "class size: " << classsize_;
  classinfo_.Reshape(vector<int>{classsize_, 2});    //classsize_ rows and 2 columns

  int max_vocabidx_end = 0;
  int *class_info_ptr = classinfo_.mutable_cpu_data();
  singa::WordClassRecord wcr;
  for(int i = 0; i < classsize_; i++){
    classshard_->Next(&class_key, &sample_);
    //wcr = sample_.GetExtension(singa::WordClassRecord::wordclass); // clee
    wcr = sample_.GetExtension(wordclass); // clee
    //classinfo_[i][0] = sample_.class_record().start();
      class_info_ptr[2 * i + 0] = wcr.start();
    //classinfo_[i][1] = sample_.class_record().end();
      class_info_ptr[2 * i + 1] = wcr.end();
    if(wcr.end() > max_vocabidx_end){
        max_vocabidx_end = wcr.end();
    }
  }
  vocabsize_ = max_vocabidx_end + 1;
        LOG(ERROR) << "vocabulary size: " << vocabsize_;
  wordshard_->Next(&word_key, &records_[windowsize_]);    //Then read the 1st record in word_shard and assign it to records_[windowsize_] for convenience & consistency in ComputeFeature()
}


void RnnlmDataLayer::ComputeFeature(Phase phase, Metric* perf){
  CHECK(records_.size() <= wordshard_->Count());
  records_[0] = records_[windowsize_];
    //LOG(ERROR) << "Training data shard info: word: " << records_[0].word_record().word() << " wordIndex: "
    //<< records_[0].word_record().word_index() << " classIndex: " << records_[0].word_record().class_index();
  while (true) {
	bool flag = true;
	for (int i = 1; i < records_.size(); i++) { //size of records_ is windowsize_ + 1; range: [1, windowsize_]
		string key;
		if (!wordshard_->Next(&key, &records_[i])) { //When throwing the ending words ( < window_size)
			wordshard_->SeekToFirst();
			flag = false;
			break;
		}
        //LOG(ERROR) << "Training data shard info: word: " << records_[i].word_record().word() << " wordIndex: "
        //<< records_[i].word_record().word_index() << " classIndex: " << records_[i].word_record().class_index();
	}
	if (flag == true) break;
}
    LOG(ERROR) << "This is data layer";
}





/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::Setup(const LayerProto& proto, int npartitions){
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();
  data_.Reshape(vector<int>{batchsize});
}

void LabelLayer::ParseRecords(Phase phase, const vector<Record>& records,
    Blob<float>* blob){
  int rid=0;
  float *label= blob->mutable_cpu_data() ;
  for(const Record& record: records){
    label[rid++]=record.image().label();
    CHECK_LT(record.image().label(),10);
  }
  CHECK_EQ(rid, blob->shape()[0]);
}

/***************** Implementation for LRNLayer *************************/
void LRNLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  lsize_ = proto.lrn_conf().local_size();
  CHECK_EQ(lsize_ % 2, 1) << "LRN only supports odd values for Localvol";
  knorm_=proto.lrn_conf().knorm();
  alpha_ = proto.lrn_conf().alpha();
  beta_ = proto.lrn_conf().beta();

  const vector<int>& s=srclayers_[0]->data(this).shape();
  data_.Reshape(s);
  grad_.Reshape(s);
  norm_.Reshape(s);
  batchsize_=s[0];
  channels_=s[1];
  height_=s[2];
  width_=s[3];
}

void LRNLayer::ComputeFeature(Phase phase, Metric* perf) {
  const float salpha = alpha_ / lsize_;
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor4(&data_);
  auto norm = Tensor4(&norm_);
  // stores normalizer without power
  norm= chpool<red::sum>( F<op::square>(src) , lsize_ ) * salpha + knorm_;
  data = src * F<op::power>(norm, -beta_ );
}

void LRNLayer::ComputeGradient(Phase phase) {
  const float salpha = alpha_ / lsize_;
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto norm = Tensor4(&norm_);
  auto grad = Tensor4(&grad_);
  auto gsrc = Tensor4(srclayers_[0]->mutable_grad(this));

  gsrc = grad * F<op::power>( norm, -beta_ );
  gsrc += ( - 2.0f * beta_ * salpha ) * chpool<red::sum>(
      grad * src * F<op::power>( norm, -beta_-1.0f ), lsize_ )  * src;
}

/**************** Implementation for MnistImageLayer******************/

void MnistLayer::ParseRecords(Phase phase,
    const vector<Record>& records, Blob<float>* blob){
  LOG_IF(ERROR, records.size()==0)<<"Empty records to parse";
  int ndim=records.at(0).image().shape_size();
  int inputsize =records.at(0).image().shape(ndim-1);
  CHECK_EQ(inputsize, blob->shape()[1]);

  float* dptr=blob->mutable_cpu_data();
  for(const Record& record: records){
    const SingleLabelImageRecord& imagerecord=record.image();
    if(imagerecord.pixel().size()) {
      string pixel=imagerecord.pixel();
      for(int i = 0, k = 0; i < inputsize; i++) {
        for(int j = 0; j < inputsize; j++) {
          // NOTE!!! must cast pixel to uint8_t then to float!!! waste a lot of
          // time to debug this
          float x =  static_cast<float>(static_cast<uint8_t>(pixel[k++]));
          x = x / norm_a_-norm_b_;
          *dptr = x;
          dptr++;
        }
      }
    } else {
      for(int i = 0, k = 0; i < inputsize; i++) {
        for(int j = 0; j < inputsize; j++) {
          *dptr = imagerecord.data(k++) / norm_a_ - norm_b_;
          dptr++;
        }
      }
    }
  }
  CHECK_EQ(dptr, blob->mutable_cpu_data()+blob->count());
}
void MnistLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers_[0])->sample();
  kernel_=proto.mnist_conf().kernel();
  sigma_=proto.mnist_conf().sigma();
  alpha_=proto.mnist_conf().alpha();
  beta_=proto.mnist_conf().beta();
  gamma_=proto.mnist_conf().gamma();
  resize_=proto.mnist_conf().resize();
  norm_a_=proto.mnist_conf().norm_a();
  norm_b_=proto.mnist_conf().norm_b();
  elastic_freq_=proto.mnist_conf().elastic_freq();

  int ndim=sample.image().shape_size();
  CHECK_GE(ndim,2);
  if(resize_)
    data_.Reshape(vector<int>{batchsize, resize_, resize_});
  else{
    int s=sample.image().shape(ndim-1);
    CHECK_EQ(s,sample.image().shape(ndim-2));
    data_.Reshape(vector<int>{batchsize, s, s });
  }
}

/******************** Implementation for PoolingLayer******************/
void PoolingLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  PoolingProto pool_conf = proto.pooling_conf();
  kernel_=pool_conf.kernel();
  stride_=pool_conf.stride();
  CHECK_LT(pad_, kernel_);
  pool_=proto.pooling_conf().pool();
  CHECK(pool_ == PoolingProto_PoolMethod_AVE
        || pool_ == PoolingProto_PoolMethod_MAX)
      << "Padding implemented only for average and max pooling.";

  const auto& srcshape=srclayers_[0]->data(this).shape();
  int dim=srcshape.size();
  CHECK_GT(dim,2);
  width_ = srcshape[dim-1];
  height_ = srcshape[dim-2];
  if(dim>3)
    channels_ = srcshape[dim-3];
  else
    channels_=1;
  batchsize_=srcshape[0];
  pooled_height_ = static_cast<int>((height_ - kernel_) / stride_) + 1;
  pooled_width_ = static_cast<int>(( width_ - kernel_) / stride_) + 1;
  data_.Reshape(vector<int>{batchsize_, channels_, pooled_height_, pooled_width_});
  grad_.ReshapeLike(data_);
}

void PoolingLayer::ComputeFeature(Phase phase, Metric* perf) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor4(&data_);
  if(pool_ == PoolingProto_PoolMethod_MAX)
    data=pool<red::maximum>(src, kernel_, stride_);
  else if(pool_ == PoolingProto_PoolMethod_AVE)
    data=pool<red::sum>(src, kernel_, stride_) *(1.0f/(kernel_*kernel_));
}

/*
 * partition only on num/channel dim
 * assume grad and data have the same paritition
 */
void PoolingLayer::ComputeGradient(Phase phase) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto gsrc = Tensor4(srclayers_[0]->mutable_grad(this));
  auto data = Tensor4(&data_);
  auto grad = Tensor4(&grad_);
  if(pool_ == PoolingProto_PoolMethod_MAX)
    gsrc = unpool<red::maximum>(src, data, grad, kernel_, stride_);
  else if(pool_ == PoolingProto_PoolMethod_AVE)
    gsrc = unpool<red::sum>(src, data, grad, kernel_, stride_)
      *(1.0f/(kernel_*kernel_));
}

/***************** Implementation for ReLULayer *****************************/

void ReLULayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(*(srclayers_[0]->mutable_grad(this)));
}

void ReLULayer::ComputeFeature(Phase phase, Metric* perf) {
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data=F<op::relu>(src);
}

void ReLULayer::ComputeGradient(Phase phase) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc=F<op::relu_grad>(data)*grad;
}

/*************** Implementation for RGBImageLayer *************************/

void RGBImageLayer::ParseRecords(Phase phase,
    const vector<Record>& records, Blob<float>* blob){
  const vector<int>& s=blob->shape();
  auto images = Tensor4(&data_);
  const SingleLabelImageRecord& r=records.at(0).image();
  Tensor<cpu, 3> raw_image(Shape3(r.shape(0),r.shape(1),r.shape(2)));
  AllocSpace(raw_image);
  Tensor<cpu, 3> croped_image(nullptr, Shape3(s[1],s[2],s[3]));
  if(cropsize_)
    AllocSpace(croped_image);
    //CHECK(std::equal(croped_image.shape(), raw_image.shape());
  int rid=0;
  const float* meandptr=mean_.cpu_data();
  for(const Record& record: records){
    auto image=images[rid];
    bool do_crop=cropsize_>0&&(phase == kTrain);
    bool do_mirror=mirror_&&rand()%2&&(phase == kTrain);
    float* dptr=nullptr;
    if(do_crop||do_mirror)
      dptr=raw_image.dptr;
    else
      dptr=image.dptr;
    if(record.image().pixel().size()){
      string pixel=record.image().pixel();
      for(size_t i=0;i<pixel.size();i++)
        dptr[i]=static_cast<float>(static_cast<uint8_t>(pixel[i]));
    }else {
      memcpy(dptr, record.image().data().data(),
          sizeof(float)*record.image().data_size());
    }
    for(int i=0;i<mean_.count();i++)
      dptr[i]-=meandptr[i];

    if(do_crop){
      int hoff=rand()%(r.shape(1)-cropsize_);
      int woff=rand()%(r.shape(2)-cropsize_);
      Shape<2> cropshape=Shape2(cropsize_, cropsize_);
      if(do_mirror){
        croped_image=crop(raw_image, cropshape, hoff, woff);
        image=mirror(croped_image);
      }else{
        image=crop(raw_image, cropshape, hoff, woff);
      }
    }else if(do_mirror){
      image=mirror(raw_image);
    }
    rid++;
  }
  if(scale_)
    images=images*scale_;

  FreeSpace(raw_image);
  if(cropsize_)
    FreeSpace(croped_image);
}
void RGBImageLayer::Setup(const LayerProto& proto, int npartitions) {
  ParserLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  scale_=proto.rgbimage_conf().scale();
  cropsize_=proto.rgbimage_conf().cropsize();
  mirror_=proto.rgbimage_conf().mirror();
  int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers_[0])->sample();
  vector<int> shape;
  shape.push_back(batchsize);
  for(int x: sample.image().shape()){
    shape.push_back(x);
  }
  CHECK_EQ(shape.size(),4);
  if(cropsize_){
    shape[2]=cropsize_;
    shape[3]=cropsize_;
  }
  data_.Reshape(shape);
  mean_.Reshape({shape[1],shape[2],shape[3]});
  if(proto.rgbimage_conf().has_meanfile()){
    if(proto.rgbimage_conf().meanfile().find("binaryproto") != string::npos) {
      CaffeBlob mean;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &mean);
      CHECK_EQ(mean_.count(), mean.data_size());
      memcpy(mean_.mutable_cpu_data(), mean.data().data(),
          sizeof(float)*mean.data_size());
    } else {
      SingleLabelImageRecord mean;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &mean);
      CHECK_EQ(mean_.count(), mean.data_size());
      memcpy(mean_.mutable_cpu_data(), mean.data().data(),
          sizeof(float)*mean.data_size());
    }
  } else {
    memset(mean_.mutable_cpu_data(),0,sizeof(float)*mean_.count());
  }
}

/***************Implementation for ShardDataLayer**************************/
void ShardDataLayer::ComputeFeature(Phase phase, Metric* perf){
  if(random_skip_){
    int nskip = rand() % random_skip_;
    LOG(INFO)<<"Random Skip "<<nskip<<" records, there are "<<shard_->Count()
      <<" records in total";
    string key;
    for(int i=0;i<nskip;i++){
      shard_->Next(&key, &sample_);
    }
    random_skip_=0;
  }
  for(auto& record: records_){
    string key;
    if(!shard_->Next(&key, &record)){
      shard_->SeekToFirst();
      CHECK(shard_->Next(&key, &record));
    }
  }
}

void ShardDataLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  shard_= std::make_shared<DataShard>(proto.sharddata_conf().path(),
      DataShard::kRead);
  string key;
  shard_->Next(&key, &sample_);
  batchsize_=proto.sharddata_conf().batchsize();
  if(partition_dim() == 0)
    batchsize_ /= npartitions;

  records_.resize(batchsize_);
  random_skip_=proto.sharddata_conf().random_skip();
}
/*******************Implementation of TanLayer***************************/
void TanhLayer::Setup(const LayerProto& proto, int npartitions){
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(srclayers_[0]->grad(this));
}

void TanhLayer::ComputeFeature(Phase phase, Metric* perf) {
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data=F<op::stanh>(src);
}

void TanhLayer::ComputeGradient(Phase phase) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc=F<op::stanh_grad>(data)*grad;
}
/********** * Implementation for SoftmaxLossLayer*************************/
void SoftmaxLossLayer::Setup(const LayerProto& proto, int npartitions) {
  LossLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),2);
  data_.Reshape(srclayers_[0]->data(this).shape());
  batchsize_=data_.shape()[0];
  dim_=data_.count()/batchsize_;
  topk_=proto.softmaxloss_conf().topk();
  metric_.Reshape(vector<int>{2});
  scale_=proto.softmaxloss_conf().scale();
}
void SoftmaxLossLayer::ComputeFeature(Phase phase, Metric* perf) {
  Shape<2> s=Shape2(batchsize_, dim_);
  Tensor<cpu, 2> prob(data_.mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers_[0]->mutable_data(this)->mutable_cpu_data(), s);
  Softmax(prob, src);
  const float* label=srclayers_[1]->data(this).cpu_data();
  const float* probptr=prob.dptr;
  float loss=0, precision=0;
  for(int n=0;n<batchsize_;n++){
    int ilabel=static_cast<int>(label[n]);
    CHECK_LT(ilabel,10);
    CHECK_GE(ilabel,0);
    float prob_of_truth=probptr[ilabel];
    loss-=log(std::max(prob_of_truth, FLT_MIN));
    vector<std::pair<float, int> > probvec;
    for (int j = 0; j < dim_; ++j) {
      probvec.push_back(std::make_pair(probptr[j], j));
    }
    std::partial_sort(
        probvec.begin(), probvec.begin() + topk_,
        probvec.end(), std::greater<std::pair<float, int> >());
    // check if true label is in top k predictions
    for (int k = 0; k < topk_; k++) {
      if (probvec[k].second == static_cast<int>(label[n])) {
        precision++;
        break;
      }
    }
    probptr+=dim_;
  }
  CHECK_EQ(probptr, prob.dptr+prob.shape.Size());
  perf->Add("loss", loss*scale_/(1.0f*batchsize_));
  perf->Add("accuracy", precision*scale_/(1.0f*batchsize_));
}

void SoftmaxLossLayer::ComputeGradient(Phase phase) {
  const float* label=srclayers_[1]->data(this).cpu_data();
  Blob<float>* gsrcblob=srclayers_[0]->mutable_grad(this);
  gsrcblob->CopyFrom(data_);
  float* gsrcptr=gsrcblob->mutable_cpu_data();
  for(int n=0;n<batchsize_;n++){
    gsrcptr[n*dim_+static_cast<int>(label[n])]-=1.0f;
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc*=scale_/(1.0f*batchsize_);
}

}  // namespace singa
