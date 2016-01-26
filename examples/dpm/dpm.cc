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
#include "./dpm.h"

#include <string>
#include <algorithm>
#include "mshadow/tensor.h"
#include "mshadow/tensor_expr.h"
#include "mshadow/cxxnet_op.h"
#include "singa/utils/singleton.h"
#include "singa/utils/math_blob.h"
#include "singa/utils/singa_op.h"
#include "./dpm.pb.h"

namespace dpm {
using std::vector;
using std::string;

using namespace mshadow;
using mshadow::cpu;
using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Tensor;


/*******DataLayer**************/
DataLayer::~DataLayer() {
  if (store_ != nullptr)
    delete store_;
  if (store2_ != nullptr)
    delete store2_;
}

void DataLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "Setup @ Data";
  InputLayer::Setup(conf, srclayers);
  batchsize_ = conf.GetExtension(data_conf).batchsize();
  unroll_len_ = conf.GetExtension(data_conf).unroll_len();
  // e.g., feature_len = 598 (age, edu, gen, lap + feature_values (e.g., 592) + deltaT + MMSCORE)
  feature_len_ = conf.GetExtension(data_conf).feature_len();
  datavec_.clear();
  // each unroll layer has a input blob
  for (int i = 0; i <= unroll_len_; i++) {
    datavec_.push_back(new Blob<float>(batchsize_,feature_len_));
  }
}

void DataLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  string key, value;
  string key2, value2;
  DynamicRecord dynamic;
  OutTimeRecord outtime;
  //LOG(ERROR) << "Comp @ Data ----------";
  if (store_ == nullptr) {
    store_ = singa::io::OpenStore(
        layer_conf_.GetExtension(data_conf).backend(),
        layer_conf_.GetExtension(data_conf).path(),
        singa::io::kRead);
  }
  if (store2_ == nullptr) {
    store2_ = singa::io::OpenStore(
        layer_conf_.GetExtension(data_conf).backend(),
        layer_conf_.GetExtension(data_conf).label_path(),
        singa::io::kRead);
  }

  // initialize data: this is to make sure that GRU units with no records are filled with 0
  for (int l=0; l<unroll_len_; l++) {
      float* ptr = datavec_[l]->mutable_cpu_data();
      memset(ptr, 0, sizeof(float) * datavec_[l]->count());
  }

  for (int b = 0; b < batchsize_; b++) {

     int l=0; // idx of unroll layers
     int scnt=0; // number of samples

     while(1) {
        if (!store_->Read(&key, &value)) {
           store_->SeekToFirst();
           CHECK(store_->Read(&key, &value));
           store2_->SeekToFirst();
        }
        dynamic.ParseFromString(value);

        if (dynamic.patient_id() == -1) {

           if (!store2_->Read(&key2, &value2)) {
              store2_->SeekToFirst();
              CHECK(store2_->Read(&key2, &value2));
           }
           outtime.ParseFromString(value2);

           float* ptr = datavec_[unroll_len_-1]->mutable_cpu_data();
           ptr[b * feature_len_ + feature_len_ - 2 + 0] = static_cast<float>(outtime.delta_time());
           ptr[b * feature_len_ + feature_len_ - 2 + 1] = static_cast<float>(outtime.mmscore());

           //LOG(ERROR) << "label: batch " << b << ", dt: " << static_cast<float>(outtime.delta_time()) << ", mm: " << static_cast<float>(outtime.mmscore());

           scnt = 0;
           break;
        }

        if (scnt++ == 0) {
           // Get index of unroll (gru unit). Fill it from the end
           l = unroll_len_ - static_cast<int>(dynamic.nb_sample());
        }
        float* ptr = datavec_[l]->mutable_cpu_data();
        ptr[b * feature_len_ + 0] = static_cast<float>(dynamic.age());  // feature_len_ is 596 (4 + 592)
        ptr[b * feature_len_ + 1] = static_cast<float>(dynamic.education());
        ptr[b * feature_len_ + 2] = static_cast<float>(dynamic.gender());
        ptr[b * feature_len_ + 3] = static_cast<float>(dynamic.lap_time());
        // all zero
        for (int i=4; i<feature_len_-2; i++)
           ptr[b * feature_len_ + i] = 0;
        // except the observed
        for (int k=0; k<dynamic.feature_value_size(); k++) {
           int idx = dynamic.observed_idx(k);
           ptr[b * feature_len_ + 4 + idx] = static_cast<float>(dynamic.feature_value(k));
           ////LOG(ERROR) << "Patient ID: " << dynamic.patient_id() << " Detailed features: " << "(l,b)=(" << l << "," << b << ")" << static_cast<float>(dynamic.feature_value(k));
        }

        //LOG(ERROR) << "(l,b)=(" << l << "," << b << "), pid " << static_cast<int>(dynamic.patient_id())
                                                 //<< ", lap: " << static_cast<int>(dynamic.lap_time());

        l++;
     }
  }
}

/*******UnrollLayer**************/
void UnrollLayer::Setup(const LayerProto& conf,
  const vector<Layer*>& srclayers) {
  InputLayer::Setup(conf, srclayers);
  batchsize_ = srclayers.at(0)->data(unroll_index()).shape(0);
  feature_len_ = dynamic_cast<DataLayer*>(srclayers[0])->feature_len();  // feature_len_ is 598 = 4 + 592 + 2
  data_.Reshape(batchsize_, feature_len_ - 2);  // reshape data for each unit
}

void UnrollLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  float* ptr = data_.mutable_cpu_data();
  memset(ptr, 0, sizeof(float) * data_.count());
  const float* idx = srclayers[0]->data(unroll_index()).cpu_data();
  for (int b = 0; b < batchsize_; b++) { // 4 demographical features
      ptr[b * (feature_len_ - 2) + 0] = static_cast<float>( idx[b * feature_len_ + 0] );  // age
      ptr[b * (feature_len_ - 2) + 1] = static_cast<float>( idx[b * feature_len_ + 1] );  // edu
      ptr[b * (feature_len_ - 2) + 2] = static_cast<float>( idx[b * feature_len_ + 2] );  // gen
      ptr[b * (feature_len_ - 2) + 3] = static_cast<float>( idx[b * feature_len_ + 3] );  // lap_time
      for (int i=4; i<feature_len_-2; i++) { // (feature_len_ - 6) = 592 features
          ptr[b * (feature_len_ - 2) + i] = static_cast<float>( idx[b * feature_len_ + i] );  // feature_value
      }
  }
}

/*******DPMLabelLayer**************/
void DPMLabelLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  InputLayer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 1); // DPMLabelLayer has only 1 src layer
  batchsize_ = srclayers.at(0)->data(0).shape(0);
  feature_len_ = dynamic_cast<DataLayer*>(srclayers[0])->feature_len();
  unroll_len_ = dynamic_cast<DataLayer*>(srclayers[0])->unroll_len();
  data_.Reshape(batchsize_, 1);
}

void DPMLabelLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  float* ptr = data_.mutable_cpu_data();
  // look at the last unroll unit only
  const float* idx = srclayers[0]->data(unroll_len_-1).cpu_data();
  for (int b = 0; b < batchsize_; b++) {
    ptr[b] = static_cast<int>(idx[b * feature_len_ + feature_len_ - 1]);  // mmscore
    //LOG(ERROR) << "data_ for DPMLabelLayer: " << ptr[b];
  }
}

/*******DPMTimeLayer**************/
void DPMTimeLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  InputLayer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 1); // DPMLabelLayer has only 1 src layer
  batchsize_ = srclayers.at(0)->data(0).shape(0);
  feature_len_ = dynamic_cast<DataLayer*>(srclayers[0])->feature_len();
  unroll_len_ = dynamic_cast<DataLayer*>(srclayers[0])->unroll_len();
  data_.Reshape(batchsize_, 1);
}

void DPMTimeLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  float* ptr = data_.mutable_cpu_data();
  // look at the last unroll unit only
  const float* idx = srclayers[0]->data(unroll_len_-1).cpu_data();
  for (int b = 0; b < batchsize_; b++) {
    ptr[b] = static_cast<int>(idx[b * feature_len_ + feature_len_ - 2]);  // delta_time
    //LOG(ERROR) << "data_ for DPMTimeLayer: " << ptr[b];
  }
}
/*******CombinationLayer**************/

CombinationLayer::~CombinationLayer() {
  delete weight1_;
  delete weight2_;
  delete bias_;
}

void CombinationLayer::Setup(const LayerProto &conf,
                             const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 2); // CombinationLayer has 2 src layers
  const auto& src_gru = srclayers[0]->data(this); // 1st src layer is gru layer
  const auto& src_timespan = srclayers[1]->data(this); // 2nd layer is time_span layer
  batchsize_ = src_gru.shape()[0];
  vdim_ = src_gru.count() / batchsize_;
  hdim_ = conf.GetExtension(combination_conf).num_output();
  transpose_ = conf.GetExtension(combination_conf).transpose();
  //if (partition_dim() > 0)
  //  hdim_ /= srclayers.at(0)->num_partitions();
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  weight1_ = Param::Create(conf.param(0));
  weight2_ = Param::Create(conf.param(1));
  bias_ = Param::Create(conf.param(2));
  if(transpose_) {
    weight1_->Setup(vector<int>{vdim_,hdim_});
    weight2_->Setup(vector<int>{1,hdim_});
  }
  else { // this is our case
    weight1_->Setup(vector<int>{hdim_,vdim_});
    weight2_->Setup(vector<int>{hdim_,1});
  }
  bias_->Setup(vector<int>{hdim_});
}

void CombinationLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  Blob<float>* tmp = new Blob<float>(batchsize_, hdim_); // Use "tmp" to store the computation result from TimeSpanUnit
  if(transpose_) {
    MMDot(srclayers[0]->data(this), weight1_->data(), &data_); // First part of data_
    MMDot(srclayers[1]->data(this), weight2_->data(), tmp); // Second part of data_
  }
  else {
    MMDot(srclayers[0]->data(this), weight1_->data().T(), &data_); // First part of data_
    MMDot(srclayers[1]->data(this), weight2_->data().T(), tmp); // Second part of data_
  }
  AXPY(1.0f, *tmp, &data_); // Combine 2 parts
  MVAddRow(bias_->data(), &data_); // Add bias
  delete tmp;
//  const float* printout = weight2_->data().cpu_data();
//  LOG(ERROR) << "Weight info: ";
//  LOG(ERROR) << "Weight info size: " << weight2_->data().shape().size();
//  LOG(ERROR) << "Weight info shape(0): " << weight2_->data().shape(0);
//  LOG(ERROR) << "Weight info shape(1): " << weight2_->data().shape(1);
//  for(int i = 0; i < weight2_->data().shape(0); i++) {
//      for(int j = 0; j < weight2_->data().shape(1); j++) {
//          LOG(ERROR) << "Weight info @ ( " << i << " , " << j << " ) is:   " << (*(printout + i * weight2_->data().shape(1) + j));
//      }
//  }
//  LOG(ERROR) << "Bias info: ";
//  LOG(ERROR) << "Bias info size: " << bias_->data().shape().size();
//  LOG(ERROR) << "Bias info shape(0): " << bias_->data().shape(0);
//  const float* printbias = bias_->data().cpu_data();
//  for(int k = 0; k < bias_->data().shape(0); k++) {
//      LOG(ERROR) << "Bias info @ " << k << " is:   " << (*(printbias + k));
//  }
  //LOG(ERROR) << "Weight info: ";
  //LOG(ERROR) << "Weight information for delta_T: " << weight2_->data();
  //LOG(ERROR) << "Bias information for delta_T: " << bias_->data();
}

void CombinationLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  float beta = 0.0f;
  if(flag & singa::kAggGrad)
    beta = 1.0f;
  MVSumRow(1.0f, beta, grad_, bias_->mutable_grad());
  if(transpose_) {
    GEMM(1.0f, beta, srclayers[0]->data(this).T(), grad_,
         weight1_->mutable_grad());
    GEMM(1.0f, beta, srclayers[1]->data(this).T(), grad_,
         weight2_->mutable_grad());
  }
  else {
    GEMM(1.0f, beta, grad_.T(), srclayers[0]->data(this),
         weight1_->mutable_grad());
    GEMM(1.0f, beta, grad_.T(), srclayers[1]->data(this),
         weight2_->mutable_grad());
  }

  if (srclayers[0]->mutable_grad(this) != nullptr) { // Compute the gradient for src_layer: gru layer; Note that gradient for src_layer: Delta_T is DataLayer, should be no grad_
    if (transpose_)
      MMDot(grad_, weight1_->data().T(), srclayers[0]->mutable_grad(this));
    else
      MMDot(grad_, weight1_->data(), srclayers[0]->mutable_grad(this));
  }
}

}   // end of namespace dpm
