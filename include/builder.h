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

#ifndef SINGA_BUILDER_H_
#define SINGA_BUILDER_H_

#include <stdarg.h>
#include "proto/job.pb.h"
#include "proto/singa.pb.h"
#include "utils/param.h"

#include <google/protobuf/text_format.h>
#include <glog/logging.h>

//namespace gp = google::protobuf;

namespace singa {

using google::protobuf::Message;
using google::protobuf::MessageFactory;
using google::protobuf::Descriptor;
using google::protobuf::FieldDescriptor;
using google::protobuf::EnumDescriptor;
using google::protobuf::Reflection;
using std::string;

class Builder {
public:

  static int uniqID[2];

  Builder(JobProto*);
  Builder(JobProto*, const char*);
 
  void Construct();
  void MLP_ex1();
  void MLP_ex2();
  void CONV_ex1();
  void CONV_ex2();
  void CONV_ex3();
  void AUTOENCODER_ex1();

  void Display();
  void Display_Layer();

  int ShowLayer(LayerType type) {
    return (int)type;  
  }

  //------------------------------
  // Layers 
  //------------------------------
  LayerProto* AddLayer(LayerType type) {
    return AddLayer(type, generateName("layer", 0), 0, nullptr);
  }

  LayerProto* AddLayer(LayerType type, const char* name) {
    return AddLayer(type, name, 0, nullptr);
  }

  LayerProto* AddLayer(LayerType type, const char* name, LayerProto* srclayer) {
    return AddLayer(type, name, 1, srclayer);
  }

  LayerProto* AddLayer(LayerType type, const char* name,
                       int num_srclayers, ...) {
    LayerProto* srclayer;
    layer = netconf->add_layer();
    layer->set_name(name);
    layer->set_type(type);
    va_list args;
    va_start(args, num_srclayers);
    for(int i=0; i<num_srclayers; i++) {
      srclayer = va_arg( args, LayerProto* );
      layer->add_srclayers(srclayer->name());
    }
    return layer;
  }

  void AddLayers(vector<LayerProto*>* out, LayerProto* srclayer,
                 int num_stacks, int num_layers, ...) {
    LayerType type;
    LayerProto* newlayer = srclayer;
    va_list args;
    for(int s=0; s<num_stacks; s++) {
      va_start(args, num_layers);
      for(int i=0; i<num_layers; i++) {
        type = (LayerType) va_arg( args, int );
        newlayer = AddLayer(type, generateName(type), newlayer);
        out->push_back(newlayer);
      }
      uniqID[0]++;
      va_end(args);
    }
  }

  void ReplicateLayers(vector<LayerProto*>* out, LayerProto* srclayer,
                       int num_stacks, int num_layers, ...) {
    LayerProto* layer;
    LayerProto* newlayer;
    va_list args;
    for(int s=0; s<num_stacks; s++) {
      va_start(args, num_layers);
      for(int i=0; i<num_layers; i++) {
        layer = va_arg( args, LayerProto* );
        newlayer = netconf->add_layer();
        newlayer->CopyFrom(*layer);
        newlayer->set_name(generateName(layer->type()));
        newlayer->clear_srclayers();
        newlayer->add_srclayers(srclayer->name());
        // rename parameters
        for(int j=0; j<newlayer->param_size(); j++) {
           const char* newname = generateName(layer->mutable_param(j)->name().c_str(),1);
           newlayer->mutable_param(j)->set_name(newname);
        }
        out->push_back(newlayer);
        srclayer = newlayer;
      }
      va_end(args);
      uniqID[0]++;
    }
  }


  // TODO: need to take care of kRecordInput
  void AddDataProto(LayerProto* layer,
                    Phase mode=kUnknown,
                    const char* path=nullptr,
                    int batchsize=0,
                    int random_skip=0) {
    Message* msg = nullptr;
    if(layer->type() == kShardData) {
      msg = layer->mutable_sharddata_conf();
    }
    else if(layer->type() == kLMDBData) {
      msg = layer->mutable_lmdbdata_conf();
    }
    CHECK(path!=nullptr);
    SetVal(msg, "path", path);
    if(batchsize>0)
      SetVal(msg, "batchsize", batchsize);
    if(random_skip>0)
      SetVal(msg, "random_skip", random_skip);
    if(mode!=kUnknown)
      layer->add_exclude(mode==kTrain?kTest:kTrain);
  }

  void AddMnistProto(LayerProto* layer,
                     float norm_a=1.0,
                     float norm_b=0.0) {
    if(layer->type() == kMnist) {
      layer->mutable_mnist_conf()->set_norm_a(norm_a);
      layer->mutable_mnist_conf()->set_norm_b(norm_b);
    }
  }

  void AddInnerProductProto(LayerProto* layer,
                            int num_output=0) {
    if(layer->type() == kInnerProduct) {
      if(num_output>0) {
        layer->mutable_innerproduct_conf()->set_num_output(num_output);
      }
    }
  }

  void SetInnerProductProto(vector<LayerProto*>* out, int num_layers, ...) {
    va_list args;
    va_start(args, num_layers);
    for(int i=0; i<static_cast<int>(out->size()); i++) {
      if(out->at(i)->type() == kInnerProduct) {
        int num_output = va_arg( args, int );
        out->at(i)->mutable_innerproduct_conf()->set_num_output(num_output);
      }
    }
    va_end(args);
  }

  void AddSoftmaxLossProto(LayerProto* layer,
                      int topk=0, float scale=0.0) {
    if(topk!=0)
      layer->mutable_softmaxloss_conf()->set_topk(topk);
    if(scale!=0)
      layer->mutable_softmaxloss_conf()->set_scale(scale);
  }

  void AddConvolutionProto(LayerProto* layer,
                     int num_filters, int kernel,
                     int pad=0, int stride=0, bool bias_term=true) {
    Message* msg = layer->mutable_convolution_conf();
    if(num_filters>0)
      SetVal(msg, "num_filters", num_filters);
    if(kernel>0)
      SetVal(msg, "kernel", kernel);
    if(pad>0)
      SetVal(msg, "pad", pad);
    if(stride>0)
      SetVal(msg, "stride", stride);
    if(bias_term==false)
      SetVal(msg, "bias_term", bias_term);
  }

  template<class T>
  void SetConvolutionProto(LayerProto* layer,
                           const char* key, T val) {
    Message* msg = layer->mutable_convolution_conf(); 
    SetVal(msg, key, val);
  }

  void AddPoolingProto(LayerProto* layer, int kernel,
                       PoolingProto::PoolMethod pool=PoolingProto::MAX,
                       int pad=0, int stride=1) {
    layer->mutable_pooling_conf()->set_kernel(kernel);
    layer->mutable_pooling_conf()->set_pool(pool);
    layer->mutable_pooling_conf()->set_pad(pad);
    layer->mutable_pooling_conf()->set_stride(stride);
  }

  template<class T>
  void SetPoolingProto(LayerProto* layer,
                       const char* key, T val) {
    Message* msg = layer->mutable_pooling_conf(); 
    SetVal(msg, key, val);
  }

  void AddDropoutProto(LayerProto* layer,
                       float ratio) {
    layer->mutable_dropout_conf()->set_dropout_ratio(ratio);
  }

  void AddRBMProto(LayerProto* layer,
                   int hdim) {
    AddRBMProto(layer, hdim, true, false);
  }

  void AddRBMProto(LayerProto* layer,
                   int hdim, bool bias_term, bool gaussian) {
    layer->mutable_rbm_conf()->set_hdim(hdim);
    layer->mutable_rbm_conf()->set_bias_term(bias_term);
    layer->mutable_rbm_conf()->set_gaussian(gaussian);
  }

  void AddReLUProto(LayerProto* layer,
                    float negative_slope) {
    layer->mutable_relu_conf()->set_negative_slope(negative_slope);
  }

  void AddLRNProto(LayerProto* layer,
                   int local_size=5, float alpha=1.0, float beta=0.75, float knorm=1.0) {
    layer->mutable_lrn_conf()->set_local_size(local_size);
    layer->mutable_lrn_conf()->set_alpha(alpha);
    layer->mutable_lrn_conf()->set_beta(beta);
    layer->mutable_lrn_conf()->set_knorm(knorm);
  }

  void AddRGBImageProto(LayerProto* layer,
                        const char* meanfile) {
    AddRGBImageProto(layer, 0, 0, false, meanfile);
  }

  void AddRGBImageProto(LayerProto* layer,
                 float scale=0, int cropsize=0, bool mirror=false, const char* meanfile=nullptr) {
    if(scale>0)
      layer->mutable_rgbimage_conf()->set_scale(scale);
    if(cropsize>0)
      layer->mutable_rgbimage_conf()->set_cropsize(cropsize);
    if(mirror)
      layer->mutable_rgbimage_conf()->set_mirror(mirror);
    if(meanfile!=nullptr)
      layer->mutable_rgbimage_conf()->set_meanfile(meanfile);
  }

  void AddSliceProto(LayerProto* layer,
                     int dim) {
    layer->mutable_slice_conf()->set_slice_dim(dim);
  }


  //----------------------------------------
  // Parameter
  //----------------------------------------
  void AddParamProto(LayerProto* layer, const char* param_name,
                     float lr_scale=1, float wd_scale=1) {
    if(param_name!=nullptr) {
      ParamProto* pp = layer->add_param();
      pp->set_name(param_name);
      pp->set_lr_scale(lr_scale);
      pp->set_wd_scale(wd_scale);
    }
  }

  template<class T>
  void SetParamProto(LayerProto* layer, int param_index,
                     const char* key, T val) {
    ParamProto* pp = layer->mutable_param(param_index);
    SetVal(pp, key, val);
  }

  void SetParamProto(vector<LayerProto*>* out, LayerType type, const char* param_name) {
    for(int i=0; i<static_cast<int>(out->size()); i++) {
      if(out->at(i)->type() == type) {
        AddParamProto(out->at(i), generateName(param_name, 1));
        uniqID[1]++;
      }
    }
  }

  void SetParamProto(const char* param_name,
                     float lr_scale=1, float wd_scale=1) {
    if(param_name!=nullptr) {
      ParamProto* pp = GetParamByName(param_name);
      pp->set_lr_scale(lr_scale);
      pp->set_wd_scale(wd_scale);
    }
  }


  void SetParamAutoEncoder(vector<LayerProto*>* out) {
    int num_layers = out->size()-1;
    for(int i=num_layers; i>=num_layers/2; i=i-2) { 
      LayerProto* layer = out->at(i-1); // InnerProduct layer
      layer->mutable_innerproduct_conf()->set_transpose(true);
      ParamProto* pp = out->at(num_layers-i)->mutable_param(0);
      SetParamProto(layer, 0, "share_from", pp->name().c_str());
    }
  }

  void SetParamProtoUniform(vector<LayerProto*>* out, LayerType type,
                            const char* param_name, float low, float high) {
    for(int i=0; i<static_cast<int>(out->size()); i++) {
      if(out->at(i)->type() == type) {
        AddParamProto(out->at(i), generateName(param_name, 1));
        AddUniformProto(generateName(param_name, 1), low, high);
        uniqID[1]++;
      }
    }
  }

  void SetParamProtoGaussian(vector<LayerProto*>* out, LayerType type,
                            const char* param_name, float mean, float std) {
    for(int i=0; i<static_cast<int>(out->size()); i++) {
      if(out->at(i)->type() == type) {
        AddParamProto(out->at(i), generateName(param_name, 1));
        AddGaussianProto(generateName(param_name, 1), mean, std);
        uniqID[1]++;
      }
    }
  }

  void SetParamProtoConstant(vector<LayerProto*>* out, LayerType type,
                            const char* param_name,
                            float lr_scale=1, float wd_scale=1,
                            float value=0) {
    for(int i=0; i<static_cast<int>(out->size()); i++) {
      if(out->at(i)->type() == type) {
        AddParamProto(out->at(i), generateName(param_name, 1), lr_scale, wd_scale);
        AddConstantProto(generateName(param_name, 1), value);
        uniqID[1]++;
      }
    }
  }

  // ParamGenProto -----------------------------------------
  void AddConstantProto(const char* param_name=nullptr,
                        float value=1.0) {
    if(param_name!=nullptr) {
      ParamProto* pp = GetParamByName(param_name);
      pp->mutable_init()->set_type(kConstant);
      pp->mutable_init()->set_value(value);
    }
  }

  void AddUniformProto(const char* param_name=nullptr,
                       float low=-1.0, float high=1.0) {
    if(param_name!=nullptr) {
      ParamProto* pp = GetParamByName(param_name);
      pp->mutable_init()->set_type(kUniform);
      pp->mutable_init()->set_low(low);
      pp->mutable_init()->set_high(high);
    }
  }

  void AddGaussianProto(const char* param_name=nullptr,
                       float mean=0.0, float std=1.0) {
    if(param_name!=nullptr) {
      ParamProto* pp = GetParamByName(param_name);
      pp->mutable_init()->set_type(kGaussian);
      pp->mutable_init()->set_mean(mean);
      pp->mutable_init()->set_std(std);
    }
  }

  template<class T>
  void SetGaussianProto(LayerProto* layer,
                        const char* key, T val) {
    for(int i=0; i<layer->param_size(); i++) {
      ParamGenProto* pgp = layer->mutable_param(i)->mutable_init();
      if(pgp->type() == kGaussian)
        SetVal(pgp, key, val);
    }
  }
  // ParamGenProto end ------------------------------------------

  // Param + ParamGenProto ------------------------------------------
  void AddParamConstantProto(LayerProto* layer,
                            const char* param_name,
                            float lr_scale=1, float wd_scale=1,
                            float value=1) {
    AddParamProto(layer, param_name, lr_scale, wd_scale);
    AddConstantProto(param_name, value);
  }

  void AddParamUniformProto(LayerProto* layer,
                            const char* param_name,
                            float lr_scale=1, float wd_scale=1,
                            float low=-1, float high=1) {
    AddParamProto(layer, param_name);
    AddUniformProto(param_name, low, high);
  }

  void AddParamGaussianProto(LayerProto* layer,
                            const char* param_name,
                            float lr_scale=1, float wd_scale=1,
                            float mean=0, float std=1) {
    AddParamProto(layer, param_name, lr_scale, wd_scale);
    AddGaussianProto(param_name, mean, std);
  }
  // Param + ParamGenProto end ------------------------------------------



  //------------------------------------------------------------------
  // Cluster 
  //------------------------------------------------------------------
  void SetCluster(const char* workspace, int nwg=1, int nsg=1) {
    cluster->set_nworker_groups(nwg);
    cluster->set_nserver_groups(nsg);
    cluster->set_workspace(workspace);
    // TODO: more parameters
  } 

  //------------------------------------------------------------------
  // TrainAlgo 
  //------------------------------------------------------------------
  void AddTrainAlgo(AlgType type) {
    jobconf->mutable_train_one_batch()->set_alg(type);
  }

  //------------------------------------------------------------------
  // Updater 
  //------------------------------------------------------------------
  void AddUpdater(UpdaterType type, float weight_decay=0, float momentum=0) {
    updater->set_type(type);
    if(weight_decay!=0)
      updater->set_weight_decay(weight_decay);
    if(momentum!=0)
      updater->set_momentum(momentum);
    // TODO: more parameters
  }

  // RMS Prop algorithm ------------------------
  void SetRMSProp(float rho) {
    updater->mutable_rmsprop_conf()->set_rho(rho);
  }

  // Learning Rate Generator ------------------------
  void SetLRGen(ChangeMethod type, float base_lr=0) {
    updater->mutable_learning_rate()->set_type(type);
    if(base_lr>0)
      updater->mutable_learning_rate()->set_base_lr(base_lr);
  }

  void SetStep(float gamma, int change_freq) {
    LRGenProto* lrgen = updater->mutable_learning_rate();
    lrgen->mutable_step_conf()->set_gamma(gamma);
    lrgen->mutable_step_conf()->set_change_freq(change_freq);
  }

  void SetFixedStep(int step=0, float step_lr=0) {
    LRGenProto* lrgen = updater->mutable_learning_rate();
    lrgen->mutable_fixedstep_conf()->add_step(step);
    lrgen->mutable_fixedstep_conf()->add_step_lr(step_lr);
  }

  void SetLinear(int change_freq, float final_lr) {
    LRGenProto* lrgen = updater->mutable_learning_rate();
    lrgen->mutable_linear_conf()->set_change_freq(change_freq);
    lrgen->mutable_linear_conf()->set_final_lr(final_lr);
  }

  void SetExponential(int change_freq) {
    LRGenProto* lrgen = updater->mutable_learning_rate();
    lrgen->mutable_exponential_conf()->set_change_freq(change_freq);
  }


  //------------------------------
  // Getters
  //------------------------------
  LayerProto* GetLayerProtoByName(const char* layer_name, Phase exclude=kUnknown) {
    for(int i=0; i<netconf->layer_size(); i++) {
      layer = netconf->mutable_layer(i);
      bool flag = true;
      if(strcmp(layer_name, layer->name().c_str())==0) {
        for(int j=0; j<layer->exclude_size(); j++) {
          if((int)exclude==(int)layer->exclude(j)) 
            flag = false;
        }
        if(flag)
          return layer;
      }
    }
    return nullptr;
  }

  ParamProto* GetParamByName(const char* param_name) {
    for(int i=0; i<netconf->layer_size(); i++) {
      for(int j=0; j<netconf->mutable_layer(i)->param_size(); j++) {
        if(strcmp(param_name, netconf->mutable_layer(i)->mutable_param(j)->name().c_str())==0)
          return netconf->mutable_layer(i)->mutable_param(j);
      }
    }
    return nullptr;
  }

  //------------------------------
  // Setters
  //------------------------------
  void setModelName(const char* name) {
    jobconf->set_name(name);
  }

  void SetTrainConfig(int step, int dispfreq, int dispafter=0) {
    jobconf->set_train_steps(step);
    jobconf->set_disp_freq(dispfreq);
    //jobconf->set_disp_after(dispafter);
  }

  void SetTestConfig(int step, int freq) {
    jobconf->set_test_steps(step);
    jobconf->set_test_freq(freq);
  }

  void SetCheckpoint(const char* path, int freq=0) {
    jobconf->add_checkpoint_path(string(path));
    jobconf->set_checkpoint_freq(freq);
  }

  void SetSrclayer(LayerProto* layer, LayerProto* srclayer) {
    layer->add_srclayers(srclayer->name());
  }

  void SetVal(Message* msg, const char* key, int value) {
    const FieldDescriptor* fd = msg->GetDescriptor()->FindFieldByName(key);
    CHECK(fd!=NULL) << key << " not found";
    if(fd->type()==FieldDescriptor::TYPE_ENUM) {
      // TODO: is there better way???
      const EnumDescriptor* ed = msg->GetDescriptor()->enum_type(0);
      if(fd->is_repeated())
        msg->GetReflection()->AddEnum(msg, fd, ed->FindValueByNumber(value));
      else
        msg->GetReflection()->SetEnum(msg, fd, ed->FindValueByNumber(value));
    }
    else {
      if(fd->is_repeated())
        msg->GetReflection()->AddInt32(msg, fd, value);
      else
        msg->GetReflection()->SetInt32(msg, fd, value);
    }
  }
  void SetVal(Message* msg, const char* key, float value) {
    const FieldDescriptor* fd = msg->GetDescriptor()->FindFieldByName(key);
    CHECK(fd!=NULL) << key << " not found";
    if(fd->is_repeated())
      msg->GetReflection()->AddFloat(msg, fd, value);
    else
      msg->GetReflection()->SetFloat(msg, fd, value);
  }
  void SetVal(Message* msg, const char* key, const char* value) {
    const FieldDescriptor* fd = msg->GetDescriptor()->FindFieldByName(key);
    CHECK(fd!=NULL) << key << " not found";
    if(fd->is_repeated())
      msg->GetReflection()->AddString(msg, fd, value);
    else
      msg->GetReflection()->SetString(msg, fd, value);
  }
  void SetVal(Message* msg, const char* key, bool value) {
    const FieldDescriptor* fd = msg->GetDescriptor()->FindFieldByName(key);
    CHECK(fd!=NULL) << key << " not found";
    if(fd->is_repeated())
      msg->GetReflection()->AddBool(msg, fd, value);
    else
      msg->GetReflection()->SetBool(msg, fd, value);
  }


private:

  JobProto* jobconf;
  NetProto* netconf;
  UpdaterProto* updater;
  ClusterProto* cluster;
  LayerProto* layer;


  void Init(JobProto* conf, const char* name);

  const char* generateName(const char* name, int op) {
     return (name+std::to_string(uniqID[op])).c_str();
  }

  const char* generateName(LayerType type) {
     string name="";
     switch(type) {
     case kInnerProduct: name="fc"; break;
     case kSTanh: name="tanh"; break;
     case kCConvolution: name="conv"; break;
     case kCPooling: name="pool"; break;
     case kReLU: name="relu"; break;
     case kLRN: name="norm"; break;
     default:
       break;
     }
     return (name+std::to_string(uniqID[0])).c_str();
  }

};

}  // namespace singa

#endif  // SINGA_BUILDER_H_
