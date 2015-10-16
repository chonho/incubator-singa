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

#include "builder.h"

#include <string>
#include <vector>

namespace singa {
using std::string;

// id for layer name, and param name
int Builder::uniqID[] = {1, 1};

Builder::Builder(JobProto* conf) {
  Init(conf, "model name");
}

Builder::Builder(JobProto* conf, const char* name) {
  Init(conf, name);
}

void Builder::Init(JobProto* conf, const char* name) {
  jobconf = conf;
  jobconf->set_name(string(name));
  netconf = jobconf->mutable_neuralnet();
  updater = jobconf->mutable_updater();
  cluster = jobconf->mutable_cluster();
}

void Builder::Construct() {

  AUTOENCODER_ex1(); 

}

void Builder::AUTOENCODER_ex1() {
  SetTrainConfig(12200, 100);
  SetTestConfig(100, 1000);
  SetCheckpoint("examples/rbm/rbm1/checkpoint/step6000-worker0");
  SetCheckpoint("examples/rbm/rbm2/checkpoint/step6000-worker0");
  AddTrainAlgo(kBP);

  AddUpdater(kAdaGrad);
  SetLRGen(kFixed, 0.01);

  LayerProto* L1 = AddLayer(kShardData, "data");
  AddDataProto(L1, kTrain, "examples/mnist/mnist_train_shard", 1000);
  LayerProto* L2 = AddLayer(kShardData, "data");
  AddDataProto(L2, kTest, "examples/mnist/mnist_test_shard", 1000);
  
  LayerProto* L3 = AddLayer(kMnist, "mnist", L1);
  AddMnistProto(L3, 255, 0);
 
  vector<LayerProto*> HLs;
  AddLayers(&HLs, L3, 8, 2, kInnerProduct, kSigmoid);
  SetInnerProductProto(&HLs, 8, 1000, 500, 250, 30, 250, 500, 1000, 784);
  SetParamProto(&HLs, kInnerProduct, "w");
  SetParamProto(&HLs, kInnerProduct, "b");
  SetParamAutoEncoder(&HLs);

  LayerProto* L4 = AddLayer(kEuclideanLoss, "loss", 2, HLs.back(), L3);
}

void Builder::CONV_ex3() {
  SetTrainConfig(1000, 30, 100);
  SetTestConfig(100, 300);
  AddTrainAlgo(kBP);

  SetCluster("examples/cifar10");

  AddUpdater(kSGD, 0.004);
  SetLRGen(kFixedStep);
  SetFixedStep(0, 0.001);
  SetFixedStep(60000, 0.0001);
  SetFixedStep(65000, 0.00001);

  LayerProto* L1 = AddLayer(kShardData, "data");
  AddDataProto(L1, kTrain, "examples/cifar10/cifar10_train_shard", 64, 5000);
  LayerProto* L2 = AddLayer(kShardData, "data");
  AddDataProto(L2, kTest, "examples/cifar10/cifar10_test_shard", 100);

  LayerProto* L3 = AddLayer(kRGBImage, "rgb", L1);
  AddRGBImageProto(L3, "examples/cifar10/image_mean.bin");

  LayerProto* L4 = AddLayer(kLabel, "label", L1);

  vector<LayerProto*> HLs;
  AddLayers(&HLs, L3, 1, 4, kCConvolution, kCPooling, kReLU, kLRN);
  AddConvolutionProto(HLs.at(0), 32, 5, 2, 1, false);
  SetParamProtoGaussian(&HLs, kCConvolution, "w", 0, 0.0001);
  SetParamProtoConstant(&HLs, kCConvolution, "b", 2.0f, 1.0f, 0);
  AddPoolingProto(HLs.at(1), 3, PoolingProto::MAX, 0, 2);
  AddLRNProto(HLs.at(3), 3, 5e-05);

  vector<LayerProto*> HLs2;
  ReplicateLayers(&HLs2, HLs.back(), 1, 4, HLs.at(0), HLs.at(2), HLs.at(1), HLs.at(3));
  SetGaussianProto<float>(HLs2.at(0), "std", 0.01);
  SetPoolingProto(HLs2.at(2), "pool", PoolingProto::AVG);

  vector<LayerProto*> HLs3;
  ReplicateLayers(&HLs3, HLs2.back(), 1, 3, HLs2.at(0), HLs2.at(1), HLs2.at(2));
  SetConvolutionProto(HLs3.at(0), "num_filters", 64);
  SetParamProto(HLs3.at(0), 1, "lr_scale", 1.0f); // change b3 value

  LayerProto* L9 = AddLayer(kInnerProduct, "ip", HLs3.back());
  AddInnerProductProto(L9, 10);
  AddParamGaussianProto(L9, "wi", 1.0, 250, 0, 0.01);
  AddParamConstantProto(L9, "bi", 2.0, 0, 0);

  // order of srclayers!!!
  LayerProto* L10 = AddLayer(kSoftmaxLoss, "loss", 2, L9, L4);
  AddSoftmaxLossProto(L10, 1);
}

void Builder::CONV_ex2() {
  SetTrainConfig(1000, 30, 100);
  SetTestConfig(100, 300);
  AddTrainAlgo(kBP);

  SetCluster("examples/cifar10");

  AddUpdater(kSGD);
  SetLRGen(kFixedStep);
  SetFixedStep(0, 0.001);
  SetFixedStep(60000, 0.0001);
  SetFixedStep(65000, 0.00001);

  LayerProto* L1 = AddLayer(kShardData, "data");
  AddDataProto(L1, kTrain, "examples/cifar10/cifar10_train_shard", 64, 5000);
  LayerProto* L2 = AddLayer(kShardData, "data");
  AddDataProto(L2, kTest, "examples/cifar10/cifar10_test_shard", 100);

  LayerProto* L3 = AddLayer(kRGBImage, "rgb", L1);
  AddRGBImageProto(L3, "examples/cifar10/image_mean.bin");

  LayerProto* L4 = AddLayer(kLabel, "label", L1);

  LayerProto* L5 = AddLayer(kCConvolution, "conv", L3);
  AddConvolutionProto(L5, 32, 5, 2, 1, false);
  AddParamGaussianProto(L5, "w", 0, 0.0001);
  AddParamConstantProto(L5, "b", 0);
  SetParamProto("b", 2.0, 0);

  LayerProto* L6 = AddLayer(kCPooling, "pool", L5);
  AddPoolingProto(L6, 3, PoolingProto::MAX);

  LayerProto* L7 = AddLayer(kReLU, "relu", L6);

  LayerProto* L8 = AddLayer(kLRN, "norm", L7);
  AddLRNProto(L8, 3, 5e-05);

  vector<LayerProto*> Ls;
  ReplicateLayers(&Ls, L8, 2, 4, L5, L6, L7, L8);
  ReplicateLayers(&Ls, Ls.back(), 1, 3, Ls.at(0), Ls.at(1), Ls.at(2));

  LayerProto* L9 = AddLayer(kInnerProduct, "ip", Ls.back());
  AddInnerProductProto(L9, 10);
  AddParamGaussianProto(L9, "w4", 0, 0.01);
  AddParamConstantProto(L9, "b4", 0);
  SetParamProto("b4", 2.0, 0);

  LayerProto* L10 = AddLayer(kSoftmaxLoss, "loss", 2, L4, L9);
  AddSoftmaxLossProto(L10, 1);
}

void Builder::CONV_ex1() {
  SetTrainConfig(1000, 30, 100);
  SetTestConfig(100, 300);
  AddTrainAlgo(kBP);

  SetCluster("examples/cifar10");

  AddUpdater(kSGD);
  SetLRGen(kFixedStep);
  SetFixedStep(0, 0.001);
  SetFixedStep(60000, 0.0001);
  SetFixedStep(65000, 0.00001);

  LayerProto* L1 = AddLayer(kShardData, "data");
  AddDataProto(L1, kTrain, "examples/cifar10/cifar10_train_shard", 64, 5000);
  LayerProto* L2 = AddLayer(kShardData, "data");
  AddDataProto(L2, kTest, "examples/cifar10/cifar10_test_shard", 100);

  LayerProto* L3 = AddLayer(kRGBImage, "rgb", L1);
  AddRGBImageProto(L3, "examples/cifar10/image_mean.bin");

  LayerProto* L4 = AddLayer(kLabel, "label", L1);

  LayerProto* L5 = AddLayer(kCConvolution, "conv", L3);
  AddConvolutionProto(L5, 32, 5, 2, 1, false);
  AddParamGaussianProto(L5, "w", 0, 0.0001);
  AddParamConstantProto(L5, "b", 0);
  SetParamProto("b", 2.0, 0);

  LayerProto* L6 = AddLayer(kCPooling, "pool", L5);
  AddPoolingProto(L6, 3, PoolingProto::MAX);

  LayerProto* L7 = AddLayer(kReLU, "relu", L6);

  LayerProto* L8 = AddLayer(kLRN, "norm", L7);
  AddLRNProto(L8, 3, 5e-05);
  
  vector<LayerProto*> Ls;
  //AddLayers(&Ls, L8, 2, 4, L5, L6, L7, L8);
  ReplicateLayers(&Ls, L8, 2, 4, L5, L6, L7, L8);
  ReplicateLayers(&Ls, Ls.back(), 1, 3, Ls.at(0), Ls.at(1), Ls.at(2));

  LayerProto* L9 = AddLayer(kInnerProduct, "ip", Ls.back());
  AddInnerProductProto(L9, 10);
  AddParamGaussianProto(L9, "w4", 0, 0.01);
  AddParamConstantProto(L9, "b4", 0);
  SetParamProto("b4", 2.0, 0);

  LayerProto* L10 = AddLayer(kSoftmaxLoss, "loss", 2, L4, L9);
  AddSoftmaxLossProto(L10, 1);
}

void Builder::MLP_ex2() {
  SetTrainConfig(1000, 100, 500);
  SetTestConfig(10, 60);
  AddTrainAlgo(kBP);

  SetCluster("examples/mnist");

  // AddUpdater(kSGD, LRGen(kStep, 0.001), Step(0.997, 60));
  AddUpdater(kSGD);
  SetLRGen(kStep, 0.001);
  SetStep(0.997, 60);
 
  LayerProto* L1 = AddLayer(kShardData, "data");
  AddDataProto(L1, kTrain, "examples/train_shard", 1000);

  LayerProto* L2 = AddLayer(kShardData, "data");
  AddDataProto(L2, kTest, "examples/test_shard", 1000);

  LayerProto* L3 = AddLayer(kLabel, "label", L1);

  LayerProto* L4 = AddLayer(kMnist, "mnist", L1);
  AddMnistProto(L4, 127.5, 1);

  vector<LayerProto*> HLs;
  AddLayers(&HLs, L4, 6, 2, kInnerProduct, kSTanh);
  SetInnerProductProto(&HLs, 6, 2500, 2000, 1500, 1000, 500, 10);
  SetParamProtoUniform(&HLs, kInnerProduct, "w", -0.05, 0.05);
  SetParamProtoUniform(&HLs, kInnerProduct, "b", -0.05, 0.05);
  
  LayerProto* L7 = AddLayer(kSoftmaxLoss, "loss", 2, L3, HLs.back());
  AddSoftmaxLossProto(L7, 3);

}

void Builder::MLP_ex1() {
  SetTrainConfig(1000, 100, 500);
  SetTestConfig(10, 60);
  AddTrainAlgo(kBP);
	
  SetCluster("examples/mnist");

  AddUpdater(kSGD);
  SetLRGen(kStep, 0.001);
  SetStep(0.997, 60);
 
  LayerProto* L1 = AddLayer(kShardData, "data");
  AddDataProto(L1, kTrain, "examples/train_shard", 1000);

  LayerProto* L2 = AddLayer(kShardData, "data");
  AddDataProto(L2, kTest, "examples/test_shard", 1000);

  LayerProto* L3 = AddLayer(kLabel, "label", L1);

  LayerProto* L4 = AddLayer(kMnist, "mnist", L1);
  AddMnistProto(L4, 127.5, 1);

  LayerProto* L5 = AddLayer(kInnerProduct, "fc1", L4);
  AddInnerProductProto(L5, 2500);
  AddParamProto(L5, "w1");
  AddUniformProto("w1", -0.05, 0.05);
  AddParamProto(L5, "b1");
  AddUniformProto("b1", -0.05, 0.05);

  LayerProto* L6 = AddLayer(kSTanh, "tanh1", L5);

  LayerProto* L7 = AddLayer(kInnerProduct, "fc2", L6);
  AddInnerProductProto(L7, 2000);
  AddParamProto(L7, "w2");
  AddUniformProto("w2", -0.05, 0.05);
  AddParamProto(L7, "b2");
  AddUniformProto("b2", -0.05, 0.05);

  LayerProto* L8 = AddLayer(kSTanh, "tanh2", L7);

  LayerProto* L9 = AddLayer(kInnerProduct, "fc3", L8);
  AddInnerProductProto(L9, 1000);
  AddParamProto(L9, "w3");
  AddUniformProto("w3", -0.05, 0.05);
  AddParamProto(L9, "b3");
  AddUniformProto("b3", -0.05, 0.05);

  LayerProto* L10 = AddLayer(kSoftmaxLoss, "loss", L9);
  SetSrclayer(L10, L3);
  AddSoftmaxLossProto(L10, 3);

}


void Builder::Display() {
  string s;
  google::protobuf::TextFormat::PrintToString(*jobconf, &s);
  //google::protobuf::TextFormat::PrintToString(*netconf_, &s);
  printf("%s\n", s.c_str());
}

}  // namespace singa
