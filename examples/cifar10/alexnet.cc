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
#include "./cifar10.h"
#include "singa/model/feed_forward_net.h"
#include "singa/model/optimizer.h"
#include "singa/model/initializer.h"
#include "singa/model/metric.h"
#include "singa/utils/channel.h"
#include "singa/utils/string.h"
#include "../../src/model/layer/cudnn_convolution.h"
#include "../../src/model/layer/cudnn_activation.h"
#include "../../src/model/layer/cudnn_pooling.h"
#include "../../src/model/layer/cudnn_lrn.h"
#include "../../src/model/layer/dense.h"
#include "../../src/model/layer/flatten.h"
namespace singa {

LayerConf GenConvConf(string name, int nb_filter, int kernel, int stride,
                      int pad, float std) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnConvolution");
  ConvolutionConf *conv = conf.mutable_convolution_conf();
  conv->set_num_output(nb_filter);
  conv->add_kernel_size(kernel);
  conv->add_stride(stride);
  conv->add_pad(pad);
  conv->set_bias_term(true);

  ParamSpec *wspec = conf.add_param();
  wspec->set_name(name + "_weight");
  auto wfill = wspec->mutable_filler();
  wfill->set_type("Gaussian");
  wfill->set_std(std);

  ParamSpec *bspec = conf.add_param();
  bspec->set_name(name + "_bias");
  bspec->set_lr_mult(2);
//  bspec->set_decay_mult(0);
  return conf;
}

LayerConf GenPoolingConf(string name, bool max_pool, int kernel, int stride,
                         int pad) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnPooling");
  PoolingConf *pool = conf.mutable_pooling_conf();
  pool->set_kernel_size(kernel);
  pool->set_stride(stride);
  pool->set_pad(pad);
  if (!max_pool) pool->set_pool(PoolingConf_PoolMethod_AVE);
  return conf;
}

LayerConf GenReLUConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("RELU");
  return conf;
}

LayerConf GenDenseConf(string name, int num_output, float std, float wd) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("Dense");
  DenseConf *dense = conf.mutable_dense_conf();
  dense->set_num_output(num_output);

  ParamSpec *wspec = conf.add_param();
  wspec->set_name(name + "_weight");
  wspec->set_decay_mult(wd);
  auto wfill = wspec->mutable_filler();
  wfill->set_type("Gaussian");
  wfill->set_std(std);

  ParamSpec *bspec = conf.add_param();
  bspec->set_name(name + "_bias");
  bspec->set_lr_mult(2);
  bspec->set_decay_mult(0);

  return conf;
}

LayerConf GenLRNConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnLRN");
  LRNConf *lrn = conf.mutable_lrn_conf();
  lrn->set_local_size(3);
  lrn->set_alpha(5e-05);
  lrn->set_beta(0.75);
  return conf;
}

LayerConf GenFlattenConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("Flatten");
  return conf;
}

FeedForwardNet CreateNet() {
  FeedForwardNet net;
  Shape s{3, 32, 32};

  net.Add(new CudnnConvolution(), GenConvConf("conv1", 32, 5, 1, 2, 0.0001),
          &s);
  net.Add(new CudnnPooling(), GenPoolingConf("pool1", true, 3, 2, 1));
  net.Add(new CudnnActivation(), GenReLUConf("relu1"));
  net.Add(new CudnnLRN(), GenLRNConf("lrn1"));
  net.Add(new CudnnConvolution(), GenConvConf("conv2", 32, 5, 1, 2, 0.01));
  net.Add(new CudnnActivation(), GenReLUConf("relu2"));
  net.Add(new CudnnPooling(), GenPoolingConf("pool2", false, 3, 2, 1));
  net.Add(new CudnnLRN(), GenLRNConf("lrn2"));
  net.Add(new CudnnConvolution, GenConvConf("conv3", 64, 5, 1, 2, 0.01));
  net.Add(new CudnnActivation(), GenReLUConf("relu3"));
  net.Add(new CudnnPooling(), GenPoolingConf("pool3", false, 3, 2, 1));
  net.Add(new Flatten(), GenFlattenConf("flat"));
  net.Add(new Dense(), GenDenseConf("ip", 10, 0.01, 250));
  return net;
}

void Train(float lr, int num_epoch, string data_dir) {
  Cifar10 data(data_dir);
  Tensor train_x, train_y, test_x, test_y;
  {
    auto train = data.ReadTrainData();
    size_t nsamples = train.first.shape(0);
    auto mtrain =
        Reshape(train.first, Shape{nsamples, train.first.Size() / nsamples});
    const Tensor& mean = Average(mtrain, 0);
    SubRow(mean, &mtrain);
    train_x = Reshape(mtrain, train.first.shape());
    train_y = train.second;
    auto test = data.ReadTestData();
    nsamples = test.first.shape(0);
    auto mtest =
        Reshape(test.first, Shape{nsamples, test.first.Size() / nsamples});
    SubRow(mean, &mtest);
    test_x = Reshape(mtest, test.first.shape());
    test_y = test.second;
  }
  CHECK_EQ(train_x.shape(0), train_y.shape(0));
  CHECK_EQ(test_x.shape(0), test_y.shape(0));
  LOG(INFO) << "Training samples = " << train_y.shape(0)
            << ", Test samples = " << test_y.shape(0);
  auto net = CreateNet();
  SGD sgd;
  OptimizerConf opt_conf;
  opt_conf.set_momentum(0.9);
  auto reg = opt_conf.mutable_regularizer();
  reg->set_coefficient(0.004);
  sgd.Setup(opt_conf);
  sgd.SetLearningRateGenerator([lr](int step) {
    if (step <= 120)
      return 0.001;
    else if (step <= 130)
      return 0.0001;
    else
      return 0.00001;
  });

  SoftmaxCrossEntropy loss;
  Accuracy acc;
  net.Compile(true, &sgd, &loss, &acc);

  auto cuda = std::make_shared<CudaGPU>();
  net.ToDevice(cuda);
  train_x.ToDevice(cuda);
  train_y.ToDevice(cuda);
  test_x.ToDevice(cuda);
  test_y.ToDevice(cuda);
  net.Train(100, num_epoch, train_x, train_y, test_x, test_y);
}
}

int main(int argc, char **argv) {
  singa::InitChannel(nullptr);
  int pos = singa::ArgPos(argc, argv, "-epoch");
  int nEpoch = 1;
  if (pos != -1) nEpoch = atoi(argv[pos + 1]);
  pos = singa::ArgPos(argc, argv, "-lr");
  float lr = 0.001;
  if (pos != -1) lr = atof(argv[pos + 1]);
  pos = singa::ArgPos(argc, argv, "-data");
  string data = "cifar-10-batches-bin";
  if (pos != -1) data = argv[pos + 1];

  LOG(INFO) << "Start training";
  singa::Train(lr, nEpoch, data);
  LOG(INFO) << "End training";
}