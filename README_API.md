### Primitive
- set value to normal field
- add value to repeated field

```
void SetVal(Message* msg, const char* key, int value)
void SetVal(Message* msg, const char* key, float value)
void SetVal(Message* msg, const char* key, const char* value)
void SetVal(Message* msg, const char* key, bool value)
```


### Layer
- handling a layer

```
LayerProto* AddLayer(LayerType type)
LayerProto* AddLayer(LayerType type, const char* name)
LayerProto* AddLayer(LayerType type, const char* name, LayerProto* srclayer)
LayerProto* AddLayer(LayerType type, const char* name, int num_srclayers, LayerProto* ...)
```

- handling multiple layers

```
void AddLayers(vector<LayerProto*>* out, LayerProto* srclayer, int num_stacks,
               int num_layers, LayerProto* ...)
               
void ReplicateLayers(vector<LayerProto*>* out, LayerProto* srclayer, int num_stacks,
                     int num_layers, LayerProto* ...)
```

```
LayerProto* GetLayerByName(const char* layer_name, Phase exclude=kUnknown)
```


#### for InputLayer
- I need to chage to adapt new proto, e.g., kRecordInput
- change exclude to include

```
void AddDataProto(LayerProto* layer, Phase mode, const char* path, int batchsize, int random_skip)
```

#### for ParserLayer
```
void AddMnistProto(LayerProto* layer, float norm_a,  float norm_b)
```

#### for NeuronLayer
```
void AddInnerProductProto(LayerProto* layer, int num_output=0)
```
```
void AddConvolutionProto(LayerProto* layer,
                         int num_filters, int kernel,
                         int pad=0, int stride=0, bool bias_term=true)
```                     
```
void AddPoolingProto(LayerProto* layer,  int kernel)
void AddPoolingProto(LayerProto* layer,  int kernel, PoolingProto::PoolMethod pool)
```                       
```
void AddDropoutProto(LayerProto* layer,  float ratio)
```                       
```
void AddRBMProto(LayerProto* layer, int hdim)
void AddRBMProto(LayerProto* layer, int hdim, bool bias_term, bool gaussian)
```                   
```
void AddReLUProto(LayerProto* layer,
                  float negative_slope)
```
```
void AddLRNProto(LayerProto* layer,
                 int local_size=5, float alpha=1.0, float beta=0.75, float knorm=1.0)
```                 
```
void AddRGBImageProto(LayerProto* layer,
                      float scale=0, int cropsize=0, bool mirror=false, const char* meanfile=nullptr)
```
```
void AddSliceProto(LayerProto* layer,  int dim) 
```


#### for LossLayer
```
void AddSoftmaxLossProto(LayerProto* layer, int topk=0, float scale=0.0)
```



### Parameter Proto
```
void AddParamProto(LayerProto* layer, const char* param_name,
                   float lr_scale=1, float wd_scale=1)
```                   
```
template<class T>
void SetParamProto(LayerProto* layer, int param_index, const char* key, T val)

void SetParamProto(vector<LayerProto*>* out, LayerType type, const char* param_name) 

void SetParamProto(const char* param_name, float lr_scale=1, float wd_scale=1) 
```

```
void SetParamAutoEncoder(vector<LayerProto*>* out) 
```


```
ParamProto* GetParamByName(const char* param_name)
```

#### for ParamGenProto
```
void AddConstantProto(const char* param_name=nullptr, float value=1.0)
void AddUniformProto(const char* param_name=nullptr, float low=-1.0, float high=1.0)
void AddGaussianProto(const char* param_name=nullptr, float mean=0.0, float std=1.0)

```
```
template<class T>
void SetConstantProto(LayerProto* layer, const char* key, T val)
template<class T>
void SetUniformProto(LayerProto* layer, const char* key, T val)
template<class T>
void SetGaussianProto(LayerProto* layer, const char* key, T val)
```
```
void SetParamProtoConstant(vector<LayerProto*>* out, LayerType type,  const char* param_name,
                          float lr_scale=1, float wd_scale=1, float value=0)
void SetParamProtoUniform(vector<LayerProto*>* out, LayerType type, const char* param_name,
                          float low, float high)
void SetParamProtoGaussian(vector<LayerProto*>* out, LayerType type, const char* param_name,
                          float mean, float std)
```

#### for Param + ParamGenProto
```
void AddParamConstantProto(LayerProto* layer, const char* param_name,
                            float lr_scale=1, float wd_scale=1,
                            float value=1)

void AddParamUniformProto(LayerProto* layer, const char* param_name,
                            float lr_scale=1, float wd_scale=1,
                            float low=-1, float high=1)

void AddParamGaussianProto(LayerProto* layer, const char* param_name,
                            float lr_scale=1, float wd_scale=1,
                            float mean=0, float std=1)
```

## Example MLP (Low flexibility, i.e., use default value)

```
  AddTrainAlgo(kBP);
  AddCluster("examples/mnist");
  AddUpdater(kSGD);

  LayerProto* L1 = AddLayer(kShardData);
  AddDataProto(L1, kTrain, "examples/train_shard", 1000);

  LayerProto* L2 = AddLayer(kShardData);
  AddDataProto(L2, kTest, "examples/test_shard", 1000);

  LayerProto* L3 = AddLayer(kLabel, L1);

  LayerProto* L4 = AddLayer(kMnist, L1);
  
  vector<LayerProto*> HLs;
  AddLayers(&HLs, L4, 6, 2, kInnerProduct, kSTanh);
  SetInnerProductProto(&HLs, 6, 2500, 2000, 1500, 1000, 500, 10);
  
  LayerProto* L7 = AddLayer(kSoftmaxLoss, 2, HLs.back(), L3); // order! 
```


## Example MLP
```
  SetTrainConfig(1000, 100, 500);
  SetTestConfig(10, 60);
  AddTrainAlgo(kBP);

  AddCluster("examples/mnist");

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

  LayerProto* L7 = AddLayer(kSoftmaxLoss, "loss", 2, HLs.back(), L3); // order!
  AddSoftmaxLossProto(L7, 3);
```

## Example CONV
```
  SetTrainConfig(1000, 30, 100);
  SetTestConfig(100, 300);
  AddTrainAlgo(kBP);

  AddCluster("examples/cifar10");

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
  SetParamProtoConstant(&HLs, kCConvolution, "b", 2.0, 1.0, 0);
  AddPoolingProto(HLs.at(1), 3, PoolingProto::MAX, 0, 2);
  AddLRNProto(HLs.at(3), 3, 5e-05);

  vector<LayerProto*> HLs2;
  ReplicateLayers(&HLs2, HLs.back(), 1, 4, HLs.at(0), HLs.at(2), HLs.at(1), HLs.at(3));
  SetGaussianProto<float>(HLs2.at(0), "std", 0.01);
  SetPoolingProto(HLs2.at(2), "pool", PoolingProto::AVG);

  vector<LayerProto*> HLs3;
  ReplicateLayers(&HLs3, HLs2.back(), 1, 3, HLs2.at(0), HLs2.at(1), HLs2.at(2));
  SetConvolutionProto(HLs3.at(0), "num_filters", 64);
  SetParamProto(HLs3.at(0), 1, "lr_scale", 1.0f);
  
  LayerProto* L9 = AddLayer(kInnerProduct, "ip", HLs3.back());
  AddInnerProductProto(L9, 10);
  AddParamGaussianProto(L9, "wi", 1.0, 250, 0, 0.01);
  AddParamConstantProto(L9, "bi", 2.0, 0, 0);

  // order of srclayers!!!
  LayerProto* L10 = AddLayer(kSoftmaxLoss, "loss", 2, L9, L4);
  AddSoftmaxLossProto(L10, 1);

```

## Example Autoencoder
```
  SetTrainConfig(12200, 100);
  SetTestConfig(100, 1000);
  // these checkpoint can be set by one function at higher level
  SetCheckpoint("examples/rbm/rbm1/checkpoint/step6000-worker0");
  SetCheckpoint("examples/rbm/rbm2/checkpoint/step6000-worker0");
  ...
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
```

