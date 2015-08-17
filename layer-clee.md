---
layout: post
title: Layer
category : docs
tags : [layer ]
---
{% include JB/setup %}

Layer is a core abstraction in SINGA. It performs a variety of feature
transformations for extracting high-level features, e.g., loading raw features,
parsing RGB values, doing convolution transformation, etc.

## Configuration

    layer {
      name: "data"
      type: kShardData
      sharddata_conf { }
      exclude: kTest
      partition_dim : 0
    }
    layer{
      name:"mnist"
      type: kMnist
      srclayers: "data"
      mnist_conf { }
    }
    layer{
      name: "fc1"
      type: kInnerProduct
      srclayers:"mnist"
      innerproduct_conf{ }
      param{ }
      param{ }
    }

The configurations of three layers from the [MLP example]({{ BASE_PATH }}/docs/mlp)
are shown above. There are some common fields for all kinds of layers:

  * name, a string used to differentiate two layers
  * type, an integer used for identifying a Layer subclass. The types of built-in
  layers are listed in [LayerType](). User-defined subclasses must not use the
  type ID of built-in subclasses.
  * srclayers, one or more layer names, for identifying the source layers.
  In SINGA, all connections are [converted]({{ BASE_PATH }}/docs/neural-net) to directed connections.
  * exclude, an enumerate value of type [Phase](), can be {kTest, kValidation,
  kTrain}. It is used to filter this layer when creating the
  [NeuralNet]({{ BASE_PATH }}/docs/neural-net) for the excluding phase. E.g.,
  the "data" layer would be filtered when creating the NeuralNet instance for test phase.
  * param, configuration for a [Param]({{ BASE_PATH }}/docs/param) instance.
  There can be multiple Param objects in one layer.
  * partition_dim, integer value indicating the partition dimension of this
  layer. -1 (the default value) for no partitioning, 0 for partitioning on batch dimension, 1 for
  partitioning on feature dimension. It is used by
  [CreateGraph]({{ BASE_PATH }}/docs/neural-net) for partitioning the neural net.

Different layers may have different configurations, these configurations are in
defined in `<type>_conf`, which is for a specific layer with type `<type>`,
e.g., the "data" layer has `sharddata_conf` and "fc1" layer has
`innerproduct_conf`. The specific configurations of built-in layers will be described at the end of this page.

## Base Layer class

All layer implementations must subclass the [base layer class]().

### Members

    LayerProto layer_proto_;
    Blob<float> data_, grad_;
    vector<Layer*> srclayers_, dstlayers_;

The base layer class keeps the user configuration in `layer_proto_`. Source
layers and destination layers are stored in `srclayers_` and `dstlayers_`.
Almost all layers has $b$ (mini-batch size) feature vectors, which are stored
in the `data_` [Blob]() (A Blob is a chunk of memory space, proposed in [Caffe]).
There are layers without feature vectors; instead, they use other
layers' feature vectors. In this case, the `data_` field is not used.
The `grad_` Blob is for storing the gradients of the
objective loss w.r.t. the `data_` Blob. It is necessary in [BP algorithm],
hence we put it as a member of the base class. For [CD algorithm], the `grad_`
field is not used; instead, the layer from RBM may have a Blob for the positive
phase feature and a Blob for the negative phase feature. For a recurrent layer
in RNN, the feature blob contains one vector per internal layer.

If a layer has parameters, these parameters are declared using type
[Param](). Since some layers do not have
parameters, we do not declare any `Param` in the base layer class.

### Functions

    virtual void Setup(const LayerProto& proto, int npartitions = 1);
    virtual void ComputeFeature(Phase phase, Metric* perf) = 0;
    virtual void ComputeGradient(Phase phase) = 0;

The `Setup` function reads user configuration, i.e. `proto`, and information
from source layers, e.g., mini-batch size,  to set the
shape of the `data_` (and `grad_`) field as well
as some other layer specific fields. If `npartitions` is larger than 1, then
users need to reduce the sizes of `data_`, `grad_` Blobs or Param objects. For
example, if the `partition_dim=0` and there is no source layer, e.g., this
layer is a (bottom) data layer, then its `data_` and `grad_` Blob should have
`b/npartitions` feature vectors; If the source layer is also partitioned on
dimension 0, then this layer should have the same number of feature vectors as
the source layer. More complex partition cases are discussed in
[Neural net partitioning]({{ BASE_PATH }}/docs/neural-net). Typically, the
setup function just set the shapes of `data_` Blobs and Param objects. Memory
will not be allocated until computation over the data structure happens.

The `ComputeFeature` function evaluates the feature blob by transforming (e.g.
convolution and pooling) features from the source layers.  `ComputeGradient`
computes the gradients of parameters associated with this layer.  These two
functions are invoked by the [TrainOneBatch]({{ BASE_PATH }}/docs/train-one-batch)
function during training. Hence, they should be consistent with the
`TrainOneBatch` function. Particularly, for feed-forward models and RNN models, they are
trained using BP algorithm, which requires each layer's `ComputeFeature`
function to compute `data_` based on source layers, and requires each layer's
`ComputeGradient` to compute gradients of parameters and source layers'
`grad_`. For energy models, e.g., RBM, they are trained by CD algorithm, which
requires each layer's `ComputeFeature` function to compute the feature vectors
for the positive phase or negative phase depending on the `phase` argument, and
requires the `ComputeGradient` function to only compute parameter gradients.
For some layers, e.g., loss layer or output layer, they can put the loss or
prediction result into the `metric` argument, which will be averaged and
displayed periodically.


## Implementing new layers

Users can extend the base layer class to implement their own feature transformation
logics as long as the two virtual functions are overridden to be consistent with
the `TrainOneBatch` function. The `Setup` function may also be overridden to
read specific layer configuration.


### Layer specific protocol message

To implement a new layer, the first step is to define the layer specific
configuration. Suppose the new layer is of type kFoo, its layer specific
google protocol message `FooProto` should be defined,

    # in user.proto
    package singa
    import "job.proto"
    message FooProto {
      // specific fields to the FooLayer
      optional int32 a = 1;
    }

    extend LayerProto {
      optional FooProto foo_conf = 101;  // unique field id, reserved for users
    }

In addition, users need to extend the original `LayerProto`
defined in job.proto of SINGA to include the `foo_conf`. If there are multiple
new layers, then each layer that has specific configurations would have a
`<type>_conf` field and take one extension number (starting from 101).

    # job.proto of SINGA
    LayerProto {
      ...
      extensions 101 to max;
    }

With user.proto defined, users can use [protoc]() to generate the `user.pb.cc`
and `user.pb.h` files. The layer configuration is like,

    layer {
      name: "foo"
      type: kFoo # must not be the same with any built-in layer
      [singa.foo_conf] { # note there is a pair of [] for extension fields
        a: 10
      }
    }

In users' code, the extension fields can be accessed via,

    auto conf = layer_proto_.GetExtension(foo_conf);
    int a = conf.a();

### Layer subclasses

The new layer subclass can be defined following the built-in layer subclasses.

    class FooLayer : public Layer {
     public:
      void Setup(const LayerProto& proto, int npartitions = 1) override;
      void ComputeFeature(Phase phase, Metric* perf) override;
      void ComputeGradient(Phase phase) override;

     private:
      //  members
    };

Users must override the two virtual functions to be called by the
`TrainOneBatch` for either BP or CD algorithm. Typically, the `Setup` function
will also be overridden to initialize some members. The user configured fields
can be accessed through `layer_proto_` as shown in the above paragraphs.

## Built-in Layer subclasses
SINGA has provided many
built-in layers, which can be used directly to create neural nets.
|*Category*|*Description*|
|:-----|:-----|
|Data layer|Load records from ﬁle, database or HDFS|
|Parser layer|Parse features from records, e.g., pixels and label|
|Neuron layer|Feature transformation, e.g., convolution, pooling|
|Loss layer|Compute objective loss, e.g., cross-entropy loss|
|Other layer|Utility layers for neural net partitioning|

The layers in SINGA are categorized as follows according to
their functionalities,

  * Data layers for loading records (e.g., images) from disk, HDFS or network into memory.
  * Parser layers for parsing features, labels, etc. from records.
  * Neuron layers for feature transformation, e.g., convolution, pooling, dropout, etc.
  * Loss layers for measuring the training objective loss, e.g., cross entropy-loss or Euclidean loss.
  * Output layers for outputting the prediction results (e.g., probabilities of each category) onto disk or network.
  * Connection layers for connecting layers when the neural net is partitioned.

###Data Layers

#### ShardData Layer

ShardData layer is used to read data from disk etc.

	layer
	{
		name:"data"
		type:kShardData
		data_param
		{
			path:"Shard_File_Path"
			batchsize:int
		}
		exclude:kTrain|kValidation|kTest|kPositive|kNegative
	}

#### LMDBData Layer
This is a data input layer, the data will be provided by the LMDB
batchsize means the quantity of the input disposable

    layer
    {
    	name:"data"
    	type:kLMDBDate
    	data_param
    	{
    		path:"LMDB_FILE_PATH"
    		batchsize:int
    	}
    	exclude:kTrain|kValidation|kTest|kPositive|kNegative
    }



###Parser Layers

#### Label Layer
Label layer is used to extract the label information from training data. The label information will be used in the loss layer to calculate the gradient.

    layer
    {
    	name:"label"
    	type:kLabel
    	srclayers:"data"
    }

#### MnistImage Layer
MnistImage is a pre-processing layer for MNIST dataset.


    layer
    {
    	name:"mnist"
    	type:kMnistImage
    	srclayers:"data"
    	mnist_param
    	{
    		sigma:int
    		alpha:int
    		gamma:int
    		kernel:int
    		elastic_freq:int
    		beta:int
    		resize:int
    		norm_a:int
    	}
    }


#### RGBImage Layer
RGBImage layer is a pre-processing layer for RGB format images.

    layer
    {
    	name:"rgb"
    	type:kRGBImage
    	srclayers:"data"
    	rgbimage_param
    	{
    		meanfile:"Image_Mean_File_Path"
    	}
    }

### Prefetch Layer
Prefetch Layer is used to pre-fetch data from disk. It ensures that the I/O task and computation/communication task can work simultaneously.

    layer
    {
    	name:"prefetch"
    	type:kPrefetch
    	sublayers
    	{
    		name:"data"
    		type:"kShardData"
    		data_param
    		{
    			path:"Shard_File_Path"
    			batchsize:int
    		}
    	}
    	sublayers
    	{
    		name:"rgb"
    		type:kRGBImage
    		srclayers:"data"
    		rgbimage_param
    		{
    			meanfile:"Image_Mean_File_Path"
    		}
    	}
    	sublayers
    	{
    		name:"label"
    		type:kLabel
    		srclayers:"data"
    	}
    	exclude:kTrain|kValidation|kTest|kPositive|kNegative
    }



### Neuron Layers

####  Convolution Layer
Convolution layer is a basic layer used in constitutional neural net. It is used to extract local feature following some local patterns from slide windows in the image.

    layer
    {
    	name:"Conv_Number"
    	type:kConvolution
    	srclayers:"Src_Layer_Name"
    	convolution_param
    	{
    		num_filters:int
    		//the count of the applied filters
    		kernel:int
    		//convolution kernel size
    		stride:int
    		//the distance between the successive filters
    		pad:int
    		//pad the images with a given int number of pixels border of zeros
    	}
    	param
    	{
    		name:"weight"
    		init_method:kGaussian|kConstant:kUniform|kPretrained|kGaussianSqrtFanIn|kUniformSqrtFanIn|kUniformSqrtFanInOut
    		/*use specific param of each init methods*/
    		learning_rate_multiplier:float
    	}
    	param
    	{
    		name:"bias"
    		init_method:kConstant|kGaussian|kUniform|kPretrained|kGaussianSqrtFanIn|kUniformSqrtFanIn|kUniformSqrtFanInOut
    		/**use specific param of each init methods**/
    		learning_rate_multiplier:float
    	}
    	//kGaussian: sample gaussian with std and mean
    	//kUniform: uniform sampling between low and high
    	//kPretrained: from Toronto Convnet, let a=1/sqrt(fan_in),w*=a after generating from Gaussian distribution
    	//kGaussianSqrtFanIn: from Toronto Convnet, rectified linear activation,
    		//let a=sqrt(3)/sqrt(fan_in),range is [-a,+a].
    		//no need to set value=sqrt(3),the program will multiply it
    	//kUniformSqrtFanIn: from Theano MLP tutorial, let a=1/sqrt(fan_in+fan_out).
    		//for tanh activation, range is [-6a,+6a], for sigmoid activation.
    		// range is [-24a,+24a],put the scale factor to value field
    	//For Constant Init, use value:float
    	//For Gaussian Init, use mean:float, std:float
    	//For Uniform Init, use low:float, high:float
    }


####  InnerProduct Layer
InnerProduct Layer is a fully connected layer which is the basic element in feed forward neural network.
It will use the lower layer as a input vector V and output a vector H by doing the following matrix-vector multiplication:
H = W*V + B // W and B are its weight and bias parameter

    layer
    {
    	name:"IP_Number"
    	type:kInnerProduct
    	srclayers:"Src_Layer_Name"
    	inner_product_param
    	{
    		num_output:int
    		//The number of the filters
    	}
    	param
    	{
    		name:"weight"
    		init_method:kGaussian|kConstant:kUniform|kPretrained|kGaussianSqrtFanIn|kUniformSqrtFanIn|kUniformSqrtFanInOut
    		std:float
    		learning_rate_multiplier:float
    		weight_decay_multiplier:int
    		/*optional:low:float,high:float*/
    	}
    	param
    	{
    		name:"bias"
    		init_method:kConstant|kGaussian|kUniform|kPretrained|kGaussianSqrtFanIn|kUniformSqrtFanIn|kUniformSqrtFanInOut
    		learning_rate_mulitiplier:float
    		weight_decay_multiplier:int
    		value:int
    		/*optional:low:float,high:float*/
    	}
    }


####  Pooling Layer
Max Pooling uses a specific scaning window to find the max value
Average Pooling scans all the values in the window to calculate the average value

    layer
    {
    	name:"Pool_Number"
    	type:kPooling
    	srclayers:Src_Layer_Name"
    	pooling_param
    	{
    		pool:AVE|MAX
    		//Choose whether use the Average Pooling or Max Pooling
    		kernel:int
			//size of the kernel filter
    		stride:int
			//the step length of the filter
    	}
    }

#### ReLU Layer

  The rectifier function is an activation function f(x) = Max(0, x) which can be used by neurons just like any other activation function, a node using the rectifier activation function is called a ReLu node. The main reason that it is used is because of how efficiently it can be computed compared to more conventional activation functions like the sigmoid and hyperbolic tangent, without making a significant difference to generalisation accuracy. The rectifier activation function is used instead of a linear activation function to add non linearity to the network, otherwise the network would only ever be able to compute a linear function.

    layer
    {
    	name:"Relu_Number"
    	type:kReLU
    	srclayers:"Src_Layer_Name"
    }

#### Tanh Layer
Tanh uses the tanh as activation function. It transforms the input into range [-1, 1].

    layer
    {
    	name:"Tanh_Number"
    	type:kTanh
    	srclayer:"Src_Layer_Name"
    }

#### Dropout Layer
Dropout Layer is a layer that randomly dropout some inputs. This scheme helps deep learning model away from over-fitting.

#### LRN Layer

Local Response Normalization normalizes over the local input areas. It provides two modes: WITHIN_CHANNEL and ACROSS_CHANNELS.
The local response normalization layer performs a kind of “lateral inhibition” by normalizing over local input regions. In ACROSS_CHANNELS mode, the local regions extend across nearby channels, but have no spatial extent (i.e., they have shape local_size x 1 x 1). In WITHIN_CHANNEL mode, the local regions extend spatially, but are in separate channels (i.e., they have sha
pe 1 x local_size x local_size). Each input value is divided by ![](http://i.imgur.com/GgTjjtR.png), where n is the size of each local region, and the sum is taken over the region centered at that value (zero padding is added where necessary).


    layer
    {
    	name:"Norm_Number"
    	type:kLRN
    	lrn_param
    	{
    		norm_region:WITHIN_CHANNEL|ACROSS_CHANNELS
    		local_size:int
			//for WITHIN_CHANNEL, it means the side length of the space region which will be summed up
			//for ACROSS_CHANNELS, it means the quantity of the adjoining channels which will be summed up
    		alpha:5e-05
    		beta:float
    	}
    	srclayers:"Src_Layer_Name"
    }


### Loss Layers

#### SoftmaxLoss Layer
Softmax Loss Layer is the implementation of multi-class softmax loss function. It is generally used as the final layer to generate labels for classification tasks.

    layer
    {
    	name:"loss"
    	type:kSoftmaxLoss
    	softmaxloss_param
    	{
    		topk:int
    	}
    	srclayers:"Src_Layer_Name"
    	srclayers:"Src_Layer_Name"
    }


### Other Layers

#### Concate Layer

Concat Layer is used to concatenate the last dimension (namely, num_feature) of the output of two nodes. It is usually used along with fully connected layer.

#### Slice Layer

The Slice layer is a utility layer that slices an input layer to multiple output layers along a given dimension (currently num or channel only) with given slice indices.

#### Split Layer

The Split Layer can seperate the input blob into several output blobs. It is used to the situation which one input blob should be input to several other output blobs.

#### BridgeSrc & BridgeDst Layer

BridgeSrc & BridgeDst Layer are utility layers implementing logics of model partition. It can be used as a lock for synchronization, a transformation storage of different type of model partition and etc.





