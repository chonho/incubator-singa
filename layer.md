# Layers Instruction  

----

## Brief Introduction  

Layer is Singa's core abstraction. It performs a variety of feature transformations for extracting high-level features.

Layer abstraction is as follow:  

    Layer: 
      Vector<Layer> srclayer 
      Blob feature 
      Func ComputeFeature() 
      Func ComputeGreadient() 
 
    Param: 
      Blob data, gradient

The layers used in the Singa model can be categorized as follow:  

|*Category*|*Description*|
|:-----|:-----|
|Data layer|Load records from ﬁle, database or HDFS|
|Parser layer|Parse features from records, e.g., pixels and label|
|Neuron layer|Feature transformation, e.g., convolution, pooling|
|Loss layer|Compute objective loss, e.g., cross-entropy loss|
|Other layer|Utility layers for neural net partitioning|




##Data Layers  

### ShardData Layer
ShardData layer is used to read data from disk etc.  

	layer   
	{
		name:"data"
		type:"kShardData"
		data_param
		{
			path:"Shard_File_Path"
			batchsize:int
		}
		exclude:kTrain|kValidation|kTest|kPositive|kNegative
	}

### LMDBData Layer  
This is a data input layer, the data will be provided by the LMDB
batchsize means the quantity of the input disposable  

    layer
    {
    	name:"data"
    	type:"kLMDBDate"
    	data_param
    	{
    		path:"LMDB_FILE_PATH"
    		batchsize:int
    	}
    	exclude:kTrain|kValidation|kTest|kPositive|kNegative
    }











##Parser Layers  

### Label Layer  
Label layer is used to extract the label information from training data. The label information will be used in the loss layer to calculate the gradient.  

    layer
    {
    	name:"label"
    	type:"kLabel"
    	srclayers:"data"
    }

### RGBImage Layer  
RGBImage layer is a pre-processing layer for RGB format images.  

    layer
    {
    	name:"rgb"
    	type:"kRGBImage"
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
    	type:"kPrefetch"
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
    		type:"kRGBImage"
    		srclayers:"data"
    		rgbimage_param
    		{
    			meanfile:"Image_Mean_File_Path"
    		}
    	}
    	sublayers
    	{
    		name:"label"
    		type:"kLabel"
    		srclayers:"data"
    	}
    	exclude:kTrain|kValidation|kTest|kPositive|kNegative
    }















##Neuron Layers

### Convolution Layer  
Convolution layer is a basic layer used in constitutional neural net. It is used to extract local feature following some local patterns from slide windows in the image.  

    layer
    {
    	name:"Conv_Number"
    	type:"kConvolution"
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
 

### InnerProduct Layer  
InnerProduct Layer is a fully connected layer which is the basic element in feed forward neural network.
It will use the lower layer as a input vector V and output a vector H by doing the following matrix-vector multiplication:
H = W*V + B // W and B are its weight and bias parameter  

    layer
    {
    	name:"IP_Number"
    	type:"kInnerProduct"
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


### Pooling Layer  
Max Pooling uses a specific scaning window to find the max value  
Average Pooling scans all the values in the window to calculate the average value  

    layer
    {
    	name:"Pool_Number"
    	type:"kPooling"
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

### ReLU Layer  
  
  The rectifier function is an activation function f(x) = Max(0, x) which can be used by neurons just like any other activation function, a node using the rectifier activation function is called a ReLu node. The main reason that it is used is because of how efficiently it can be computed compared to more conventional activation functions like the sigmoid and hyperbolic tangent, without making a significant difference to generalisation accuracy. The rectifier activation function is used instead of a linear activation function to add non linearity to the network, otherwise the network would only ever be able to compute a linear function.

    layer
    {
    	name:"Relu_Number"
    	type:"kReLU"
    	srclayers:"Src_Layer_Name"
    }

### Tanh Layer  
Tanh uses the tanh as activation function. It transforms the input into range [-1, 1].  

    layer
    {
    	name:"Tanh_Number"
    	type:"kTanh"
    	srclayer:"Src_Layer_Name"
    }

















##Loss Layers

### SoftmaxLoss Layer  
Softmax Loss Layer is the implementation of multi-class softmax loss function. It is generally used as the final layer to generate labels for classification tasks.  

    layer
    {
    	name:"loss"
    	type:"kSoftmaxLoss"
    	softmaxloss_param
    	{
    		topk:int
    	}
    	srclayers:"Src_Layer_Name"
    	srclayers:"Src_Layer_Name"
    }






















##Other Layers

### Dropout Layer
Dropout Layer is a layer that randomly dropout some inputs. This scheme helps deep learning model away from over-fitting.  

### LRN Layer  

Local Response Normalization normalizes over the local input areas. It provides two modes: WITHIN_CHANNEL and ACROSS_CHANNELS. 
The local response normalization layer performs a kind of “lateral inhibition” by normalizing over local input regions. In ACROSS_CHANNELS mode, the local regions extend across nearby channels, but have no spatial extent (i.e., they have shape local_size x 1 x 1). In WITHIN_CHANNEL mode, the local regions extend spatially, but are in separate channels (i.e., they have sha
pe 1 x local_size x local_size). Each input value is divided by ![](http://i.imgur.com/GgTjjtR.png), where n is the size of each local region, and the sum is taken over the region centered at that value (zero padding is added where necessary).


    layer
    {
    	name:"Norm_Number"
    	type:"kLRN"
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

### MnistImage Layer  
MnistImage is a pre-processing layer for MNIST dataset.  


    layer
    {
    	name:"mnist"
    	type:"kMnistImage"
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

### Concate Layer  

Concat Layer is used to concatenate the last dimension (namely, num_feature) of the output of two nodes. It is usually used along with fully connected layer.

### Slice Layer    

The Slice layer is a utility layer that slices an input layer to multiple output layers along a given dimension (currently num or channel only) with given slice indices.

### Split Layer  

The Split Layer can seperate the input blob into several output blobs. It is used to the situation which one input blob should be input to several other output blobs.

### BridgeSrc & BridgeDst Layer  

BridgeSrc & BridgeDst Layer are utility layers implementing logics of model partition. It can be used as a lock for synchronization, a transformation storage of different type of model partition and etc.













