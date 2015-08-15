# RNN for Language Modeling
----

## Overall Framework  

The whole RNN language model implemented in SINGA can be seen in Figure (?). In this model, given the current input word, the next 
word is predicted, and the training objective is to maximize the probability of predicting the next word correctly. The performance metric - perplexity per word is employed in this language model. It should be noted that minimizing the perplexity value is equivalent to maximizing the probability of correct prediction. Please refer to this [page] [1] to find more about perplexity.

In this RNN language model, 7 layers are implemented specific for this application, including 1 data layer which will fetch data records from the data shard below, 2 parser layers, 3 neuron layers and 1 loss layer ([more details for layers][2]). After illustrating the data shard and how to create the data shard for this application, we will dicuss the configuration and functionality of each layer layer-wise as follows.



## Data Shard

Example files for RNN can be found in "SINGA_ROOT/examples/rnnlm", which we assume to be WORKSPACE.

(a) Define your own record. Please refer to [Data Preparation][3] for details.

Records for RNN example are defined in "user.proto" as an extension.

    package singa;

    import "common.proto";  // Record message for SINGA is defined
    import "job.proto";     // Layer message for SINGA is defined

    extend Record {
        optional WordClassRecord wordclass = 101;
        optional SingleWordRecord singleword = 102;
    }

    message WordClassRecord {
        optional int32 class_index = 1; // the index of this class
        optional int32 start = 2; // the index of the start word in this class;
        optional int32 end = 3; // the index of the end word in this class
    }

    message SingleWordRecord {
        optional string word = 1;
        optional int32 word_index = 2;   // the index of this word in the vocabulary
        optional int32 class_index = 3;   // the index of the class corresponding to this word
    }

(b) Download raw data

This example downloads rnnlm-0.4b from [www.rnnlm.org][4] by a command 

    make download

The raw data is stored in a folder "rnnlm-0.4b/train" and "rnnlm-0.4b/test".

(c) Create data shard for training and testing

Data shards (e.g., "shard.dat") will be created in "rnnlm_class_shard", "rnnlm_vocab_shard", "rnnlm_word_shard_train" and "rnnlm_word_shard_test" by a command

    make create

## Layers' Configuration and Functionality

Similar to records, layers are also defined in "user.proto" as an extension.

Firstly, we need to add the layer types for this application, then for the layers with special configuration requirements, a new configuration field is added in the configuration file. In this language model, we have 5 layers with special requirements as follows.


    package singa;

    import "common.proto";  // Record message for SINGA is defined
    import "job.proto";     // Layer message for SINGA is defined

    //For implementation of RNNLM application
    extend LayerProto {
        optional RnnlmComputationProto rnnlmcomputation_conf = 201;
        optional RnnlmSigmoidProto rnnlmsigmoid_conf = 202;
        optional RnnlmInnerproductProto rnnlminnerproduct_conf = 203;
        optional RnnlmWordinputProto rnnlmwordinput_conf = 204;
        optional RnnlmDataProto rnnlmdata_conf = 207;
    }

Then for each layer, the detailed configuration and functioanlity information are discusses below.

### RnnlmDataLayer

#### Configuration

In order to improve the computation speed when predicting the next word, all words in the vocabulary are divided into classes. Then during the prediction process, firstly the class of the next word is predicted. Next, inside the predicted class, the next word is predicted.

This RnnlmDataLayer is in charge of reading the information from both a class shard and a word shard. Then a parameter - window_size is configured in this layer. Some important configuration parameters in this layer (set in [job.proto][5]) are shown below.

    message RnnlmDataProto {
        required string class_path = 1;   // path to the data file/folder, absolute or relative to the workspace
        required string word_path = 2;
        required int32 window_size = 3;   // window size.
    }

#### Functionality

In setup phase, this layer first constructs the mapping information between classes and words by reading information from ClassShard. Then in order to maintain the consistency of operations for each training iteration, it maintains a record vector (length of window_size + 1) and then reads 1st record from the WordShard and puts it in the last position of record vector.

    void RnnlmDataLayer::Setup() {
    	Assign values to classinfo_;	//Mapping information between classes and words
        records_.resize(windowsize_ + 1);
        wordshard_->Next(&word_key, &records_[windowsize_]);	//Read 1st word record to the last position   
    }

After setting up this layer, the forward computation phase begins.

    void RnnlmDataLayer::ComputeFeature() {
		records_[0] = records_[windowsize_];	//Copy the last record to 1st position in the record vector
        Assign values to records_;	//Read window_size new word records from WordShard
    }


### RnnlmWordParserLayer

#### Configuration
This layer is in charge of reading word records from RnnlmDataLayer and then passes the records to the neuron layers above. We will configure the name, type and srclayers for this parser layer. More details of this layer's configuration can be seen in [job.conf][6].

#### Functionality
In setup phase, this layer obtains the window size and vocabulary size from its source layer RnnlmDataLayer, and then reshape the data in this layer specific to its functionality.

    void RnnlmWordparserLayer::Setup(){
        windowsize_ = static_cast<RnnlmDataLayer*>(srclayers_[0])->windowsize();	//Obtain window size
        vocabsize_ = static_cast<RnnlmDataLayer*>(srclayers_[0])->vocabsize();	//Obtain vocabulary size
        data_.Reshape(vector<int>{windowsize_}); 	//Reshape data
    }

After setting up this layer, this parser layer fetches the first window_size number of word records from RnnlmDataLayer and store them as data. The details can be seen below.

    void RnnlmWordparserLayer::ParseRecords(){
        for(int i = 0; i < records.size() - 1; i++){	//The first window_size records
            data_[i] = records[i].word_record().word_index();
        }
    }

### RnnlmClassParserLayer

#### Configuration
This layer fetches the class information (the mapping information between classes and words) from RnnlmDataLayer and maintains this information as the data in this layer.

#### Functionality
In setup phase, this layer obtains the window size, vocabulary size and class size from its source layer RnnlmDataLayer, and then reshapes the data in this layer according to the needs.

    void RnnlmClassparserLayer::Setup(const LayerProto& proto, int npartitions){
      windowsize_ = static_cast<RnnlmDataLayer*>(srclayers_[0])->windowsize();	//Obtain window size
      vocabsize_ = static_cast<RnnlmDataLayer*>(srclayers_[0])->vocabsize();	//Obtain vocaubulary size
      classsize_ = static_cast<RnnlmDataLayer*>(srclayers_[0])->classsize();	//Obtain class size
      data_.Reshape(vector<int>{windowsize_, 4});	//Reshape data
    }

Next, this layer parses the last window_size number of word records from RnnlmDataLayer and store them as data. Then for each input word, this layer retrieves its corresponding class, and then stores the starting word index of this class, ending word index of this class, word index and class index respectively.

    void RnnlmClassparserLayer::ParseRecords(){
        float *data_dptr = data_.mutable_cpu_data();
        int *class_info_ptr_tmp = (static_cast<RnnlmDataLayer*>(srclayers_[0])->classinfo())->mutable_cpu_data();
        for(int i = 1; i < records.size(); i++){
            data_dptr[i][0] = start_index;	//Starting word index in this class
            data_dptr[i][1] = end_index;	//Ending word index in this class
            data_dptr[i][2] = word_index;	/Index of input word
            data_dptr[i][3] = class_index;	//Index of input word's class
        }
    }


### RnnlmWordInputLayer

This layer is responsible for using the input word records, obtain corresponding word vectors as its data. Then this layer passes its data to RnnlmInnerProductLayer above for processing.

#### Configuration
In this layer, the length of each word vector needs to be configured. Besides, whether to use bias term during the training process should also be configured (See more in [job.proto][5]).

    message RnnlmWordinputProto {
        required int32 word_length = 1;  // vector length for each input word
        optional bool bias_term = 30 [default = true];  // use bias vector or not
    }

#### Functionality
In setup phase, this layer first reshapes its members: data, grad, and weight matrix. Then RnnlmWordInputLayer obtains the vocabulary size value from its source layer RnnlmWordParserLayer. 

Then in the forward phase of this layer, the member data is obtained. To be specific, by using the window_size number of input word indice, window_size number of word vectors are selected from this layer's weight matrix, each word index corresponding to one row.

    void RnnlmWordinputLayer::ComputeFeature(Phase phase, Metric* perf) {
        for(int t = 0; t < windowsize_; t++){
            data[t] = weight[src[t]];
        }
    }

In the backward phase, after computing this layer's gradient in its destination layer RnnlmInnerProductLayer, here the gradient of the weight matrix in this layer is copied (by row corresponding to word indice) from this layer's grad.

    void RnnlmWordinputLayer::ComputeGradient(Phase phas) {   
        for(int t = 0; t < windowsize_; t++){
            gweight[src[t]] = grad[t];
        }
    }


### RnnlmInnerProductLayer

This is a neuron layer which receives the data from RnnlmWordInputLayer and sends the computation results to RnnlmSigmoidLayer.

#### Configuration
In this layer, the number of neurons needs to be specified. Besides, whether to use a bias term should also be configured.

    message RnnlmInnerproductProto {
        required int32 num_output = 1;	//Number of outputs for the layer
        optional bool bias_term = 30 [default = true];	//Use bias vector or not
    }

#### Functionality

In the forward phase, this layer is in charge of executing the dot multiplication between its weight matrix and the data in its source layer (RnnlmWordInputLayer).

    void RnnlmInnerproductLayer::ComputeFeature() {
        data = dot(src, weight);	//Dot multiplication operation
    }
    
In the backward phase, this layer needs to first compute the gradient of its source layer, i.e., RnnlmWordInputLayer; Then it needs to compute the gradient of its weight matrix by aggregating computation results for each timestamp. The details can be seen as follows.

    void RnnlmInnerproductLayer::ComputeGradient() {
        for (int t = 0; t < windowsize_; t++) {
            gweight += dot(src[t].transpose(), grad[t]); 	//Compute the gradient for the weight matrix            
        }
        gsrc = dot(grad, weight.T());	//Compute the gradient for src layer
    }
   
### RnnlmSigmoidLayer

This is a neuron layer for computation. During the computation in this layer, each component of the member data specific to one timestamp uses its previous timestamp's data component as part of the input. This is how the time-order information is utilized in this language model application.

#### Configuration

In this layer, whether to use a bias term needs to be specified.    

    message RnnlmSigmoidProto {
        optional bool bias_term = 1 [default = true];  // use bias vector or not
    }

#### Functionality

In the forward phase, this RnnlmSigmoidLayer first receives data from its source layer RnnlmInnerProductLayer which is used as one part input for computation. Then for each timestampe this RnnlmSigmoidLayer executes a dot multiplication between its previous timestamp information and its own weight matrix. The results are the other part for computation. Then this layer sums these two parts together and executes an activation operation. The detailed descriptions for this process are illustrated as follows.

    void RnnlmSigmoidLayer::ComputeFeature(Phase phase, Metric* perf) {
        for(int t = 0; t < window_size; t++){
            if(t == 0) data[t] = F<op::sigmoid>(src[t]);
            else data[t] = dot(data[t - 1], weight) + F<op::sigmoid>(src[t]);	//Sum 2 parts together
       }
    }

In the backward phase, this RnnlmSigmoidLayer first updates this layer's member grad utilizing the information from current timestamp's next timestamp. Then respectively, this layer computes the gradient for its weight matrix and its source layer RnnlmInnerProductLayer by iterating different timestamps. The process can be seen below.

    void RnnlmSigmoidLayer::ComputeGradient(Phase phase){
        //1-Update the gradient for the current layer, add a new term from next timestamp
        Update grad[t];
        //2-Compute the gradient for the weight matrix and then compute the gradient for src layer
        for (int t = 0; t < windowsize_; t++) {
                Update gweight;
                Compute gsrc[t];
        }
    }
 


### RnnlmComputationLayer

This layer is a loss layer in which the performance metrics, both the probability of predicting the next word correctly, and perplexity (PPL in short) are computed. To be specific, this layer is composed of the class information part and the word information part. Therefore, the computation can be essentially divided into two parts by slicing this layer's weight matrix.

#### Configuration

In this layer, it is needed to specify whether to use a bias term during training.

    message RnnlmComputationProto {
        optional bool bias_term = 1 [default = true];  // use bias vector or not
    }


#### Functionality

In the forward phase, by using the two sliced weight matrices (one is for class information, the other is for the words in this class), this RnnlmComputationLayer calculates the dot multiplication between the source layer's input and the sliced matrices. The results can be denoted as y1 and y2. Then after a softmax function, for each input word, the probability distribution of classes and the words in this classes are computed. The activated results can be denoted as p1 and p2. Next, using the probability distribution, the PPL value is computed.

    void RnnlmComputationLayer::ComputeFeature() {
        Compute y1 and y2;
        p1 = Softmax(y1);
        p2 = Softmax(y2);
        Compute perplexity value;
    }
    
    
In the backward phase, this layer executes the following three computation operations. First, it computes the member gradient of the current layer by each timestamp. Second, this layer computes the gradient of its own weight matrix by aggregating calculated results from all timestamps. Third, it computes the gradient of its source layer, RnnlmSigmoidLayer timestamp-wise.
        
    void RnnlmComputationLayer::ComputeGradient(){
    	Compute grad[t] for all timestamps;
        Compute gweight by aggregating results computed in different timestamps;
        Compute gsrc[t] for all timestamps; 
    }


## Configure Job

Job configuration is written in "job.conf".

Note: Extended field names should be embraced with square-parenthesis [], e.g., [singa.rnnlmdata_conf].


## Run Training

Start training by the following commands

    cd SINGA_ROOT
    ./bin/singa-run.sh -workspace=examples/rnnlm


[1]: https://en.wikipedia.org/wiki/Perplexity
[2]: http://singa.incubator.apache.org/docs/layer.html
[3]: http://singa.incubator.apache.org/docs/data.html
[4]: www.rnnlm.org
[5]: https://github.com/kaiping/incubator-singa/blob/rnnlm/src/proto/job.proto
[6]: https://github.com/kaiping/incubator-singa/blob/rnnlm/examples/rnnlm/job.conf
