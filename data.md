## Data Preparation

To submit a training job, users need to convert raw data (e.g., images, text documents) into records that can be recognized by SINGA. SINGA uses a DataLayer
to load these records into memory and uses ParserLayer to parse features (e.g., image pixels and labels) from these records. The records could be organized and
stored using many different ways, e.g., a file, a light database, or HDFS, as long as there is a corresponding DataLayer that can load the records.

### DataShard

To create shard for your own data, users may need to implement or modify the following files

- user.proto
- create_shard.cc
- Makefile

**1. Define record**

Record class is inherited from Message class whose format follows Google protocol buffers. Please refer to the [Tutorial][1].

Your record will be defined in a file, user.proto, as an extension of Record written in SINGA_ROOT/src/proto/common.proto.

(a) Declare the user-defined record as an extension of Record. Please refer to the [Tutorial][2] for Extension.

    package singa;  

    import "common.proto";  // required to import common.proto

    extend Record {
        optional UserRecord1 user_record1 = 100;  // unique extension field id, reserved for users (e.g., 100-199)
        optional UserRecord2 user_record2 = 101;  // unique extension field id, reserved for users (e.g., 100-199)
    }

(b) Define the user own record

    message UserRecord1 {
        repeated int userVAR1 = 1; // unique field id
        optional string userVAR2 = 2; // unique field id
        ...
    }

    message UserRecord2 {
        ...
    }

*Note*

There is an alternative way to define the proto extension.

    message UserRecord {
        extend Record {
            optional UserRecord user_record = 100;  // unique extension field id, reserved for users (e.g., 100-199)
        }
        repeated int userVAR1 = 1; // unique field id
        optional string userVAR2 = 2; // unique field id
        ...
    }

In this way, you should be careful of the scope of fields and how to access the fields, which are different from the above.

**2. Compile proto/Singa**

Once you defined protos, you need to compile them as follow.

    protoc -I=IMPORT_DIR --cpp_out=DST_DIR [proto1] [proto2] ...

    - IMPORT_DIR: path to contain proto files
    - DST_DIR: path to generate C++ codes (.pb.h and .pb.cc)

Assume that users will implement their own [Layers][3] as well. The following commands help you to compile Singa sources as well as proto.

    cd SINGA_ROOT
    ./configure
    make

**3. Create shard**

(a) Create a folder for dataset and shard, e.g., we call it "USER_DATA". (This folder would be the same as WORKSPACE.)

(b) Source files, e.g., create_shard.cc, will be in SINGA_ROOT/USER_DATA/

DataShard object

    singa::DataShard myShard( outputpath, mode );

    - `string outputpath`, where user wants to create shard.
    - `int mode := kRead | kCreate | kAppend`, is defined in SINGA_ROOT/include/utils/data_shard.h

Record object

    singa::Record record;

    singa::UserRecord *myRecord = record.MutableExtension(singa::user_record);

    - `MutableExtension()` is templated method, generated after compile at Step 2 and processed at runtime.

(c) Set/Add values into your record

    myRecord->add_userVAR1( int_val );     // for repeated field
    myRecord->set_userVAR2( string_val );

(d) Write the record to shard

    myShard.Insert( key, myRecord );
- `String key`, will be a unique id for a message

**Note: Preparing Makefile**

There are some notes when you write your own Makefile

- Specify `outputpath` of output directory for shard.

- Make sure you copy `user.proto` into SINGA_ROOT/src/proto

Examples of creating shard can be found at [RNNLM][4]

### LMDB



### HDFS


  [1]: https://developers.google.com/protocol-buffers/docs/cpptutorial
  [2]: https://developers.google.com/protocol-buffers/docs/reference/cpp-generated?hl=en#extension
  [3]: http://singa.incubator.apache.org/docs/layer.html
  [4]: http://singa.incubator.apache.org/docs/examples.html

