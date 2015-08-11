libs :=singa glog protobuf

singaroot = ../..
dirproto  = $(singaroot)/src/proto
userproto = user.proto

# for dataset
filename = rnnlm-0.4b.tgz
# note: filelink for rnnlm-0.4b may change    
filelink = https://f25ea9ccb7d3346ce6891573d543960492b92c30.googledrive.com/host/0ByxdPXuxLPS5RFM5dVNvWVhTd0U
filedir  = $(patsubst %.tgz,%, $(filename))
numclass = 3 # parameter for rnnlm example

# ouput for shards 
dirshards = rnnlm_word_shard_train rnnlm_word_shard_test rnnlm_word_shard_valid rnnlm_class_shard rnnlm_vocab_shard

.PHONY: all download proto create

download:
	wget $(filelink)/$(filename)
	tar zxf $(filename)
	rm $(filename)

rnnlm: proto create

proto:
	cp $(userproto) $(dirproto)
	protoc -I=$(dirproto) --cpp_out=$(dirproto) $(dirproto)/$(userproto) $(dirproto)/common.proto $(dirproto)/job.proto
	mkdir -p $(singaroot)/tool/pb2/
	touch $(singaroot)/tool/pb2/__init__.py
	protoc -I=$(dirproto) --python_out=$(singaroot)/tool/pb2  $(dirproto)/$(userproto) $(dirproto)/common.proto $(dirproto)/job.proto
	mkdir -p $(singaroot)/include/proto
	cp $(dirproto)/*.pb.h $(singaroot)/include/proto

create:
	$(CXX) create_shard.cc -std=c++11 -lsinga -lprotobuf -lzookeeper_mt -lglog -I../../include \
	       -L../../.libs/ -L/usr/local/lib -Wl,-unresolved-symbols=ignore-in-shared-libs -Wl,-rpath=../../.libs/ \
	       -o create_shard.bin
	for d in $(dirshards); do mkdir -p $${d}; done 
	./create_shard.bin $(filedir)/train $(numclass)
	./create_shard.bin $(filedir)/test -1 
	./create_shard.bin $(filedir)/valid -1 
