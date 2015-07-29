libs :=singa glog protobuf
filename = rnnlm-0.4b.tgz
# note: filelink for rnnlm-0.4b may change
filelink = https://f25ea9ccb7d3346ce6891573d543960492b92c30.googledrive.com/host/0ByxdPXuxLPS5RFM5dVNvWVhTd0U
dirname = $(patsubst %.tgz,%, $(filename))
numclass = 3
dirshards = rnnlm_word_shard_train rnnlm_word_shard_test rnnlm_word_shard_valid rnnlm_class_shard rnnlm_vocab_shard


.PHONY: all download create

download: rnnlm 

rnnlm:
	wget $(filelink)/$(filename)
	tar zxf $(filename)
	rm $(filename)

create:
	$(CXX) create_shard.cc -std=c++11 -lsinga -lprotobuf -lglog -I../../include \
		-L../../.libs/ -Wl,-unresolved-symbols=ignore-in-shared-libs -Wl,-rpath=../../.libs/ \
		-o create_shard.bin
	for d in $(dirshards); do mkdir -p $${d}; done 
	./create_shard.bin $(dirname)/train $(numclass)
	./create_shard.bin $(dirname)/test -1 
	./create_shard.bin $(dirname)/valid -1 
