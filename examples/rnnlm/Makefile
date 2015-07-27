libs :=singa glog protobuf
filename = rnnlm-0.4b.tgz
# note: filelink for rnnlm-0.4b may change
filelink = https://f25ea9ccb7d3346ce6891573d543960492b92c30.googledrive.com/host/0ByxdPXuxLPS5RFM5dVNvWVhTd0U
dirname = $(patsubst %.tgz,%, $(filename))
numclass = 3
dirshards = rnnlm_train_shard rnnlm_test_shard rnnlm_valid_shard


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
	for d in $(dirshards); do \
		mkdir -p $${d}; cd $${d}; \
		mkdir -p class_shard; \
		mkdir -p vocab_shard; \
		mkdir -p word_shard; cd ..; \
	done	
	./create_shard.bin $(dirname)/train rnnlm_train_shard $(numclass)
	./create_shard.bin $(dirname)/test rnnlm_test_shard -1
	./create_shard.bin $(dirname)/valid rnnlm_valid_shard -1
