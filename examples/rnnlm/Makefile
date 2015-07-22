libs :=singa glog protobuf
filename = rnnlm-0.4b.tgz
filelink = https://f25ea9ccb7d3346ce6891573d543960492b92c30.googledrive.com/host/0ByxdPXuxLPS5RFM5dVNvWVhTd0U
dirname = $(patsubst %.tgz,%, $(filename))

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
	mkdir -p rnnlm_train_shard
	mkdir -p rnnlm_test_shard
	mkdir -p rnnlm_valid_shard
	./create_shard.bin $(dirname)/train rnnlm_train_shard
	./create_shard.bin $(dirname)/test rnnlm_test_shard
	./create_shard.bin $(dirname)/valid rnnlm_valid_shard

