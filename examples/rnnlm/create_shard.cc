//
// This code creates DataShard for RNNLM dataset.
// It is adapted from the convert_mnist_data from Caffe
// The RNNLM dataset could be downloaded at
//    http://www.rnnlm.org/
//
// Usage:
//    create_shard.bin input_filename output_foldername

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <string>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

#include "utils/data_shard.h"
#include "utils/common.h"
#include "proto/common.pb.h"


using singa::DataShard;

//using StrIntMap = std::map<std::string, int, std::greater<std::string> >;
using StrIntMap = std::map<std::string, int>;
using StrIntPair = std::pair<std::string, int>;

void doClusterForTrainMode(const char *input, const char *output, int nclass, StrIntMap& wordIdxMap, StrIntMap& wordClassIdxMap) {
    // init
    wordIdxMap.clear();
    wordClassIdxMap.clear();

    // load input file
    std::ifstream in(input);
    CHECK(in) << "Unable to open file " << input;

    // calc word's frequency by map
    std::string word;
    StrIntMap wordFreqMap;
    while (in >> word) {
        // TODO(kaiping): improve tokenize logic for complex input format (such as symbols)
        ++wordFreqMap[word];
    }
    int nword = static_cast<int>(wordFreqMap.size());

    // count sum of freq
    int sumFreq = 0;
    for (auto& it : wordFreqMap) {
        sumFreq += it.second;
    }
    
    // index words after sorting
    std::vector<StrIntPair> wordFreqSortedVec(wordFreqMap.begin(), wordFreqMap.end());
    auto pairSecondGreater = [](StrIntPair a, StrIntPair b) { return a.second > b.second; };
    std::sort(wordFreqSortedVec.begin(), wordFreqSortedVec.end(), pairSecondGreater);

    // split words into classes by freq
    std::vector<std::pair<int, int> > classInfo;
    int classIdxCnt = 0;
    int tmpWordFreqSum = 0;
    int nextStartPos = 0;
    int wordIdxCnt = 0;
    for (auto& it : wordFreqSortedVec) {
        // index words
        wordIdxMap[it.first] = static_cast<int>(wordIdxMap.size());
        // generate classes
        tmpWordFreqSum += it.second;
        wordClassIdxMap[it.first] = classIdxCnt;
        // split a new class (ensure no empyt class)
        if ((tmpWordFreqSum >= (classIdxCnt + 1) * sumFreq / nclass) || (nword - wordIdxCnt) <= nclass - classIdxCnt) {
            classInfo.emplace_back(std::make_pair(nextStartPos, wordIdxCnt));
            nextStartPos = wordIdxCnt + 1;
            ++classIdxCnt;
        }
        ++wordIdxCnt;
    }
    
    // generate class shard
    const int kMaxKeyLength = 10;
    char key[kMaxKeyLength];
    DataShard classShard(std::string(output)+"/class_shard", DataShard::kCreate);
    singa::Record record;
    record.set_type(singa::Record::kWordClass);
    singa::WordClassRecord *classRecord = record.mutable_class_record();
    for (int i = 0; i != classInfo.size(); ++i) {
        classRecord->set_start(classInfo[i].first);
        classRecord->set_end(classInfo[i].second);
        classRecord->set_class_index(i);
        snprintf(key, kMaxKeyLength, "%08d", i);
        classShard.Insert(std::string(key), record);
    }
    classShard.Flush();
    record.clear_class_record();

    // generate vocabulary shard
    DataShard vocabShard(std::string(output)+"/vocab_shard", DataShard::kCreate);
    record.set_type(singa::Record::kSingleWord);
    singa::SingleWordRecord *wordRecord = record.mutable_word_record();
    for (auto& it : wordFreqSortedVec) {
	std::cout << it.first << "\t" << wordIdxMap[it.first] << "\t" << wordClassIdxMap[it.first] << "\n";
        wordRecord->set_word(it.first);
        wordRecord->set_word_index(wordIdxMap[it.first]);
        wordRecord->set_class_index(wordClassIdxMap[it.first]);
        snprintf(key, kMaxKeyLength, "%08d", wordIdxMap[it.first]);
        vocabShard.Insert(std::string(key), record);
    }
    vocabShard.Flush();
    in.close();

}

void loadClusterForNonTrainMode(const char *input, const char *output, int nclass, StrIntMap& wordIdxMap, StrIntMap& wordClassIdxMap) {
    // init
    wordIdxMap.clear();
    wordClassIdxMap.clear();

    // load vocabulary shard data
    DataShard vocabShard(std::string(output)+"/vocab_shard", DataShard::kRead);
    const int kMaxKeyLength = 10;
    char key[kMaxKeyLength];
    singa::Record record;

    // fill value into map
    while (vocabShard.Next(key, &record)) {
        singa::SingleWordRecord *wordRecord = record.mutable_word_record();
        wordIdxMap[wordRecord->word()] = wordRecord->word_index();
        wordClassIdxMap[wordRecord->word()] = wordRecord->class_index();
    }

}

void create_shard(const char *input, const char *output, int nclass) {

    StrIntMap wordIdxMap, wordClassIdxMap;
    
    if (-1 == nclass) {
        loadClusterForNonTrainMode(input, output, nclass, wordIdxMap, wordClassIdxMap);
    } else {
        doClusterForTrainMode(input, output, nclass, wordIdxMap, wordClassIdxMap);
    }
    // generate word data
    // load input file
    std::ifstream in(input);
    CHECK(in) << "Unable to open file " << input;
    DataShard wordShard(std::string(output)+"/word_shard", DataShard::kCreate);
    singa::Record record;
    record.set_type(singa::Record::kSingleWord);
    singa::SingleWordRecord *wordRecord = record.mutable_word_record();
    int wordStreamCnt = 0;
    const int kMaxKeyLength = 10;
    char key[kMaxKeyLength];
    std::string word;
    while (in >> word) {
        // TODO (kaiping): do not forget here if modify tokenize logic
        // TODO (kaiping): how to handle unknown word, just skip for now
        if (wordIdxMap.end() == wordIdxMap.find(word)) continue;
        wordRecord->set_word(word);
        wordRecord->set_word_index(wordIdxMap[word]);
        wordRecord->set_class_index(wordClassIdxMap[word]);
        snprintf(key, kMaxKeyLength, "%08d", wordIdxMap[word]);
        wordShard.Insert(std::string(key), record);
    }
    wordShard.Flush();
    in.close();

}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "This program create a DataShard for a RNNLM dataset\n"
                "The RNNLM dataset could be downloaded at http://www.rnnlm.org/\n"
                "You should gunzip them after downloading.\n"
                "Usage:\n"
                "    create_shard.bin [in_text_file] [out_shard_dir] [class_size]\n"
                "    class_size=-1 means test or validate mode, elsewise indicates train mode\n";
    } else {
        google::InitGoogleLogging(argv[0]);
        int classSize = atoi(argv[3]);
        CHECK(classSize > 0 || -1 == classSize) << "class size parse failed. [" << argv[3] << "]";
        create_shard(argv[1], argv[2], classSize);
    }
    return 0;
}
