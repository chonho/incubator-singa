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
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <map>
#include <set>

#include "utils/data_shard.h"
#include "utils/common.h"
#include "proto/common.pb.h"

using singa::DataShard;
using singa::WriteProtoToBinaryFile;
using std::string;

// key: word index, value: pair of <start, end> of word index
typedef std::map< int, std::pair<int,int> > ClassMap;
// key: word, value: tuple of <word index, word freq, class index>
typedef std::unordered_map< string, std::tuple<int,int,int> > WordMap;
// pair of <word freq, word>, used for sorting
typedef std::set< std::pair<int,string> > IntStrSet;
typedef std::unordered_map< string, int> StrIntMap;

void displayMap(WordMap &wmap, ClassMap &cmap) {
  for (WordMap::iterator p = wmap.begin(); p != wmap.end(); ++p) {
    std::cout << p->first << "\t"
              << std::get<0>(p->second) << "\t"
              << std::get<1>(p->second) << "\t"
              << std::get<2>(p->second) << "\n";
  }
  std::cout << cmap[1].first << " " << cmap[1].second << ", "
            << cmap[2].first << " " << cmap[2].second << ", "
            << cmap[3].first << " " << cmap[3].second << "\n";

  std::cout << "# of words " << wmap.size() << "\n";
}

void create_shard(const char* input, const char* output) {

  // Open file
  std::ifstream in(input);
  CHECK(in) << "Unable to open file " << input;

  // Count word frequency, sort the map, compute class index
  uint8_t nclass = 3;
  uint32_t nwords = 0;
  WordMap myWordMap;
  ClassMap myClassMap;
  IntStrSet wordset;
  StrIntMap wordmap;
  string word;
  char delim[] = ".,':;";
  
  while (in >> word) {
    std::size_t pos = word.find_first_of(delim);
    ++wordmap[(pos==string::npos) ? word : word.substr(0,pos)]; 
    ++nwords;
  }

  for (StrIntMap::iterator p = wordmap.begin(); p != wordmap.end(); ++p)
    wordset.emplace( std::make_pair(p->second, p->first) );

  int wordIdx=1, start=wordIdx, count=0, classIdx=1;
  for (IntStrSet::reverse_iterator p = wordset.rbegin(); p != wordset.rend(); ++p) {
    count += p->first;
    if(classIdx < nclass && count > classIdx*nwords/nclass) {
      myClassMap[classIdx] = std::make_pair(start,wordIdx-1);
      start = wordIdx;
      classIdx++;
    }
    myWordMap[p->second] = std::make_tuple(wordIdx++,p->first,classIdx);
  }
  myClassMap[classIdx] = std::make_pair(start,wordIdx-1); // <-- last class index

  //displayMap(myWordMap, myClassMap);
  wordset.clear();
  wordmap.clear();

  // Create datashard
  DataShard shard(output, DataShard::kCreate);
  singa::Record record;
  singa::WordClassRecord* wordclass = record.mutable_recordclass();
  singa::SingleWordRecord* singleword = record.mutable_recordword();
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  int item_id = 0;
 
  for (ClassMap::iterator p = myClassMap.begin(); p != myClassMap.end(); ++p) {
    wordclass->set_start( p->second.first );    
    wordclass->set_end( p->second.second );    
    snprintf(key, kMaxKeyLength, "%08d", item_id++);
    shard.Insert(string(key), record);
  }
  
  for (WordMap::iterator p = myWordMap.begin(); p != myWordMap.end(); ++p) {
    singleword->set_name( p->first );
    singleword->set_vocab_index( std::get<0>(p->second) );
    //singleword->set_freq( std::get<1>(p->second) ); // ignore. word frequency
    singleword->set_class_index( std::get<2>(p->second) );
    snprintf(key, kMaxKeyLength, "%08d", item_id++);
    shard.Insert(string(key), record);
  }
  myClassMap.clear();
  myWordMap.clear();
  shard.Flush();
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "This program create a DataShard for a RNNLM dataset\n"
        "The RNNLM dataset could be downloaded at\n"
        "    http://www.rnnlm.org/\n"
        "You should gunzip them after downloading.\n"
        "Usage:\n"
        "    create_shard.bin  input_file output_dirname";
  } else {
    google::InitGoogleLogging(argv[0]);
    create_shard(argv[1], argv[2]);
  }
  return 0;
}
