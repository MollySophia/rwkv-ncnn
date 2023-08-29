#include <iostream>
#include <fstream>
#include <string_view>
#include <msgpack.hpp>
#include <chrono>

#include "rwkv_tokenizer.h"

#define DEBUG_TIME 1

namespace rwkv {
TRIE_Tokenizer::TRIE_Tokenizer(const char *file_path) {
  std::ifstream infile;
  infile.open(file_path, std::ios::binary | std::ios::in);
  infile.seekg(0, std::ios::end);
  int64_t length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  char *data = new char[length];
  infile.read(data, length);
  infile.close();

  auto unpacker = msgpack::unpack(data, length);
  auto obj = unpacker.get();
  idx2token = obj.as<std::unordered_map<int, std::string>>();
  for (auto &pair : idx2token) {
    token2idx[pair.second] = pair.first;
  }
}

std::vector<int> TRIE_Tokenizer::Encode(std::string_view str) {
  std::vector<int> ids;
  int str_idx = 0;
  int word_len = 1;
  int id = 0;
  while (str_idx < str.size()) {
    if (str_idx + word_len > str.size()) {
      ids.push_back(id);
      break;
    }
    auto substr = str.substr(str_idx, word_len);
    auto it = token2idx.find(std::string(substr));
    if (it == token2idx.end()) {
      ids.push_back(id);
      str_idx += (word_len - 1);
      word_len = 1;
    } else {
      id = it->second;
      word_len++;
    }
  }

  return ids;
}

std::string TRIE_Tokenizer::Decode(int id) {
  auto it = idx2token.find(id);
  if (it == idx2token.end()) {
    return "<unk>";
  } else {
    return it->second;
  }
}

std::string TRIE_Tokenizer::Decode(const std::vector<int> &ids) {
  std::string str;
  for (auto id : ids) {
    str += Decode(id);
  }
  return str;
}

}