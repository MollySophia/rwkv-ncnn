#pragma once
#include <unordered_map>
#include <string>
#include <string_view>
#include <vector>

// TODO
namespace rwkv {

class TRIE {
public:
    TRIE();

private:
    bool is_end;
    TRIE *next;
};

class TRIE_Tokenizer {
public:

    TRIE_Tokenizer(const char *file_path);

    std::vector<int> Encode(std::string_view str);

    std::string Decode(int id);

    std::string Decode(const std::vector<int> &ids);

private:
    std::unordered_map<int, std::string> idx2token;
    std::unordered_map<std::string, int> token2idx;
};

}