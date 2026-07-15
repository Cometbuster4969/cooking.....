#include <iostream>
#include "bpe_tokenizer.hpp"

int main() {
    BPETokenizer tok("toy_vocab.json", "toy_merges.txt");
    auto ids = tok.encode("unbelievable world", true);
    std::cout << "IDs: [ ";
    for (int id : ids) std::cout << id << ' ';
    std::cout << "]\n";
    std::cout << "Decoded: " << tok.decode(ids) << "\n";
}
