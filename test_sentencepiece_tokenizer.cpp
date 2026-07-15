#include <iostream>
#include "sentencepiece_tokenizer.hpp"

int main() {
    SentencePieceTokenizer tok("toy_tokenizer.model");
    auto ids = tok.encode("hello world", true);
    std::cout << "Vocab size: " << tok.vocab_size() << "\nIDs: [ ";
    for (int id : ids) std::cout << id << ' ';
    std::cout << "]\nDecoded: " << tok.decode(ids) << "\n";
}
