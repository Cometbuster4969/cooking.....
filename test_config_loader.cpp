#include <iostream>
#include "llm_config.hpp"

int main() {
    LLMConfig cfg = LLMConfig::from_json_file("toy_config.json");
    std::cout << "vocab_size=" << cfg.vocab_size << "\n";
    std::cout << "hidden_size=" << cfg.hidden_size << "\n";
    std::cout << "layers=" << cfg.num_hidden_layers << "\n";
    std::cout << "heads=" << cfg.num_attention_heads << "\n";
    std::cout << "kv_heads=" << cfg.num_key_value_heads << "\n";
    std::cout << "intermediate=" << cfg.intermediate_size << "\n";
}
