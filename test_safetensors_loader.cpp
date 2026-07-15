#include <iostream>
#include "safetensors_loader.hpp"

int main() {
    SafeTensors st("tiny_test.safetensors");
    std::cout << "Loaded tensors:\n";
    for (const auto& name : st.names()) {
        const auto& info = st.info(name);
        std::cout << "- " << name << " dtype=" << info.dtype << " shape=[";
        for (size_t i = 0; i < info.shape.size(); ++i) std::cout << (i ? "," : "") << info.shape[i];
        std::cout << "]\n";
    }
    auto v = st.tensor_as_double("x");
    std::cout << "x: ";
    for (double d : v) std::cout << d << ' ';
    std::cout << '\n';
}
