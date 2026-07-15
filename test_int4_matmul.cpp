#include <iostream>
#include <vector>
#include "quantized_matmul_int4.hpp"

int main() {
    std::vector<double> W = {
        0.10, -0.20, 0.30,
        0.40,  0.50, -0.60,
       -0.70,  0.80, 0.90,
        1.00, -1.10, 1.20
    };
    std::vector<double> X = {
        1.0, 2.0, 3.0, 4.0,
       -1.0, 0.5, 2.0, -0.5
    };

    QuantizedMatrixInt4 q;
    q.quantize_from_row_major(W, 4, 3);
    std::vector<double> Y;
    q.matmul(X, 2, 4, Y, 3, 2);

    std::cout << "INT4 packed matmul output:\n";
    for (size_t i = 0; i < Y.size(); ++i) std::cout << Y[i] << ((i + 1) % 3 == 0 ? '\n' : ' ');
    std::cout << "Quantized bytes: " << q.bytes() << "\n";
}
