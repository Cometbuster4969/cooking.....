#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "thread_pool.hpp"

// Symmetric per-output-column packed INT4 weight-only quantization for Y = X * W.
//
// Original W layout: row-major [K, N].
// Packed layout: each output column is stored contiguously, two signed 4-bit values per byte.
// Signed INT4 range used here: [-8, 7]. Scale is max(abs(W[:,n])) / 7.
//
// This is a memory-focused implementation. It is intentionally scalar while unpacking nibbles;
// production engines use architecture-specific packed-nibble SIMD kernels.
class QuantizedMatrixInt4 {
private:
    size_t rows_ = 0; // K
    size_t cols_ = 0; // N
    size_t packed_col_bytes_ = 0;
    std::vector<uint8_t> packed_col_major_; // [N, ceil(K/2)]
    std::vector<double> scales_;            // [N]

    static int8_t unpack_signed4(uint8_t byte, bool high) {
        uint8_t nib = high ? static_cast<uint8_t>((byte >> 4) & 0x0F) : static_cast<uint8_t>(byte & 0x0F);
        // two's-complement sign extend 4-bit to 8-bit
        return static_cast<int8_t>(nib & 0x08 ? static_cast<int>(nib) - 16 : static_cast<int>(nib));
    }

    static uint8_t pack_signed4(int8_t lo, int8_t hi) {
        uint8_t ulo = static_cast<uint8_t>(lo) & 0x0F;
        uint8_t uhi = static_cast<uint8_t>(hi) & 0x0F;
        return static_cast<uint8_t>(ulo | (uhi << 4));
    }

    double dot_col(const double* x, size_t col) const {
        const uint8_t* packed = packed_col_major_.data() + col * packed_col_bytes_;
        double sum = 0.0;
        size_t k = 0;
        for (size_t b = 0; b < packed_col_bytes_; ++b) {
            uint8_t byte = packed[b];
            if (k < rows_) sum += x[k++] * static_cast<double>(unpack_signed4(byte, false));
            if (k < rows_) sum += x[k++] * static_cast<double>(unpack_signed4(byte, true));
        }
        return sum;
    }

public:
    QuantizedMatrixInt4() = default;

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    bool empty() const { return rows_ == 0 || cols_ == 0 || packed_col_major_.empty(); }
    size_t bytes() const { return packed_col_major_.size() * sizeof(uint8_t) + scales_.size() * sizeof(double); }

    void quantize_from_row_major(const std::vector<double>& w, size_t rows, size_t cols) {
        if (w.size() != rows * cols) throw std::invalid_argument("QuantizedMatrixInt4 shape mismatch");
        rows_ = rows;
        cols_ = cols;
        packed_col_bytes_ = (rows_ + 1) / 2;
        packed_col_major_.assign(cols_ * packed_col_bytes_, 0);
        scales_.assign(cols_, 1.0);

        for (size_t n = 0; n < cols_; ++n) {
            double max_abs = 0.0;
            for (size_t k = 0; k < rows_; ++k) max_abs = std::max(max_abs, std::abs(w[k * cols_ + n]));
            double scale = max_abs > 0.0 ? max_abs / 7.0 : 1.0;
            scales_[n] = scale;

            for (size_t k = 0; k < rows_; k += 2) {
                auto quant_one = [&](size_t kk) -> int8_t {
                    if (kk >= rows_) return 0;
                    long q = std::lround(w[kk * cols_ + n] / scale);
                    q = std::max<long>(-8, std::min<long>(7, q));
                    return static_cast<int8_t>(q);
                };
                int8_t lo = quant_one(k);
                int8_t hi = quant_one(k + 1);
                packed_col_major_[n * packed_col_bytes_ + k / 2] = pack_signed4(lo, hi);
            }
        }
    }

    void matmul(const std::vector<double>& x,
                size_t m,
                size_t k,
                std::vector<double>& y,
                size_t n,
                size_t num_threads = 1) const {
        if (empty()) throw std::runtime_error("QuantizedMatrixInt4 is empty");
        if (k != rows_ || n != cols_) throw std::invalid_argument("QuantizedMatrixInt4 matmul shape mismatch");
        if (x.size() != m * k) throw std::invalid_argument("QuantizedMatrixInt4 input shape mismatch");
        y.assign(m * n, 0.0);

        tiny_threads::parallel_for(static_cast<size_t>(0), m * n, num_threads, [&](size_t idx) {
            size_t row = idx / n;
            size_t col = idx % n;
            const double* xrow = x.data() + row * k;
            y[row * n + col] = dot_col(xrow, col) * scales_[col];
        });
    }
};
