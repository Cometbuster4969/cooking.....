#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "thread_pool.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Symmetric per-output-column INT8 weight-only quantization for Y = X * W.
//
// Original W layout: row-major [K, N], where K=input features and N=output features.
// Quantized layout: column-major [N, K], so each output neuron's weight vector is contiguous.
//
// Each column n has scale[n] = max(abs(W[:,n])) / 127.
// Dequantization during matmul: W[k,n] ~= q[n*K+k] * scale[n].
//
// This is intentionally simple and dependency-free. Production engines usually use INT4/INT8
// block quantization, activation quantization, packed kernels, prefetching, thread pools, etc.
class QuantizedMatrixInt8 {
private:
    size_t rows_ = 0; // K
    size_t cols_ = 0; // N
    std::vector<int8_t> q_col_major_; // [N, K]
    std::vector<double> scales_;      // [N]

#if defined(__AVX2__)
    static double dot_i8_f64_avx2(const double* x, const int8_t* q, size_t k) {
        size_t i = 0;
        __m256d acc = _mm256_setzero_pd();
        for (; i + 4 <= k; i += 4) {
            __m256d xv = _mm256_loadu_pd(x + i);

            int32_t packed = 0;
            std::memcpy(&packed, q + i, 4);
            __m128i qb = _mm_cvtsi32_si128(packed);
            __m128i qi32 = _mm_cvtepi8_epi32(qb);
            __m256d qd = _mm256_cvtepi32_pd(qi32);

#if defined(__FMA__)
            acc = _mm256_fmadd_pd(xv, qd, acc);
#else
            acc = _mm256_add_pd(acc, _mm256_mul_pd(xv, qd));
#endif
        }

        alignas(32) double tmp[4];
        _mm256_store_pd(tmp, acc);
        double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        for (; i < k; ++i) sum += x[i] * static_cast<double>(q[i]);
        return sum;
    }
#endif

    static double dot_i8_f64_scalar(const double* x, const int8_t* q, size_t k) {
        double sum = 0.0;
        for (size_t i = 0; i < k; ++i) sum += x[i] * static_cast<double>(q[i]);
        return sum;
    }

public:
    QuantizedMatrixInt8() = default;

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    bool empty() const { return rows_ == 0 || cols_ == 0 || q_col_major_.empty(); }
    size_t bytes() const { return q_col_major_.size() * sizeof(int8_t) + scales_.size() * sizeof(double); }

    void quantize_from_row_major(const std::vector<double>& w, size_t rows, size_t cols) {
        if (w.size() != rows * cols) throw std::invalid_argument("QuantizedMatrixInt8 shape mismatch");
        rows_ = rows;
        cols_ = cols;
        q_col_major_.assign(rows_ * cols_, 0);
        scales_.assign(cols_, 1.0);

        for (size_t n = 0; n < cols_; ++n) {
            double max_abs = 0.0;
            for (size_t k = 0; k < rows_; ++k) {
                max_abs = std::max(max_abs, std::abs(w[k * cols_ + n]));
            }
            double scale = max_abs > 0.0 ? max_abs / 127.0 : 1.0;
            scales_[n] = scale;

            for (size_t k = 0; k < rows_; ++k) {
                double v = w[k * cols_ + n] / scale;
                long rounded = std::lround(v);
                rounded = std::max<long>(-127, std::min<long>(127, rounded));
                q_col_major_[n * rows_ + k] = static_cast<int8_t>(rounded);
            }
        }
    }

    void matmul(const std::vector<double>& x, size_t m, size_t k, std::vector<double>& y, size_t n, size_t num_threads = 1) const {
        if (empty()) throw std::runtime_error("QuantizedMatrixInt8 is empty");
        if (k != rows_ || n != cols_) throw std::invalid_argument("QuantizedMatrixInt8 matmul shape mismatch");
        if (x.size() != m * k) throw std::invalid_argument("QuantizedMatrixInt8 input shape mismatch");
        y.assign(m * n, 0.0);

        tiny_threads::parallel_for(static_cast<size_t>(0), m * n, num_threads, [&](size_t idx) {
            size_t row = idx / n;
            size_t col = idx % n;
            const double* xrow = x.data() + row * k;
            const int8_t* qcol = q_col_major_.data() + col * rows_;
#if defined(__AVX2__)
            double qdot = dot_i8_f64_avx2(xrow, qcol, k);
#else
            double qdot = dot_i8_f64_scalar(xrow, qcol, k);
#endif
            y[row * n + col] = qdot * scales_[col];
        });
    }
};
