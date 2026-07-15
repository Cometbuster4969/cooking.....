#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "thread_pool.hpp"

#if defined(USE_CBLAS)
extern "C" {
#include <cblas.h>
}
#endif

// Runtime backend selector for dense FP matmul. Default build is portable CPU.
// Optional BLAS build:
//   g++ ... -DUSE_CBLAS -lopenblas
// or link against MKL/Accelerate providing cblas_dgemm.
enum class DenseBackend {
    CpuScalar,
    CpuThreaded,
    CBLAS
};

struct MatmulRuntime {
    static size_t& threads() {
        static size_t t = 1;
        return t;
    }

    static DenseBackend& dense_backend() {
        static DenseBackend b = DenseBackend::CpuScalar;
        return b;
    }

    static void set_threads(size_t n) { threads() = n == 0 ? 1 : n; }

    static void set_backend_from_string(const std::string& name) {
        if (name == "cpu" || name == "scalar") dense_backend() = DenseBackend::CpuScalar;
        else if (name == "threaded" || name == "threads") dense_backend() = DenseBackend::CpuThreaded;
        else if (name == "blas" || name == "cblas") dense_backend() = DenseBackend::CBLAS;
        else if (name == "cuda" || name == "metal") {
            throw std::runtime_error(name + " backend is not compiled in this portable C++ build. Use the backend interface to add a platform kernel.");
        } else {
            throw std::runtime_error("Unknown dense backend: " + name);
        }
    }

    static const char* backend_name() {
        switch (dense_backend()) {
            case DenseBackend::CpuScalar: return "CPU scalar";
            case DenseBackend::CpuThreaded: return "CPU threaded";
            case DenseBackend::CBLAS: return "CBLAS dgemm";
        }
        return "unknown";
    }
};

inline void dense_matmul_backend(const std::vector<double>& A,
                                 const std::vector<double>& B,
                                 std::vector<double>& C,
                                 size_t M,
                                 size_t K,
                                 size_t N) {
#if defined(USE_CBLAS)
    if (MatmulRuntime::dense_backend() == DenseBackend::CBLAS) {
        C.assign(M * N, 0.0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0, A.data(), static_cast<int>(K),
                    B.data(), static_cast<int>(N),
                    0.0, C.data(), static_cast<int>(N));
        return;
    }
#else
    if (MatmulRuntime::dense_backend() == DenseBackend::CBLAS) {
        throw std::runtime_error("CBLAS backend requested but binary was not compiled with -DUSE_CBLAS and linked to BLAS");
    }
#endif

    C.assign(M * N, 0.0);
    auto compute_row = [&](size_t i) {
        for (size_t k = 0; k < K; ++k) {
            double a = A[i * K + k];
            for (size_t j = 0; j < N; ++j) C[i * N + j] += a * B[k * N + j];
        }
    };

    if (MatmulRuntime::dense_backend() == DenseBackend::CpuThreaded && MatmulRuntime::threads() > 1) {
        tiny_threads::parallel_for(static_cast<size_t>(0), M, MatmulRuntime::threads(), compute_row);
    } else {
        for (size_t i = 0; i < M; ++i) compute_row(i);
    }
}
