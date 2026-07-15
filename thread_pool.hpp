#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

// Tiny dependency-free parallel_for.
// It creates worker threads per call. That is simple and portable; production engines keep
// persistent worker threads alive to avoid launch overhead.
namespace tiny_threads {

inline size_t hardware_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : static_cast<size_t>(n);
}

template <class Fn>
void parallel_for(size_t begin, size_t end, size_t num_threads, Fn fn) {
    if (end <= begin) return;
    size_t total = end - begin;
    if (num_threads <= 1 || total <= 1) {
        for (size_t i = begin; i < end; ++i) fn(i);
        return;
    }

    num_threads = std::min(num_threads, total);
    std::atomic<size_t> next{begin};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        workers.emplace_back([&]() {
            while (true) {
                size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= end) break;
                fn(i);
            }
        });
    }

    for (auto& w : workers) w.join();
}

template <class Fn>
void parallel_for_blocks(size_t begin, size_t end, size_t num_threads, size_t block_size, Fn fn) {
    if (end <= begin) return;
    if (block_size == 0) throw std::invalid_argument("block_size must be positive");
    size_t total_blocks = (end - begin + block_size - 1) / block_size;
    parallel_for(static_cast<size_t>(0), total_blocks, num_threads, [&](size_t b) {
        size_t lo = begin + b * block_size;
        size_t hi = std::min(end, lo + block_size);
        fn(lo, hi);
    });
}

} // namespace tiny_threads
