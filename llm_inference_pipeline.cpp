#include <algorithm>
#include <cmath>
#include <cctype>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "safetensors_loader.hpp"
#include "bpe_tokenizer.hpp"
#include "sentencepiece_tokenizer.hpp"
#include "llm_config.hpp"
#include "quantized_matmul.hpp"
#include "quantized_matmul_int4.hpp"
#include "matmul_backend.hpp"

// -----------------------------
// Tensor: flat row-major storage
// -----------------------------
struct Tensor {
    std::vector<size_t> shape;
    std::vector<double> data;

    explicit Tensor(std::vector<size_t> s, bool init_xavier = false) : shape(std::move(s)) {
        if (shape.empty()) throw std::invalid_argument("Tensor shape cannot be empty");

        size_t total_size = 1;
        for (size_t dim : shape) {
            if (dim == 0) throw std::invalid_argument("Tensor dimensions must be non-zero");
            total_size *= dim;
        }
        data.assign(total_size, 0.0);

        if (init_xavier) {
            size_t fan_in = shape.size() >= 2 ? shape[0] : shape[0];
            size_t fan_out = shape.size() >= 2 ? shape[1] : shape[0];

            static std::mt19937 gen(1337);
            double limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
            std::uniform_real_distribution<double> dist(-limit, limit);
            for (auto& val : data) val = dist(gen);
        }
    }
};

// -----------------------------
// Tokenizer: text <-> token IDs
// -----------------------------
class Tokenizer {
private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

    static std::string lower(std::string s) {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

public:
    Tokenizer() {
        // Tiny demonstration vocabulary. Real LLMs use BPE/SentencePiece vocabularies with thousands of tokens.
        id_to_token_ = {
            "<unk>", "<bos>", "<eos>", "<pad>",
            "hello", "world", "i", "you", "we", "they", "am", "are", "is",
            "a", "an", "the", "tiny", "llm", "model", "decoder", "token", "tokens",
            "build", "generates", "text", "from", "prompt", "with", "attention", "and",
            "rope", "swiglu", "inference", "pipeline", "works", "!", ".", ",", "?"
        };

        for (size_t i = 0; i < id_to_token_.size(); ++i) {
            token_to_id_[id_to_token_[i]] = static_cast<int>(i);
        }
    }

    int unk_id() const { return 0; }
    int bos_id() const { return 1; }
    int eos_id() const { return 2; }
    size_t vocab_size() const { return id_to_token_.size(); }

    std::vector<int> encode(const std::string& text, bool add_bos = true) const {
        std::vector<int> ids;
        if (add_bos) ids.push_back(bos_id());

        std::string current;
        auto flush_word = [&]() {
            if (!current.empty()) {
                std::string word = lower(current);
                auto it = token_to_id_.find(word);
                ids.push_back(it == token_to_id_.end() ? unk_id() : it->second);
                current.clear();
            }
        };

        for (char ch : text) {
            unsigned char uch = static_cast<unsigned char>(ch);
            if (std::isalnum(uch)) {
                current.push_back(ch);
            } else {
                flush_word();
                if (!std::isspace(uch)) {
                    std::string punct(1, ch);
                    auto it = token_to_id_.find(punct);
                    ids.push_back(it == token_to_id_.end() ? unk_id() : it->second);
                }
            }
        }
        flush_word();
        return ids;
    }

    std::string decode(const std::vector<int>& ids, bool skip_special = true) const {
        std::ostringstream oss;
        bool first = true;
        for (int id : ids) {
            if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) id = unk_id();
            const std::string& tok = id_to_token_[static_cast<size_t>(id)];

            if (skip_special && (tok == "<bos>" || tok == "<eos>" || tok == "<pad>")) continue;

            bool punctuation = tok == "." || tok == "," || tok == "!" || tok == "?";
            if (!first && !punctuation) oss << ' ';
            oss << tok;
            first = false;
        }
        return oss.str();
    }

    std::string token(int id) const {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return "<unk>";
        return id_to_token_[static_cast<size_t>(id)];
    }
};

// -----------------------------
// Core neural-network operators
// -----------------------------
class ModernLLMEngine {
public:
    static void rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, double eps = 1e-6) {
        if (x.shape.size() != 2 || out.shape.size() != 2 || weight.shape.size() != 1) {
            throw std::invalid_argument("rms_norm expects x/out as 2D and weight as 1D");
        }
        size_t seq_len = x.shape[0];
        size_t embed_dim = x.shape[1];
        if (out.shape[0] != seq_len || out.shape[1] != embed_dim || weight.shape[0] != embed_dim) {
            throw std::invalid_argument("rms_norm shape mismatch");
        }

        for (size_t i = 0; i < seq_len; ++i) {
            double ss = 0.0;
            for (size_t j = 0; j < embed_dim; ++j) {
                double val = x.data[i * embed_dim + j];
                ss += val * val;
            }
            double inv_rms = 1.0 / std::sqrt(ss / static_cast<double>(embed_dim) + eps);
            for (size_t j = 0; j < embed_dim; ++j) {
                out.data[i * embed_dim + j] = x.data[i * embed_dim + j] * inv_rms * weight.data[j];
            }
        }
    }

    static void swiglu(const Tensor& gate, const Tensor& up, Tensor& out) {
        if (gate.data.size() != up.data.size() || gate.data.size() != out.data.size()) {
            throw std::invalid_argument("swiglu shape mismatch");
        }
        for (size_t i = 0; i < out.data.size(); ++i) {
            double g = gate.data[i];
            double silu = g / (1.0 + std::exp(-g));
            out.data[i] = silu * up.data[i];
        }
    }

    static void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
        if (A.shape.size() != 2 || B.shape.size() != 2 || C.shape.size() != 2) {
            throw std::invalid_argument("matmul expects 2D tensors");
        }
        size_t M = A.shape[0], K = A.shape[1], Kb = B.shape[0], N = B.shape[1];
        if (K != Kb || C.shape[0] != M || C.shape[1] != N) {
            throw std::invalid_argument("matmul shape mismatch");
        }

        dense_matmul_backend(A.data, B.data, C.data, M, K, N);
    }

    static void apply_rope(Tensor& t, size_t num_heads, size_t head_dim, double rope_theta = 10000.0) {
        if (t.shape.size() != 2) throw std::invalid_argument("apply_rope expects 2D tensor");
        if (head_dim % 2 != 0) throw std::invalid_argument("RoPE requires even head_dim");
        if (t.shape[1] != num_heads * head_dim) throw std::invalid_argument("apply_rope shape mismatch");

        size_t seq_len = t.shape[0];
        for (size_t i = 0; i < seq_len; ++i) {
            apply_rope_at_position(t, num_heads, head_dim, i, i, rope_theta);
        }
    }

    // RoPE for one row using an absolute sequence position. This is required by the KV-cache
    // path, where the model receives only one new token at a time but still needs its real
    // position in the full generated sequence.
    static void apply_rope_at_position(Tensor& t, size_t num_heads, size_t head_dim, size_t row, size_t absolute_pos, double rope_theta = 10000.0) {
        if (t.shape.size() != 2) throw std::invalid_argument("apply_rope_at_position expects 2D tensor");
        if (head_dim % 2 != 0) throw std::invalid_argument("RoPE requires even head_dim");
        if (t.shape[1] != num_heads * head_dim) throw std::invalid_argument("apply_rope_at_position shape mismatch");
        if (row >= t.shape[0]) throw std::invalid_argument("RoPE row out of range");

        for (size_t h = 0; h < num_heads; ++h) {
            size_t head_offset = row * num_heads * head_dim + h * head_dim;
            for (size_t d = 0; d < head_dim; d += 2) {
                double theta = 1.0 / std::pow(rope_theta, static_cast<double>(d) / static_cast<double>(head_dim));
                double angle = static_cast<double>(absolute_pos) * theta;
                double cos_a = std::cos(angle), sin_a = std::sin(angle);
                double x0 = t.data[head_offset + d];
                double x1 = t.data[head_offset + d + 1];
                t.data[head_offset + d] = x0 * cos_a - x1 * sin_a;
                t.data[head_offset + d + 1] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
};

// -----------------------------
// One decoder block
// -----------------------------
class DecoderLayer {
private:
    size_t embed_dim, num_heads, num_kv_heads, head_dim, kv_dim, kv_group_size, hidden_dim;
    double rms_eps, rope_theta;
    Tensor rms_attn_w, rms_ffn_w;
    Tensor Wq, Wk, Wv, Wo;
    Tensor W_gate, W_up, W_down;

    QuantizedMatrixInt8 qWq, qWk, qWv, qWo;
    QuantizedMatrixInt8 qW_gate, qW_up, qW_down;
    QuantizedMatrixInt4 q4Wq, q4Wk, q4Wv, q4Wo;
    QuantizedMatrixInt4 q4W_gate, q4W_up, q4W_down;
    int weight_mode_ = 0; // 0=FP, 8=INT8, 4=INT4

    void matmul_weight(const Tensor& A, const Tensor& W, const QuantizedMatrixInt8& QW, const QuantizedMatrixInt4& Q4W, Tensor& C) const {
        if (weight_mode_ == 8) QW.matmul(A.data, A.shape[0], A.shape[1], C.data, C.shape[1], MatmulRuntime::threads());
        else if (weight_mode_ == 4) Q4W.matmul(A.data, A.shape[0], A.shape[1], C.data, C.shape[1], MatmulRuntime::threads());
        else ModernLLMEngine::matmul(A, W, C);
    }

    // Persistent generation-time KV cache. Layout is [cache_len, embed_dim].
    // Each decoder layer owns its own cache because K/V are layer-specific.
    std::vector<double> k_cache_;
    std::vector<double> v_cache_;
    size_t cache_len_ = 0;

public:
    DecoderLayer(size_t embed, size_t heads, size_t hidden)
        : DecoderLayer(embed, heads, heads, hidden, 1e-6, 10000.0) {}

    DecoderLayer(size_t embed, size_t heads, size_t kv_heads, size_t hidden, double eps = 1e-6, double rope = 10000.0)
        : embed_dim(embed),
          num_heads(heads),
          num_kv_heads(kv_heads),
          head_dim(embed / heads),
          kv_dim(kv_heads * (embed / heads)),
          kv_group_size(heads / kv_heads),
          hidden_dim(hidden),
          rms_eps(eps),
          rope_theta(rope),
          rms_attn_w({embed}),
          rms_ffn_w({embed}),
          Wq({embed, embed}, true), Wk({embed, kv_heads * (embed / heads)}, true), Wv({embed, kv_heads * (embed / heads)}, true), Wo({embed, embed}, true),
          W_gate({embed, hidden}, true), W_up({embed, hidden}, true), W_down({hidden, embed}, true) {
        if (heads == 0 || kv_heads == 0 || embed % heads != 0) throw std::invalid_argument("embed_dim must be divisible by num_heads");
        if (heads % kv_heads != 0) throw std::invalid_argument("num_heads must be divisible by num_kv_heads");
        if (head_dim % 2 != 0) throw std::invalid_argument("head_dim must be even for RoPE");
        std::fill(rms_attn_w.data.begin(), rms_attn_w.data.end(), 1.0);
        std::fill(rms_ffn_w.data.begin(), rms_ffn_w.data.end(), 1.0);
    }

    // Load one HuggingFace/LLaMA-style decoder layer from a .safetensors checkpoint.
    // HF Linear weights are stored as [out_features, in_features], while this demo engine
    // multiplies row vectors as X * W and therefore stores them as [in_features, out_features].
    // That is why all projection/MLP matrices are loaded with transpose_2d=true.
    void load_from_safetensors(const SafeTensors& st, size_t layer_idx, const std::string& prefix = "model.layers.") {
        std::string base = prefix + std::to_string(layer_idx) + ".";

        st.copy_to(base + "input_layernorm.weight", rms_attn_w.data, rms_attn_w.shape);
        st.copy_to(base + "post_attention_layernorm.weight", rms_ffn_w.data, rms_ffn_w.shape);

        st.copy_to(base + "self_attn.q_proj.weight", Wq.data, Wq.shape, true);
        st.copy_to(base + "self_attn.k_proj.weight", Wk.data, Wk.shape, true);
        st.copy_to(base + "self_attn.v_proj.weight", Wv.data, Wv.shape, true);
        st.copy_to(base + "self_attn.o_proj.weight", Wo.data, Wo.shape, true);

        st.copy_to(base + "mlp.gate_proj.weight", W_gate.data, W_gate.shape, true);
        st.copy_to(base + "mlp.up_proj.weight", W_up.data, W_up.shape, true);
        st.copy_to(base + "mlp.down_proj.weight", W_down.data, W_down.shape, true);
    }

    void quantize_weights_int8(bool release_fp = true) {
        qWq.quantize_from_row_major(Wq.data, Wq.shape[0], Wq.shape[1]);
        qWk.quantize_from_row_major(Wk.data, Wk.shape[0], Wk.shape[1]);
        qWv.quantize_from_row_major(Wv.data, Wv.shape[0], Wv.shape[1]);
        qWo.quantize_from_row_major(Wo.data, Wo.shape[0], Wo.shape[1]);
        qW_gate.quantize_from_row_major(W_gate.data, W_gate.shape[0], W_gate.shape[1]);
        qW_up.quantize_from_row_major(W_up.data, W_up.shape[0], W_up.shape[1]);
        qW_down.quantize_from_row_major(W_down.data, W_down.shape[0], W_down.shape[1]);
        weight_mode_ = 8;
        if (release_fp) {
            auto drop = [](Tensor& t) { std::vector<double>().swap(t.data); };
            drop(Wq); drop(Wk); drop(Wv); drop(Wo);
            drop(W_gate); drop(W_up); drop(W_down);
        }
    }

    void quantize_weights_int4(bool release_fp = true) {
        q4Wq.quantize_from_row_major(Wq.data, Wq.shape[0], Wq.shape[1]);
        q4Wk.quantize_from_row_major(Wk.data, Wk.shape[0], Wk.shape[1]);
        q4Wv.quantize_from_row_major(Wv.data, Wv.shape[0], Wv.shape[1]);
        q4Wo.quantize_from_row_major(Wo.data, Wo.shape[0], Wo.shape[1]);
        q4W_gate.quantize_from_row_major(W_gate.data, W_gate.shape[0], W_gate.shape[1]);
        q4W_up.quantize_from_row_major(W_up.data, W_up.shape[0], W_up.shape[1]);
        q4W_down.quantize_from_row_major(W_down.data, W_down.shape[0], W_down.shape[1]);
        weight_mode_ = 4;
        if (release_fp) {
            auto drop = [](Tensor& t) { std::vector<double>().swap(t.data); };
            drop(Wq); drop(Wk); drop(Wv); drop(Wo);
            drop(W_gate); drop(W_up); drop(W_down);
        }
    }

    void reset_kv_cache() {
        k_cache_.clear();
        v_cache_.clear();
        cache_len_ = 0;
    }

    size_t kv_cache_len() const { return cache_len_; }

    // Incremental decoder pass for exactly one token. It computes Q/K/V for that token,
    // appends K and V to this layer's cache, and attends the single Q against all cached K/V.
    // Complexity per generated token becomes O(context_length) for attention instead of
    // recomputing the full O(context_length^2) prompt on every generation step.
    Tensor forward_incremental(const Tensor& X_one, size_t absolute_pos) {
        if (X_one.shape.size() != 2 || X_one.shape[0] != 1 || X_one.shape[1] != embed_dim) {
            throw std::invalid_argument("forward_incremental expects X with shape [1, embed_dim]");
        }

        Tensor norm_X({1, embed_dim});
        ModernLLMEngine::rms_norm(X_one, rms_attn_w, norm_X, rms_eps);

        Tensor Q({1, embed_dim}), K({1, kv_dim}), V({1, kv_dim});
        matmul_weight(norm_X, Wq, qWq, q4Wq, Q);
        matmul_weight(norm_X, Wk, qWk, q4Wk, K);
        matmul_weight(norm_X, Wv, qWv, q4Wv, V);
        ModernLLMEngine::apply_rope_at_position(Q, num_heads, head_dim, 0, absolute_pos, rope_theta);
        ModernLLMEngine::apply_rope_at_position(K, num_kv_heads, head_dim, 0, absolute_pos, rope_theta);

        k_cache_.insert(k_cache_.end(), K.data.begin(), K.data.end());
        v_cache_.insert(v_cache_.end(), V.data.begin(), V.data.end());
        ++cache_len_;

        Tensor Attn_Out({1, embed_dim});
        double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

        for (size_t h = 0; h < num_heads; ++h) {
            std::vector<double> scores(cache_len_);
            double max_score = -1e300;
            size_t q_off = h * head_dim;

            for (size_t t = 0; t < cache_len_; ++t) {
                size_t kv_h = h / kv_group_size;
                size_t k_off = t * kv_dim + kv_h * head_dim;
                double score = 0.0;
                for (size_t d = 0; d < head_dim; ++d) score += Q.data[q_off + d] * k_cache_[k_off + d];
                score *= scale;
                scores[t] = score;
                max_score = std::max(max_score, score);
            }

            double sum = 0.0;
            for (double& s : scores) {
                s = std::exp(s - max_score);
                sum += s;
            }
            for (double& s : scores) s /= sum;

            for (size_t d = 0; d < head_dim; ++d) {
                double val_sum = 0.0;
                for (size_t t = 0; t < cache_len_; ++t) {
                    size_t kv_h = h / kv_group_size;
                    size_t v_off = t * kv_dim + kv_h * head_dim;
                    val_sum += scores[t] * v_cache_[v_off + d];
                }
                Attn_Out.data[h * head_dim + d] = val_sum;
            }
        }

        Tensor Projected_Attn({1, embed_dim});
        matmul_weight(Attn_Out, Wo, qWo, q4Wo, Projected_Attn);

        Tensor Mid_X({1, embed_dim});
        for (size_t i = 0; i < embed_dim; ++i) Mid_X.data[i] = X_one.data[i] + Projected_Attn.data[i];

        Tensor norm_Mid({1, embed_dim});
        ModernLLMEngine::rms_norm(Mid_X, rms_ffn_w, norm_Mid, rms_eps);

        Tensor Gate_Out({1, hidden_dim}), Up_Out({1, hidden_dim});
        matmul_weight(norm_Mid, W_gate, qW_gate, q4W_gate, Gate_Out);
        matmul_weight(norm_Mid, W_up, qW_up, q4W_up, Up_Out);

        Tensor SwiGLU_Out({1, hidden_dim});
        ModernLLMEngine::swiglu(Gate_Out, Up_Out, SwiGLU_Out);

        Tensor FFN_Out({1, embed_dim});
        matmul_weight(SwiGLU_Out, W_down, qW_down, q4W_down, FFN_Out);

        Tensor Final_Out({1, embed_dim});
        for (size_t i = 0; i < embed_dim; ++i) Final_Out.data[i] = Mid_X.data[i] + FFN_Out.data[i];
        return Final_Out;
    }

    Tensor forward(const Tensor& X) const {
        if (X.shape.size() != 2 || X.shape[1] != embed_dim) {
            throw std::invalid_argument("DecoderLayer expects X with shape [seq_len, embed_dim]");
        }

        size_t seq_len = X.shape[0];
        Tensor norm_X({seq_len, embed_dim});
        ModernLLMEngine::rms_norm(X, rms_attn_w, norm_X, rms_eps);

        Tensor Q({seq_len, embed_dim}), K({seq_len, kv_dim}), V({seq_len, kv_dim});
        matmul_weight(norm_X, Wq, qWq, q4Wq, Q);
        matmul_weight(norm_X, Wk, qWk, q4Wk, K);
        matmul_weight(norm_X, Wv, qWv, q4Wv, V);
        ModernLLMEngine::apply_rope(Q, num_heads, head_dim, rope_theta);
        ModernLLMEngine::apply_rope(K, num_kv_heads, head_dim, rope_theta);

        Tensor Attn_Out({seq_len, embed_dim});
        double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

        for (size_t h = 0; h < num_heads; ++h) {
            Tensor Scores({seq_len, seq_len});

            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    double score = 0.0;
                    size_t q_off = i * num_heads * head_dim + h * head_dim;
                    size_t kv_h = h / kv_group_size;
                    size_t k_off = j * kv_dim + kv_h * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) score += Q.data[q_off + d] * K.data[k_off + d];
                    Scores.data[i * seq_len + j] = score * scale;
                }
                for (size_t j = i + 1; j < seq_len; ++j) Scores.data[i * seq_len + j] = -1e9;
            }

            for (size_t i = 0; i < seq_len; ++i) {
                double max_val = Scores.data[i * seq_len];
                for (size_t j = 1; j < seq_len; ++j) max_val = std::max(max_val, Scores.data[i * seq_len + j]);

                double sum = 0.0;
                for (size_t j = 0; j < seq_len; ++j) {
                    Scores.data[i * seq_len + j] = std::exp(Scores.data[i * seq_len + j] - max_val);
                    sum += Scores.data[i * seq_len + j];
                }
                for (size_t j = 0; j < seq_len; ++j) Scores.data[i * seq_len + j] /= sum;
            }

            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < head_dim; ++d) {
                    double val_sum = 0.0;
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t kv_h = h / kv_group_size;
                        size_t v_off = j * kv_dim + kv_h * head_dim;
                        val_sum += Scores.data[i * seq_len + j] * V.data[v_off + d];
                    }
                    Attn_Out.data[i * embed_dim + h * head_dim + d] = val_sum;
                }
            }
        }

        Tensor Projected_Attn({seq_len, embed_dim});
        matmul_weight(Attn_Out, Wo, qWo, q4Wo, Projected_Attn);

        Tensor Mid_X({seq_len, embed_dim});
        for (size_t i = 0; i < X.data.size(); ++i) Mid_X.data[i] = X.data[i] + Projected_Attn.data[i];

        Tensor norm_Mid({seq_len, embed_dim});
        ModernLLMEngine::rms_norm(Mid_X, rms_ffn_w, norm_Mid, rms_eps);

        Tensor Gate_Out({seq_len, hidden_dim}), Up_Out({seq_len, hidden_dim});
        matmul_weight(norm_Mid, W_gate, qW_gate, q4W_gate, Gate_Out);
        matmul_weight(norm_Mid, W_up, qW_up, q4W_up, Up_Out);

        Tensor SwiGLU_Out({seq_len, hidden_dim});
        ModernLLMEngine::swiglu(Gate_Out, Up_Out, SwiGLU_Out);

        Tensor FFN_Out({seq_len, embed_dim});
        matmul_weight(SwiGLU_Out, W_down, qW_down, q4W_down, FFN_Out);

        Tensor Final_Out({seq_len, embed_dim});
        for (size_t i = 0; i < Mid_X.data.size(); ++i) Final_Out.data[i] = Mid_X.data[i] + FFN_Out.data[i];
        return Final_Out;
    }
};

// -----------------------------
// Full inference model pipeline
// -----------------------------
struct GenerationConfig {
    int max_new_tokens = 12;
    double temperature = 0.9;
    int top_k = 8;
    bool stop_on_eos = true;
};

class TransformerLM {
private:
    size_t vocab_size_, embed_dim_, num_heads_, num_kv_heads_, hidden_dim_, num_layers_;
    double rms_eps_, rope_theta_;
    Tensor token_embedding_;       // [vocab_size, embed_dim]
    std::vector<DecoderLayer> layers_;
    Tensor final_norm_w_;          // [embed_dim]
    Tensor lm_head_;               // [embed_dim, vocab_size]
    QuantizedMatrixInt8 q_lm_head_;
    QuantizedMatrixInt4 q4_lm_head_;
    int weight_mode_ = 0;
    mutable std::mt19937 sampler_rng_{2026};

public:
    TransformerLM(size_t vocab_size, size_t embed_dim, size_t num_heads, size_t hidden_dim, size_t num_layers)
        : TransformerLM(vocab_size, embed_dim, num_heads, num_heads, hidden_dim, num_layers, 1e-6, 10000.0) {}

    TransformerLM(size_t vocab_size, size_t embed_dim, size_t num_heads, size_t num_kv_heads,
                  size_t hidden_dim, size_t num_layers, double rms_eps = 1e-6, double rope_theta = 10000.0)
        : vocab_size_(vocab_size),
          embed_dim_(embed_dim),
          num_heads_(num_heads),
          num_kv_heads_(num_kv_heads),
          hidden_dim_(hidden_dim),
          num_layers_(num_layers),
          rms_eps_(rms_eps),
          rope_theta_(rope_theta),
          token_embedding_({vocab_size, embed_dim}, true),
          final_norm_w_({embed_dim}),
          lm_head_({embed_dim, vocab_size}, true) {
        if (num_layers == 0) throw std::invalid_argument("num_layers must be positive");
        layers_.reserve(num_layers);
        for (size_t i = 0; i < num_layers; ++i) layers_.emplace_back(embed_dim, num_heads, num_kv_heads, hidden_dim, rms_eps, rope_theta);
        std::fill(final_norm_w_.data.begin(), final_norm_w_.data.end(), 1.0);
    }

    explicit TransformerLM(const LLMConfig& cfg)
        : TransformerLM(static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size),
                        static_cast<size_t>(cfg.num_attention_heads), static_cast<size_t>(cfg.num_key_value_heads),
                        static_cast<size_t>(cfg.intermediate_size), static_cast<size_t>(cfg.num_hidden_layers),
                        cfg.rms_norm_eps, cfg.rope_theta) {}

    void quantize_all_weights_int8(bool release_fp = true) {
        for (auto& layer : layers_) layer.quantize_weights_int8(release_fp);
        q_lm_head_.quantize_from_row_major(lm_head_.data, lm_head_.shape[0], lm_head_.shape[1]);
        weight_mode_ = 8;
        if (release_fp) std::vector<double>().swap(lm_head_.data);
    }

    void quantize_all_weights_int4(bool release_fp = true) {
        for (auto& layer : layers_) layer.quantize_weights_int4(release_fp);
        q4_lm_head_.quantize_from_row_major(lm_head_.data, lm_head_.shape[0], lm_head_.shape[1]);
        weight_mode_ = 4;
        if (release_fp) std::vector<double>().swap(lm_head_.data);
    }

    // Load a single-file HuggingFace/LLaMA-style .safetensors checkpoint into the model.
    // For sharded checkpoints, call this with a merged file or extend it to iterate over
    // multiple SafeTensors objects.
    void load_llama_safetensors(const std::string& path) {
        SafeTensors st(path);
        load_llama_safetensors(st);
    }

    void load_llama_safetensors(const SafeTensors& st) {
        st.copy_to("model.embed_tokens.weight", token_embedding_.data, token_embedding_.shape);
        st.copy_to("model.norm.weight", final_norm_w_.data, final_norm_w_.shape);

        for (size_t i = 0; i < layers_.size(); ++i) {
            layers_[i].load_from_safetensors(st, i);
        }

        if (st.contains("lm_head.weight")) {
            // HF lm_head.weight is [vocab, embed]; this engine wants [embed, vocab].
            st.copy_to("lm_head.weight", lm_head_.data, lm_head_.shape, true);
        } else {
            // Some causal LMs tie output projection to token embeddings.
            for (size_t v = 0; v < vocab_size_; ++v) {
                for (size_t d = 0; d < embed_dim_; ++d) {
                    lm_head_.data[d * vocab_size_ + v] = token_embedding_.data[v * embed_dim_ + d];
                }
            }
        }
    }

    Tensor embed_tokens(const std::vector<int>& token_ids) const {
        Tensor X({token_ids.size(), embed_dim_});
        for (size_t pos = 0; pos < token_ids.size(); ++pos) {
            int id = token_ids[pos];
            if (id < 0 || static_cast<size_t>(id) >= vocab_size_) id = 0;
            for (size_t d = 0; d < embed_dim_; ++d) {
                X.data[pos * embed_dim_ + d] = token_embedding_.data[static_cast<size_t>(id) * embed_dim_ + d];
            }
        }
        return X;
    }

    Tensor forward_logits(const std::vector<int>& token_ids) const {
        if (token_ids.empty()) throw std::invalid_argument("Cannot run model on an empty token sequence");

        // Embedding layer: token IDs -> dense vectors.
        Tensor X = embed_tokens(token_ids);

        // Multi-layer decoder stack: output of layer n becomes input of layer n + 1.
        for (const auto& layer : layers_) {
            X = layer.forward(X);
        }

        // Final RMSNorm before the LM head, common in LLaMA-style models.
        Tensor norm_X({token_ids.size(), embed_dim_});
        ModernLLMEngine::rms_norm(X, final_norm_w_, norm_X, rms_eps_);

        // LM head: hidden states -> vocabulary logits.
        Tensor logits({token_ids.size(), vocab_size_});
        if (weight_mode_ == 8) q_lm_head_.matmul(norm_X.data, norm_X.shape[0], norm_X.shape[1], logits.data, logits.shape[1], MatmulRuntime::threads());
        else if (weight_mode_ == 4) q4_lm_head_.matmul(norm_X.data, norm_X.shape[0], norm_X.shape[1], logits.data, logits.shape[1], MatmulRuntime::threads());
        else ModernLLMEngine::matmul(norm_X, lm_head_, logits);
        return logits;
    }

    void reset_kv_cache() {
        for (auto& layer : layers_) layer.reset_kv_cache();
    }

    Tensor forward_one_logits(int token_id, size_t absolute_pos) {
        if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size_) token_id = 0;

        Tensor X({1, embed_dim_});
        for (size_t d = 0; d < embed_dim_; ++d) {
            X.data[d] = token_embedding_.data[static_cast<size_t>(token_id) * embed_dim_ + d];
        }

        for (auto& layer : layers_) {
            X = layer.forward_incremental(X, absolute_pos);
        }

        Tensor norm_X({1, embed_dim_});
        ModernLLMEngine::rms_norm(X, final_norm_w_, norm_X, rms_eps_);

        Tensor logits({1, vocab_size_});
        if (weight_mode_ == 8) q_lm_head_.matmul(norm_X.data, norm_X.shape[0], norm_X.shape[1], logits.data, logits.shape[1], MatmulRuntime::threads());
        else if (weight_mode_ == 4) q4_lm_head_.matmul(norm_X.data, norm_X.shape[0], norm_X.shape[1], logits.data, logits.shape[1], MatmulRuntime::threads());
        else ModernLLMEngine::matmul(norm_X, lm_head_, logits);
        return logits;
    }

    std::vector<int> generate_cached(std::vector<int> ids, const GenerationConfig& cfg, int eos_id) {
        if (ids.empty()) throw std::invalid_argument("Cannot generate from an empty prompt");
        reset_kv_cache();

        Tensor logits({1, vocab_size_});
        for (size_t pos = 0; pos < ids.size(); ++pos) {
            logits = forward_one_logits(ids[pos], pos); // prefill cache token-by-token
        }

        for (int step = 0; step < cfg.max_new_tokens; ++step) {
            int next_id = sample_next_token(logits, cfg.temperature, cfg.top_k);
            ids.push_back(next_id);
            if (cfg.stop_on_eos && next_id == eos_id) break;
            logits = forward_one_logits(next_id, ids.size() - 1);
        }
        return ids;
    }

    int sample_next_token(const Tensor& logits, double temperature, int top_k) const {
        if (logits.shape.size() != 2 || logits.shape[1] != vocab_size_) throw std::invalid_argument("Invalid logits shape");
        if (temperature <= 0.0) throw std::invalid_argument("temperature must be > 0");

        size_t seq_len = logits.shape[0];
        size_t row = seq_len - 1;

        std::vector<int> indices(vocab_size_);
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return logits.data[row * vocab_size_ + static_cast<size_t>(a)] > logits.data[row * vocab_size_ + static_cast<size_t>(b)];
        });

        int k = top_k <= 0 ? static_cast<int>(vocab_size_) : std::min(top_k, static_cast<int>(vocab_size_));
        indices.resize(static_cast<size_t>(k));

        double max_logit = -1e300;
        for (int id : indices) {
            double v = logits.data[row * vocab_size_ + static_cast<size_t>(id)] / temperature;
            max_logit = std::max(max_logit, v);
        }

        std::vector<double> probs(indices.size());
        double sum = 0.0;
        for (size_t i = 0; i < indices.size(); ++i) {
            double v = logits.data[row * vocab_size_ + static_cast<size_t>(indices[i])] / temperature;
            probs[i] = std::exp(v - max_logit);
            sum += probs[i];
        }
        for (double& p : probs) p /= sum;

        std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
        return indices[dist(sampler_rng_)];
    }

    std::vector<int> generate(std::vector<int> ids, const GenerationConfig& cfg, int eos_id) const {
        for (int step = 0; step < cfg.max_new_tokens; ++step) {
            // 1. Run the prompt/current sequence through the model.
            Tensor logits = forward_logits(ids);

            // 2 and 3. Read final-position logits and sample one next token.
            int next_id = sample_next_token(logits, cfg.temperature, cfg.top_k);

            // 4. Append token and feed the expanded sequence back into the model on the next iteration.
            ids.push_back(next_id);

            if (cfg.stop_on_eos && next_id == eos_id) break;
        }
        return ids;
    }
};

int main(int argc, char** argv) {
    try {
        std::string config_path;
        std::string weights_path;
        std::string sp_model_path;
        std::string bpe_vocab_path;
        std::string bpe_merges_path;
        std::string prompt = "Hello world";
        int max_new_tokens = 12;
        bool enable_int8 = false;
        bool enable_int4 = false;
        size_t num_threads = 1;
        std::string backend = "scalar";

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto need = [&](const std::string& flag) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("Missing value after " + flag);
                return argv[++i];
            };
            if (a == "--config") config_path = need(a);
            else if (a == "--weights") weights_path = need(a);
            else if (a == "--sp") sp_model_path = need(a);
            else if (a == "--bpe-vocab") bpe_vocab_path = need(a);
            else if (a == "--bpe-merges") bpe_merges_path = need(a);
            else if (a == "--prompt") prompt = need(a);
            else if (a == "--max-new") max_new_tokens = std::stoi(need(a));
            else if (a == "--int8") enable_int8 = true;
            else if (a == "--int4") enable_int4 = true;
            else if (a == "--threads") num_threads = static_cast<size_t>(std::stoul(need(a)));
            else if (a == "--backend") backend = need(a);
            else if (a == "--help") {
                std::cout << "Usage:\n"
                          << "  ./llm_inference_pipeline [demo]\n"
                          << "  ./llm_inference_pipeline --config config.json --weights model.safetensors --sp tokenizer.model --prompt \"Hello\"\n"
                          << "  ./llm_inference_pipeline --config config.json --weights model.safetensors --bpe-vocab vocab.json --bpe-merges merges.txt\n"
                          << "  add --int8 or --int4 to quantize weights; use --threads N; use --backend scalar|threaded|blas\n";
                return 0;
            } else if (a.size() && a[0] != '-' && weights_path.empty()) {
                // Backward-compatible positional checkpoint argument.
                weights_path = a;
            } else {
                throw std::runtime_error("Unknown argument: " + a);
            }
        }

        if (enable_int8 && enable_int4) throw std::runtime_error("Choose only one of --int8 or --int4");
        MatmulRuntime::set_threads(num_threads);
        MatmulRuntime::set_backend_from_string(backend);

        std::unique_ptr<Tokenizer> demo_tokenizer;
        std::unique_ptr<BPETokenizer> bpe_tokenizer;
        std::unique_ptr<SentencePieceTokenizer> sp_tokenizer;
        std::function<std::vector<int>(const std::string&, bool)> encode;
        std::function<std::string(const std::vector<int>&)> decode;
        int eos_id = 2;
        size_t tokenizer_vocab = 0;

        if (!sp_model_path.empty()) {
            sp_tokenizer = std::make_unique<SentencePieceTokenizer>(sp_model_path);
            tokenizer_vocab = sp_tokenizer->vocab_size();
            eos_id = sp_tokenizer->eos_id();
            encode = [&](const std::string& text, bool bos) { return sp_tokenizer->encode(text, bos); };
            decode = [&](const std::vector<int>& ids) { return sp_tokenizer->decode(ids); };
        } else if (!bpe_vocab_path.empty() || !bpe_merges_path.empty()) {
            if (bpe_vocab_path.empty() || bpe_merges_path.empty()) {
                throw std::runtime_error("Both --bpe-vocab and --bpe-merges are required for BPE tokenization");
            }
            bpe_tokenizer = std::make_unique<BPETokenizer>(bpe_vocab_path, bpe_merges_path);
            tokenizer_vocab = bpe_tokenizer->vocab_size();
            eos_id = bpe_tokenizer->eos_id();
            encode = [&](const std::string& text, bool bos) { return bpe_tokenizer->encode(text, bos); };
            decode = [&](const std::vector<int>& ids) { return bpe_tokenizer->decode(ids); };
        } else {
            demo_tokenizer = std::make_unique<Tokenizer>();
            tokenizer_vocab = demo_tokenizer->vocab_size();
            eos_id = demo_tokenizer->eos_id();
            encode = [&](const std::string& text, bool bos) { return demo_tokenizer->encode(text, bos); };
            decode = [&](const std::vector<int>& ids) { return demo_tokenizer->decode(ids); };
        }

        LLMConfig cfg;
        bool loaded_config = false;
        if (!config_path.empty()) {
            cfg = LLMConfig::from_json_file(config_path);
            loaded_config = true;
            if (tokenizer_vocab != static_cast<size_t>(cfg.vocab_size)) {
                std::cerr << "Warning: tokenizer vocab size (" << tokenizer_vocab
                          << ") differs from config vocab_size (" << cfg.vocab_size << ").\n";
            }
            eos_id = cfg.eos_token_id;
        } else {
            // Tiny demo model dimensions. Random weights mean this demonstrates mechanics, not intelligence.
            cfg.vocab_size = static_cast<int>(tokenizer_vocab);
            cfg.hidden_size = 64;
            cfg.num_attention_heads = 4;
            cfg.num_key_value_heads = 4;
            cfg.intermediate_size = 128;
            cfg.num_hidden_layers = 3;
            cfg.rms_norm_eps = 1e-6;
            cfg.rope_theta = 10000.0;
        }

        TransformerLM model(cfg);

        if (!weights_path.empty()) {
            std::cout << "Loading safetensors checkpoint: " << weights_path << "\n";
            model.load_llama_safetensors(weights_path);
            std::cout << "Checkpoint loaded successfully.\n";
        }

        if (enable_int8) {
            std::cout << "Quantizing linear weights to symmetric per-column INT8 and releasing FP copies...\n";
            model.quantize_all_weights_int8(true);
#if defined(__AVX2__)
            std::cout << "INT8 path enabled with AVX2 dot-product kernels.\n\n";
#else
            std::cout << "INT8 path enabled with scalar fallback. Compile with -mavx2 -mfma for SIMD.\n\n";
#endif
        } else if (enable_int4) {
            std::cout << "Quantizing linear weights to packed symmetric per-column INT4 and releasing FP copies...\n";
            model.quantize_all_weights_int4(true);
            std::cout << "INT4 path enabled. Packed weights use two signed 4-bit values per byte.\n\n";
        } else if (!weights_path.empty()) {
            std::cout << "\n";
        }

        std::vector<int> prompt_ids = encode(prompt, true);

        std::cout << "=== Tokenizer ===\n";
        if (!sp_model_path.empty()) std::cout << "Type: SentencePiece .model greedy loader\n";
        else if (!bpe_vocab_path.empty()) std::cout << "Type: BPE vocab.json + merges.txt\n";
        else std::cout << "Type: built-in tiny demo tokenizer\n";
        std::cout << "Prompt: " << prompt << "\nToken IDs: [ ";
        for (int id : prompt_ids) std::cout << id << " ";
        std::cout << "]\nDecoded: " << decode(prompt_ids) << "\n\n";

        std::cout << "=== Model ===\n";
        std::cout << "Config source: " << (loaded_config ? config_path : "built-in demo config") << "\n";
        std::cout << "Vocab size: " << cfg.vocab_size << "\n";
        std::cout << "Hidden size: " << cfg.hidden_size << "\n";
        std::cout << "Intermediate size: " << cfg.intermediate_size << "\n";
        std::cout << "Decoder layers: " << cfg.num_hidden_layers << "\n";
        std::cout << "Attention heads: " << cfg.num_attention_heads << "\n";
        std::cout << "KV heads: " << cfg.num_key_value_heads << "\n";
        std::cout << "Dense backend: " << MatmulRuntime::backend_name() << "\n";
        std::cout << "Threads: " << MatmulRuntime::threads() << "\n";
        std::cout << "Weight mode: " << (enable_int8 ? "INT8 quantized linear layers" : (enable_int4 ? "INT4 packed quantized linear layers" : "FP64 educational tensors")) << "\n\n";

        GenerationConfig gen_cfg;
        gen_cfg.max_new_tokens = max_new_tokens;
        gen_cfg.temperature = 0.9;
        gen_cfg.top_k = 8;

        std::cout << "=== Autoregressive Generation with KV Cache ===\n";
        std::vector<int> generated = model.generate_cached(prompt_ids, gen_cfg, eos_id);

        std::cout << "Generated IDs: [ ";
        for (int id : generated) std::cout << id << " ";
        std::cout << "]\n";
        std::cout << "Generated text: " << decode(generated) << "\n\n";

        if (weights_path.empty()) {
            std::cout << "Note: no checkpoint was loaded, so weights are random and output is only a pipeline demo.\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
