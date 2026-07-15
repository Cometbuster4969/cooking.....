#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <stdexcept>

// Flat tensor representation with row-major layout.
struct Tensor {
    std::vector<size_t> shape;
    std::vector<double> data;

    explicit Tensor(std::vector<size_t> s, bool init_xavier = false) : shape(std::move(s)) {
        if (shape.empty()) {
            throw std::invalid_argument("Tensor shape cannot be empty");
        }

        size_t total_size = 1;
        for (size_t dim : shape) {
            if (dim == 0) throw std::invalid_argument("Tensor dimensions must be non-zero");
            total_size *= dim;
        }
        data.assign(total_size, 0.0);

        if (init_xavier) {
            size_t fan_in = 1;
            size_t fan_out = 1;

            if (shape.size() >= 2) {
                fan_in = shape[0];
                fan_out = shape[1];
            } else {
                fan_in = shape[0];
                fan_out = shape[0];
            }

            static std::mt19937 gen(1337);
            double limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
            std::uniform_real_distribution<double> dist(-limit, limit);
            for (auto& val : data) val = dist(gen);
        }
    }

    double& at(size_t i) { return data.at(i); }
    const double& at(size_t i) const { return data.at(i); }

    size_t dim(size_t i) const {
        if (i >= shape.size()) throw std::out_of_range("Tensor dimension index out of range");
        return shape[i];
    }
};

class ModernLLMEngine {
public:
    // Root Mean Square Normalization.
    static void rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, double eps = 1e-6) {
        if (x.shape.size() != 2 || out.shape.size() != 2 || weight.shape.size() != 1) {
            throw std::invalid_argument("rms_norm expects x/out as 2D tensors and weight as 1D tensor");
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

    // SwiGLU activation: SiLU(gate) * up.
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

    // Matrix multiplication: C = A * B, where A is [M,K], B is [K,N], C is [M,N].
    static void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
        if (A.shape.size() != 2 || B.shape.size() != 2 || C.shape.size() != 2) {
            throw std::invalid_argument("matmul expects 2D tensors");
        }

        size_t M = A.shape[0];
        size_t K = A.shape[1];
        size_t Kb = B.shape[0];
        size_t N = B.shape[1];

        if (K != Kb || C.shape[0] != M || C.shape[1] != N) {
            throw std::invalid_argument("matmul shape mismatch");
        }

        std::fill(C.data.begin(), C.data.end(), 0.0);

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                double a_val = A.data[i * K + k];
                for (size_t j = 0; j < N; ++j) {
                    C.data[i * N + j] += a_val * B.data[k * N + j];
                }
            }
        }
    }

    // Applies Rotary Position Embeddings to a [seq_len, num_heads * head_dim] tensor.
    static void apply_rope(Tensor& t, size_t num_heads, size_t head_dim) {
        if (t.shape.size() != 2) throw std::invalid_argument("apply_rope expects a 2D tensor");
        if (head_dim % 2 != 0) throw std::invalid_argument("RoPE requires even head_dim");
        if (t.shape[1] != num_heads * head_dim) throw std::invalid_argument("apply_rope shape mismatch");

        size_t seq_len = t.shape[0];

        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t h = 0; h < num_heads; ++h) {
                size_t head_offset = i * num_heads * head_dim + h * head_dim;

                for (size_t d = 0; d < head_dim; d += 2) {
                    double theta = 1.0 / std::pow(10000.0, static_cast<double>(d) / static_cast<double>(head_dim));
                    double angle = static_cast<double>(i) * theta;
                    double cos_a = std::cos(angle);
                    double sin_a = std::sin(angle);

                    double x0 = t.data[head_offset + d];
                    double x1 = t.data[head_offset + d + 1];

                    t.data[head_offset + d]     = x0 * cos_a - x1 * sin_a;
                    t.data[head_offset + d + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
    }
};

// LLaMA/Gemma-style Transformer decoder layer.
class DecoderLayer {
private:
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;
    size_t hidden_dim;

    Tensor rms_attn_w;
    Tensor rms_ffn_w;
    Tensor Wq, Wk, Wv, Wo;
    Tensor W_gate, W_up, W_down;

public:
    DecoderLayer(size_t embed, size_t heads, size_t hidden)
        : embed_dim(embed),
          num_heads(heads),
          head_dim(embed / heads),
          hidden_dim(hidden),
          rms_attn_w({embed}),
          rms_ffn_w({embed}),
          Wq({embed, embed}, true),
          Wk({embed, embed}, true),
          Wv({embed, embed}, true),
          Wo({embed, embed}, true),
          W_gate({embed, hidden}, true),
          W_up({embed, hidden}, true),
          W_down({hidden, embed}, true) {

        if (heads == 0 || embed % heads != 0) {
            throw std::invalid_argument("embed_dim must be divisible by num_heads");
        }
        if (head_dim % 2 != 0) {
            throw std::invalid_argument("head_dim must be even for RoPE");
        }

        std::fill(rms_attn_w.data.begin(), rms_attn_w.data.end(), 1.0);
        std::fill(rms_ffn_w.data.begin(), rms_ffn_w.data.end(), 1.0);
    }

    Tensor forward(const Tensor& X) {
        if (X.shape.size() != 2 || X.shape[1] != embed_dim) {
            throw std::invalid_argument("DecoderLayer::forward expects X with shape [seq_len, embed_dim]");
        }

        size_t seq_len = X.shape[0];
        Tensor norm_X({seq_len, embed_dim});

        // 1. Pre-attention RMSNorm.
        ModernLLMEngine::rms_norm(X, rms_attn_w, norm_X);

        // 2. QKV projections.
        Tensor Q({seq_len, embed_dim}), K({seq_len, embed_dim}), V({seq_len, embed_dim});
        ModernLLMEngine::matmul(norm_X, Wq, Q);
        ModernLLMEngine::matmul(norm_X, Wk, K);
        ModernLLMEngine::matmul(norm_X, Wv, V);

        // 3. RoPE on Q and K.
        ModernLLMEngine::apply_rope(Q, num_heads, head_dim);
        ModernLLMEngine::apply_rope(K, num_heads, head_dim);

        // 4. Multi-head causal self-attention.
        Tensor Attn_Out({seq_len, embed_dim});
        double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

        for (size_t h = 0; h < num_heads; ++h) {
            Tensor Scores({seq_len, seq_len});

            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    double score = 0.0;
                    size_t q_off = i * num_heads * head_dim + h * head_dim;
                    size_t k_off = j * num_heads * head_dim + h * head_dim;

                    for (size_t d = 0; d < head_dim; ++d) {
                        score += Q.data[q_off + d] * K.data[k_off + d];
                    }
                    Scores.data[i * seq_len + j] = score * scale;
                }

                for (size_t j = i + 1; j < seq_len; ++j) {
                    Scores.data[i * seq_len + j] = -1e9;
                }
            }

            // Stable row-wise softmax.
            for (size_t i = 0; i < seq_len; ++i) {
                double max_val = Scores.data[i * seq_len];
                for (size_t j = 1; j < seq_len; ++j) {
                    max_val = std::max(max_val, Scores.data[i * seq_len + j]);
                }

                double sum = 0.0;
                for (size_t j = 0; j < seq_len; ++j) {
                    Scores.data[i * seq_len + j] = std::exp(Scores.data[i * seq_len + j] - max_val);
                    sum += Scores.data[i * seq_len + j];
                }

                for (size_t j = 0; j < seq_len; ++j) {
                    Scores.data[i * seq_len + j] /= sum;
                }
            }

            // Attention-weighted value aggregation.
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < head_dim; ++d) {
                    double val_sum = 0.0;
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t v_off = j * num_heads * head_dim + h * head_dim;
                        val_sum += Scores.data[i * seq_len + j] * V.data[v_off + d];
                    }
                    Attn_Out.data[i * embed_dim + h * head_dim + d] = val_sum;
                }
            }
        }

        // 5. Output projection and residual connection.
        Tensor Projected_Attn({seq_len, embed_dim});
        ModernLLMEngine::matmul(Attn_Out, Wo, Projected_Attn);

        Tensor Mid_X({seq_len, embed_dim});
        for (size_t i = 0; i < X.data.size(); ++i) {
            Mid_X.data[i] = X.data[i] + Projected_Attn.data[i];
        }

        // 6. Pre-FFN RMSNorm.
        Tensor norm_Mid({seq_len, embed_dim});
        ModernLLMEngine::rms_norm(Mid_X, rms_ffn_w, norm_Mid);

        // 7. SwiGLU FFN.
        Tensor Gate_Out({seq_len, hidden_dim}), Up_Out({seq_len, hidden_dim});
        ModernLLMEngine::matmul(norm_Mid, W_gate, Gate_Out);
        ModernLLMEngine::matmul(norm_Mid, W_up, Up_Out);

        Tensor SwiGLU_Out({seq_len, hidden_dim});
        ModernLLMEngine::swiglu(Gate_Out, Up_Out, SwiGLU_Out);

        Tensor FFN_Out({seq_len, embed_dim});
        ModernLLMEngine::matmul(SwiGLU_Out, W_down, FFN_Out);

        // Final residual connection.
        Tensor Final_Out({seq_len, embed_dim});
        for (size_t i = 0; i < Mid_X.data.size(); ++i) {
            Final_Out.data[i] = Mid_X.data[i] + FFN_Out.data[i];
        }

        return Final_Out;
    }
};

int main() {
    try {
        std::cout << "RUNNING LLM DECODER BLOCK (LLaMA-STYLE ARCHITECTURE)\n\n";

        size_t seq_len = 4;
        size_t embed_dim = 64;
        size_t num_heads = 4;
        size_t hidden_dim = 128;

        Tensor InputTokens({seq_len, embed_dim}, true);
        DecoderLayer llama_layer(embed_dim, num_heads, hidden_dim);

        std::cout << "Executing forward pass through RMSNorm, attention, RoPE, and SwiGLU...\n";
        Tensor Output = llama_layer.forward(InputTokens);

        std::cout << "\nExecution successful. Output tensor shape: ["
                  << Output.shape[0] << " tokens x "
                  << Output.shape[1] << " feature channels]\n";

        std::cout << "Top features of final sequence state:\n[ ";
        for (size_t i = 0; i < 6 && i < Output.data.size(); ++i) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << Output.data[i] << " ";
        }
        std::cout << "... ]\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
