#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Minimal SentencePiece .model vocabulary reader + greedy longest-match encoder.
//
// This closes the practical file-format gap for many LLaMA-style tokenizer.model files:
// it can read the protobuf ModelProto enough to extract repeated SentencePiece records
// (field 1 of ModelProto, fields 1=piece string, 2=score float, 3=type enum inside each piece).
//
// Important: official SentencePiece encoding can use Unigram dynamic programming or BPE rules
// depending on trainer_spec.model_type. This small implementation is intentionally inference-embed
// friendly: it loads real .model vocabularies, then encodes by greedy longest-match over pieces.
// That is useful and deterministic, but not byte-for-byte identical to libsentencepiece for all models.

class SentencePieceTokenizer {
private:
    struct Piece { std::string text; float score = 0.0f; int type = 1; };
    std::vector<Piece> pieces_;
    std::unordered_map<std::string, int> id_;
    size_t max_piece_bytes_ = 0;
    int unk_id_ = 0, bos_id_ = 1, eos_id_ = 2, pad_id_ = -1;
    std::string word_marker_ = "▁";

    static uint64_t read_varint(const std::vector<uint8_t>& b, size_t& p, size_t end) {
        uint64_t v = 0; int shift = 0;
        while (p < end) {
            uint8_t c = b[p++];
            v |= static_cast<uint64_t>(c & 0x7F) << shift;
            if (!(c & 0x80)) return v;
            shift += 7;
            if (shift > 63) throw std::runtime_error("Invalid protobuf varint");
        }
        throw std::runtime_error("Truncated protobuf varint");
    }

    static void skip_field(const std::vector<uint8_t>& b, size_t& p, size_t end, int wire) {
        switch (wire) {
            case 0: (void)read_varint(b, p, end); break;
            case 1: p += 8; break;
            case 2: { uint64_t n = read_varint(b, p, end); p += static_cast<size_t>(n); break; }
            case 5: p += 4; break;
            default: throw std::runtime_error("Unsupported protobuf wire type");
        }
        if (p > end) throw std::runtime_error("Truncated protobuf field");
    }

    static float read_fixed32_float(const std::vector<uint8_t>& b, size_t p) {
        if (p + 4 > b.size()) throw std::runtime_error("Truncated protobuf float");
        uint32_t u = static_cast<uint32_t>(b[p]) |
                     (static_cast<uint32_t>(b[p + 1]) << 8) |
                     (static_cast<uint32_t>(b[p + 2]) << 16) |
                     (static_cast<uint32_t>(b[p + 3]) << 24);
        float f;
        std::memcpy(&f, &u, sizeof(float));
        return f;
    }

    static Piece parse_piece_msg(const std::vector<uint8_t>& b, size_t p, size_t end) {
        Piece piece;
        while (p < end) {
            uint64_t key = read_varint(b, p, end);
            int field = static_cast<int>(key >> 3);
            int wire = static_cast<int>(key & 7);
            if (field == 1 && wire == 2) {
                uint64_t n = read_varint(b, p, end);
                if (p + n > end) throw std::runtime_error("Truncated piece string");
                piece.text.assign(reinterpret_cast<const char*>(b.data() + p), static_cast<size_t>(n));
                p += static_cast<size_t>(n);
            } else if (field == 2 && wire == 5) {
                piece.score = read_fixed32_float(b, p);
                p += 4;
            } else if (field == 3 && wire == 0) {
                piece.type = static_cast<int>(read_varint(b, p, end));
            } else {
                skip_field(b, p, end, wire);
            }
        }
        return piece;
    }

    std::vector<std::string> pretokenize(const std::string& text) const {
        std::vector<std::string> out;
        std::string cur;
        bool new_word = true;
        auto flush = [&]() { if (!cur.empty()) { out.push_back(cur); cur.clear(); } };
        for (char ch : text) {
            unsigned char c = static_cast<unsigned char>(ch);
            if (std::isspace(c)) { flush(); new_word = true; }
            else {
                if (cur.empty() && new_word) cur += word_marker_;
                cur.push_back(ch);
                new_word = false;
            }
        }
        flush();
        return out;
    }

public:
    SentencePieceTokenizer() = default;
    explicit SentencePieceTokenizer(const std::string& model_path) { load(model_path); }

    void load(const std::string& model_path) {
        pieces_.clear(); id_.clear(); max_piece_bytes_ = 0;
        std::ifstream f(model_path, std::ios::binary);
        if (!f) throw std::runtime_error("Could not open tokenizer.model: " + model_path);
        std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        size_t p = 0, end = b.size();
        while (p < end) {
            uint64_t key = read_varint(b, p, end);
            int field = static_cast<int>(key >> 3), wire = static_cast<int>(key & 7);
            if (field == 1 && wire == 2) { // repeated SentencePiece pieces = 1;
                uint64_t n = read_varint(b, p, end);
                size_t msg_end = p + static_cast<size_t>(n);
                if (msg_end > end) throw std::runtime_error("Truncated SentencePiece ModelProto");
                Piece piece = parse_piece_msg(b, p, msg_end);
                int idx = static_cast<int>(pieces_.size());
                if (!piece.text.empty()) {
                    id_[piece.text] = idx;
                    max_piece_bytes_ = std::max(max_piece_bytes_, piece.text.size());
                }
                pieces_.push_back(std::move(piece));
                p = msg_end;
            } else {
                skip_field(b, p, end, wire);
            }
        }
        if (pieces_.empty()) throw std::runtime_error("No pieces found in tokenizer.model");
        if (id_.count("<unk>")) unk_id_ = id_["<unk>"];
        if (id_.count("<s>")) bos_id_ = id_["<s>"];
        if (id_.count("</s>")) eos_id_ = id_["</s>"];
        if (id_.count("<pad>")) pad_id_ = id_["<pad>"];
    }

    int unk_id() const { return unk_id_; }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    size_t vocab_size() const { return pieces_.size(); }

    std::vector<int> encode(const std::string& text, bool add_bos = true) const {
        if (pieces_.empty()) throw std::runtime_error("SentencePieceTokenizer is not loaded");
        std::vector<int> ids;
        if (add_bos && bos_id_ >= 0) ids.push_back(bos_id_);
        for (const std::string& word : pretokenize(text)) {
            size_t p = 0;
            while (p < word.size()) {
                int best_id = -1;
                size_t best_len = 0;
                size_t max_len = std::min(max_piece_bytes_, word.size() - p);
                for (size_t len = max_len; len > 0; --len) {
                    auto it = id_.find(word.substr(p, len));
                    if (it != id_.end()) { best_id = it->second; best_len = len; break; }
                }
                if (best_id >= 0) { ids.push_back(best_id); p += best_len; }
                else { ids.push_back(unk_id_); ++p; }
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids, bool skip_special = true) const {
        std::string out;
        for (int id : ids) {
            if (id < 0 || static_cast<size_t>(id) >= pieces_.size()) id = unk_id_;
            const std::string& tok = pieces_[static_cast<size_t>(id)].text;
            if (skip_special && (tok == "<unk>" || tok == "<s>" || tok == "</s>" || tok == "<pad>")) continue;
            out += tok;
        }
        size_t pos = 0;
        while ((pos = out.find(word_marker_, pos)) != std::string::npos) {
            out.replace(pos, word_marker_.size(), " ");
            ++pos;
        }
        if (!out.empty() && out[0] == ' ') out.erase(out.begin());
        return out;
    }
};
