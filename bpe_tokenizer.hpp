#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// A compact dependency-free BPE tokenizer suitable for educational inference engines.
//
// Supported files:
//   1. vocab.json: a JSON object mapping token string -> integer id
//      Example: {"<unk>":0,"<bos>":1,"<eos>":2,"▁":3,"u":4,"n":5,"un":6}
//   2. merges.txt: one merge per line, highest priority first
//      Example:
//        #version: 0.2
//        u n
//        un believable
//
// Notes:
//   - This is BPE, not a full Google SentencePiece protobuf parser. Many LLaMA-style
//     tokenizers are distributed as tokenizer.model protobuf files. Loading those exactly
//     requires parsing SentencePiece's ModelProto. This class implements the core BPE
//     merge algorithm and can ingest the common vocab.json + merges.txt representation.
//   - It uses the SentencePiece-style word-start marker "▁". If your checkpoint uses GPT-2
//     byte-level BPE with "Ġ" markers, set word_start_marker to "Ġ" or adapt pretokenize().

class BPETokenizer {
private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> merge_rank_;
    std::string word_start_marker_ = "▁";
    int unk_id_ = 0;
    int bos_id_ = 1;
    int eos_id_ = 2;

    static std::string trim(const std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
        return s.substr(a, b - a);
    }

    static std::string parse_json_string(const std::string& s, size_t& p) {
        while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
        if (p >= s.size() || s[p] != '"') throw std::runtime_error("Expected JSON string");
        ++p;
        std::string out;
        while (p < s.size()) {
            char c = s[p++];
            if (c == '"') return out;
            if (c == '\\') {
                if (p >= s.size()) throw std::runtime_error("Bad JSON escape");
                char e = s[p++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        // Minimal Unicode escape support for common tokenizer markers.
                        if (p + 4 > s.size()) throw std::runtime_error("Bad JSON unicode escape");
                        std::string hex = s.substr(p, 4);
                        p += 4;
                        unsigned code = static_cast<unsigned>(std::stoul(hex, nullptr, 16));
                        if (code <= 0x7F) out.push_back(static_cast<char>(code));
                        else if (code <= 0x7FF) {
                            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        } else {
                            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        }
                        break;
                    }
                    default: throw std::runtime_error("Unsupported JSON escape");
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    static int parse_json_int(const std::string& s, size_t& p) {
        while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
        bool neg = false;
        if (p < s.size() && s[p] == '-') { neg = true; ++p; }
        if (p >= s.size() || !std::isdigit(static_cast<unsigned char>(s[p]))) throw std::runtime_error("Expected JSON integer");
        int v = 0;
        while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) {
            v = v * 10 + (s[p++] - '0');
        }
        return neg ? -v : v;
    }

    static std::string pair_key(const std::string& a, const std::string& b) {
        return a + "\x01" + b;
    }

    static std::vector<std::string> utf8_chars(const std::string& s) {
        std::vector<std::string> out;
        for (size_t i = 0; i < s.size();) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            if (i + len > s.size()) len = 1;
            out.push_back(s.substr(i, len));
            i += len;
        }
        return out;
    }

    std::vector<std::string> bpe_merge(std::vector<std::string> pieces) const {
        if (pieces.size() < 2) return pieces;

        while (pieces.size() >= 2) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_i = pieces.size();

            for (size_t i = 0; i + 1 < pieces.size(); ++i) {
                auto it = merge_rank_.find(pair_key(pieces[i], pieces[i + 1]));
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }

            if (best_i == pieces.size()) break;

            pieces[best_i] += pieces[best_i + 1];
            pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best_i + 1));
        }
        return pieces;
    }

    std::vector<std::string> pretokenize(const std::string& text) const {
        std::vector<std::string> words;
        std::string cur;
        bool at_word_start = true;

        auto flush = [&]() {
            if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        };

        for (char ch : text) {
            unsigned char c = static_cast<unsigned char>(ch);
            if (std::isspace(c)) {
                flush();
                at_word_start = true;
            } else if (std::isalnum(c) || c == '_' || c == '-') {
                if (cur.empty() && at_word_start) cur += word_start_marker_;
                cur.push_back(ch);
                at_word_start = false;
            } else {
                flush();
                std::string p;
                if (at_word_start) p += word_start_marker_;
                p.push_back(ch);
                words.push_back(p);
                at_word_start = false;
            }
        }
        flush();
        return words;
    }

public:
    BPETokenizer() = default;

    BPETokenizer(const std::string& vocab_json, const std::string& merges_txt, std::string word_start_marker = "▁") {
        load(vocab_json, merges_txt, std::move(word_start_marker));
    }

    void load(const std::string& vocab_json, const std::string& merges_txt, std::string word_start_marker = "▁") {
        word_start_marker_ = std::move(word_start_marker);
        token_to_id_.clear();
        id_to_token_.clear();
        merge_rank_.clear();

        std::ifstream vf(vocab_json);
        if (!vf) throw std::runtime_error("Could not open vocab JSON: " + vocab_json);
        std::string json((std::istreambuf_iterator<char>(vf)), std::istreambuf_iterator<char>());

        size_t p = 0;
        while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
        if (p >= json.size() || json[p] != '{') throw std::runtime_error("vocab.json must be a JSON object");
        ++p;

        int max_id = -1;
        while (true) {
            while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
            if (p < json.size() && json[p] == '}') { ++p; break; }
            std::string tok = parse_json_string(json, p);
            while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
            if (p >= json.size() || json[p] != ':') throw std::runtime_error("Expected ':' in vocab.json");
            ++p;
            int id = parse_json_int(json, p);
            token_to_id_[tok] = id;
            max_id = std::max(max_id, id);
            while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
            if (p < json.size() && json[p] == ',') { ++p; continue; }
            if (p < json.size() && json[p] == '}') { ++p; break; }
            throw std::runtime_error("Expected ',' or '}' in vocab.json");
        }

        id_to_token_.assign(static_cast<size_t>(max_id + 1), "<unk>");
        for (const auto& kv : token_to_id_) {
            if (kv.second >= 0) id_to_token_[static_cast<size_t>(kv.second)] = kv.first;
        }
        if (token_to_id_.count("<unk>")) unk_id_ = token_to_id_["<unk>"];
        if (token_to_id_.count("<bos>")) bos_id_ = token_to_id_["<bos>"];
        if (token_to_id_.count("<eos>")) eos_id_ = token_to_id_["<eos>"];

        std::ifstream mf(merges_txt);
        if (!mf) throw std::runtime_error("Could not open merges file: " + merges_txt);
        std::string line;
        int rank = 0;
        while (std::getline(mf, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string a, b;
            if (!(iss >> a >> b)) continue;
            merge_rank_[pair_key(a, b)] = rank++;
        }
    }

    int unk_id() const { return unk_id_; }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    size_t vocab_size() const { return id_to_token_.size(); }

    std::vector<int> encode(const std::string& text, bool add_bos = true) const {
        if (id_to_token_.empty()) throw std::runtime_error("BPETokenizer has no vocabulary loaded");
        std::vector<int> ids;
        if (add_bos) ids.push_back(bos_id_);

        for (const std::string& word : pretokenize(text)) {
            std::vector<std::string> pieces = utf8_chars(word);
            pieces = bpe_merge(std::move(pieces));
            for (const std::string& piece : pieces) {
                auto it = token_to_id_.find(piece);
                ids.push_back(it == token_to_id_.end() ? unk_id_ : it->second);
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids, bool skip_special = true) const {
        std::string out;
        for (int id : ids) {
            if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) id = unk_id_;
            std::string tok = id_to_token_[static_cast<size_t>(id)];
            if (skip_special && (tok == "<unk>" || tok == "<bos>" || tok == "<eos>" || tok == "<pad>")) continue;
            size_t pos = 0;
            while ((pos = tok.find(word_start_marker_, pos)) != std::string::npos) {
                tok.replace(pos, word_start_marker_.size(), " ");
                ++pos;
            }
            out += tok;
        }
        if (!out.empty() && out[0] == ' ') out.erase(out.begin());
        return out;
    }
};
