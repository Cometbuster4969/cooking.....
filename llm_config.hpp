#pragma once

#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Minimal dependency-free config.json reader for HuggingFace/LLaMA-like causal LMs.
// It intentionally supports just enough JSON to read ordinary model config files.

class MiniJSON {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<MiniJSON> array;
    std::map<std::string, MiniJSON> object;

    bool has(const std::string& key) const { return object.find(key) != object.end(); }

    const MiniJSON& at(const std::string& key) const {
        auto it = object.find(key);
        if (it == object.end()) throw std::runtime_error("Missing JSON key: " + key);
        return it->second;
    }

    int get_int(const std::string& key, int def) const {
        auto it = object.find(key);
        if (it == object.end() || it->second.type != Type::Number) return def;
        return static_cast<int>(it->second.number);
    }

    double get_number(const std::string& key, double def) const {
        auto it = object.find(key);
        if (it == object.end() || it->second.type != Type::Number) return def;
        return it->second.number;
    }

    std::string get_string(const std::string& key, const std::string& def) const {
        auto it = object.find(key);
        if (it == object.end() || it->second.type != Type::String) return def;
        return it->second.string;
    }

    bool get_bool(const std::string& key, bool def) const {
        auto it = object.find(key);
        if (it == object.end() || it->second.type != Type::Bool) return def;
        return it->second.boolean;
    }
};

class MiniJSONParser {
private:
    const std::string& s_;
    size_t p_ = 0;

    void ws() { while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_; }

    bool consume(const std::string& lit) {
        ws();
        if (s_.compare(p_, lit.size(), lit) == 0) { p_ += lit.size(); return true; }
        return false;
    }

    char peek() { ws(); if (p_ >= s_.size()) throw std::runtime_error("Unexpected JSON EOF"); return s_[p_]; }
    char get() { if (p_ >= s_.size()) throw std::runtime_error("Unexpected JSON EOF"); return s_[p_++]; }
    void expect(char c) { ws(); if (get() != c) throw std::runtime_error(std::string("Expected JSON char: ") + c); }

    static void append_utf8(std::string& out, unsigned code) {
        if (code <= 0x7F) out.push_back(static_cast<char>(code));
        else if (code <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else if (code <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    std::string parse_string_raw() {
        expect('"');
        std::string out;
        while (p_ < s_.size()) {
            char c = get();
            if (c == '"') return out;
            if (c == '\\') {
                char e = get();
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
                        if (p_ + 4 > s_.size()) throw std::runtime_error("Bad unicode escape");
                        unsigned code = static_cast<unsigned>(std::stoul(s_.substr(p_, 4), nullptr, 16));
                        p_ += 4;
                        append_utf8(out, code);
                        break;
                    }
                    default: throw std::runtime_error("Bad JSON escape");
                }
            } else out.push_back(c);
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    MiniJSON parse_number() {
        ws();
        size_t start = p_;
        if (p_ < s_.size() && s_[p_] == '-') ++p_;
        while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        if (p_ < s_.size() && s_[p_] == '.') {
            ++p_;
            while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        }
        if (p_ < s_.size() && (s_[p_] == 'e' || s_[p_] == 'E')) {
            ++p_;
            if (p_ < s_.size() && (s_[p_] == '+' || s_[p_] == '-')) ++p_;
            while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        }
        MiniJSON v; v.type = MiniJSON::Type::Number; v.number = std::stod(s_.substr(start, p_ - start)); return v;
    }

    MiniJSON parse_array() {
        MiniJSON v; v.type = MiniJSON::Type::Array; expect('['); ws();
        if (peek() == ']') { get(); return v; }
        while (true) {
            v.array.push_back(parse_value()); ws(); char c = get();
            if (c == ']') return v;
            if (c != ',') throw std::runtime_error("Expected ',' or ']' in JSON array");
        }
    }

    MiniJSON parse_object() {
        MiniJSON v; v.type = MiniJSON::Type::Object; expect('{'); ws();
        if (peek() == '}') { get(); return v; }
        while (true) {
            std::string key = parse_string_raw(); expect(':'); v.object.emplace(std::move(key), parse_value()); ws(); char c = get();
            if (c == '}') return v;
            if (c != ',') throw std::runtime_error("Expected ',' or '}' in JSON object");
        }
    }

public:
    explicit MiniJSONParser(const std::string& s) : s_(s) {}

    MiniJSON parse_value() {
        ws(); char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') { MiniJSON v; v.type = MiniJSON::Type::String; v.string = parse_string_raw(); return v; }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        if (consume("true")) { MiniJSON v; v.type = MiniJSON::Type::Bool; v.boolean = true; return v; }
        if (consume("false")) { MiniJSON v; v.type = MiniJSON::Type::Bool; v.boolean = false; return v; }
        if (consume("null")) { MiniJSON v; return v; }
        throw std::runtime_error("Unsupported JSON token");
    }
};

struct LLMConfig {
    int vocab_size = 32000;
    int hidden_size = 4096;
    int intermediate_size = 11008;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    int num_key_value_heads = 32;
    int max_position_embeddings = 2048;
    int bos_token_id = 1;
    int eos_token_id = 2;
    double rms_norm_eps = 1e-6;
    double rope_theta = 10000.0;
    bool tie_word_embeddings = false;
    std::string model_type = "llama";

    static LLMConfig from_json_file(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Could not open config.json: " + path);
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        MiniJSON root = MiniJSONParser(s).parse_value();
        if (root.type != MiniJSON::Type::Object) throw std::runtime_error("config.json root must be object");

        LLMConfig c;
        c.vocab_size = root.get_int("vocab_size", c.vocab_size);
        c.hidden_size = root.get_int("hidden_size", root.get_int("n_embd", c.hidden_size));
        c.intermediate_size = root.get_int("intermediate_size", c.intermediate_size);
        c.num_hidden_layers = root.get_int("num_hidden_layers", root.get_int("n_layer", c.num_hidden_layers));
        c.num_attention_heads = root.get_int("num_attention_heads", root.get_int("n_head", c.num_attention_heads));
        c.num_key_value_heads = root.get_int("num_key_value_heads", c.num_attention_heads);
        c.max_position_embeddings = root.get_int("max_position_embeddings", root.get_int("n_positions", c.max_position_embeddings));
        c.bos_token_id = root.get_int("bos_token_id", c.bos_token_id);
        c.eos_token_id = root.get_int("eos_token_id", c.eos_token_id);
        c.rms_norm_eps = root.get_number("rms_norm_eps", root.get_number("layer_norm_epsilon", c.rms_norm_eps));
        c.rope_theta = root.get_number("rope_theta", c.rope_theta);
        c.tie_word_embeddings = root.get_bool("tie_word_embeddings", c.tie_word_embeddings);
        c.model_type = root.get_string("model_type", c.model_type);

        if (c.hidden_size <= 0 || c.num_attention_heads <= 0 || c.num_hidden_layers <= 0 || c.vocab_size <= 0) {
            throw std::runtime_error("Invalid model dimensions in config.json");
        }
        if (c.hidden_size % c.num_attention_heads != 0) {
            throw std::runtime_error("hidden_size must be divisible by num_attention_heads");
        }
        if (c.num_attention_heads % c.num_key_value_heads != 0) {
            throw std::runtime_error("num_attention_heads must be divisible by num_key_value_heads");
        }
        return c;
    }
};
