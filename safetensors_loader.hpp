#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Minimal dependency-free Safetensors reader for inference projects.
//
// Format summary:
//   bytes 0..7              : little-endian uint64 JSON header length
//   bytes 8..8+header_len-1 : UTF-8 JSON metadata
//   remaining bytes         : raw tensor payloads
//
// This loader supports common inference dtypes: F64, F32, F16, BF16, I64, I32, I16, I8, U8.
// It converts loaded values into double so they can be copied into the Tensor<double> used by
// the educational LLM engine.

struct SafeTensorInfo {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    uint64_t begin = 0; // Offset relative to the start of the payload, not start of file.
    uint64_t end = 0;
};

class SafeTensors {
private:
    std::unordered_map<std::string, SafeTensorInfo> tensors_;
    std::vector<uint8_t> payload_;

    // -----------------------------
    // Tiny JSON parser for the subset used by .safetensors headers.
    // -----------------------------
    struct JsonValue {
        enum class Type { Object, Array, String, Number } type;
        std::map<std::string, JsonValue> object;
        std::vector<JsonValue> array;
        std::string string;
        uint64_t number = 0;
    };

    class JsonParser {
    private:
        const std::string& s_;
        size_t p_ = 0;

        void skip_ws() {
            while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\n' || s_[p_] == '\r' || s_[p_] == '\t')) ++p_;
        }

        char peek() {
            skip_ws();
            if (p_ >= s_.size()) throw std::runtime_error("Unexpected end of JSON");
            return s_[p_];
        }

        char get() {
            if (p_ >= s_.size()) throw std::runtime_error("Unexpected end of JSON");
            return s_[p_++];
        }

        void expect(char c) {
            skip_ws();
            if (get() != c) {
                std::ostringstream oss;
                oss << "Expected JSON character '" << c << "'";
                throw std::runtime_error(oss.str());
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
                            // Safetensors tensor names are normally ASCII. Preserve unicode escapes literally.
                            if (p_ + 4 > s_.size()) throw std::runtime_error("Invalid unicode escape in JSON string");
                            out += "\\u";
                            out.append(s_.substr(p_, 4));
                            p_ += 4;
                            break;
                        }
                        default: throw std::runtime_error("Invalid escape sequence in JSON string");
                    }
                } else {
                    out.push_back(c);
                }
            }
            throw std::runtime_error("Unterminated JSON string");
        }

        JsonValue parse_number() {
            skip_ws();
            size_t start = p_;
            while (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
            if (start == p_) throw std::runtime_error("Expected JSON unsigned integer");
            JsonValue v;
            v.type = JsonValue::Type::Number;
            v.number = static_cast<uint64_t>(std::stoull(s_.substr(start, p_ - start)));
            return v;
        }

        JsonValue parse_array() {
            JsonValue v;
            v.type = JsonValue::Type::Array;
            expect('[');
            skip_ws();
            if (peek() == ']') {
                get();
                return v;
            }
            while (true) {
                v.array.push_back(parse_value());
                skip_ws();
                char c = get();
                if (c == ']') return v;
                if (c != ',') throw std::runtime_error("Expected ',' or ']' in JSON array");
            }
        }

        JsonValue parse_object() {
            JsonValue v;
            v.type = JsonValue::Type::Object;
            expect('{');
            skip_ws();
            if (peek() == '}') {
                get();
                return v;
            }
            while (true) {
                std::string key = parse_string_raw();
                expect(':');
                v.object.emplace(std::move(key), parse_value());
                skip_ws();
                char c = get();
                if (c == '}') return v;
                if (c != ',') throw std::runtime_error("Expected ',' or '}' in JSON object");
            }
        }

    public:
        explicit JsonParser(const std::string& s) : s_(s) {}

        JsonValue parse_value() {
            skip_ws();
            char c = peek();
            if (c == '{') return parse_object();
            if (c == '[') return parse_array();
            if (c == '"') {
                JsonValue v;
                v.type = JsonValue::Type::String;
                v.string = parse_string_raw();
                return v;
            }
            if (c >= '0' && c <= '9') return parse_number();
            throw std::runtime_error("Unsupported JSON value in safetensors header");
        }
    };

    static uint64_t read_le_u64(const uint8_t* p) {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
        return v;
    }

    static uint16_t read_le_u16(const uint8_t* p) {
        return static_cast<uint16_t>(p[0]) | static_cast<uint16_t>(p[1] << 8);
    }

    static uint32_t read_le_u32(const uint8_t* p) {
        return static_cast<uint32_t>(p[0]) |
               (static_cast<uint32_t>(p[1]) << 8) |
               (static_cast<uint32_t>(p[2]) << 16) |
               (static_cast<uint32_t>(p[3]) << 24);
    }

    static uint64_t read_le_u64_payload(const uint8_t* p) { return read_le_u64(p); }

    static float read_f32(const uint8_t* p) {
        uint32_t u = read_le_u32(p);
        float f;
        std::memcpy(&f, &u, sizeof(float));
        return f;
    }

    static double read_f64(const uint8_t* p) {
        uint64_t u = read_le_u64_payload(p);
        double d;
        std::memcpy(&d, &u, sizeof(double));
        return d;
    }

    static float fp16_to_float(uint16_t h) {
        uint32_t sign = (h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1Fu;
        uint32_t mant = h & 0x03FFu;
        uint32_t out;

        if (exp == 0) {
            if (mant == 0) {
                out = sign;
            } else {
                exp = 1;
                while ((mant & 0x0400u) == 0) {
                    mant <<= 1;
                    --exp;
                }
                mant &= 0x03FFu;
                uint32_t fexp = exp + (127 - 15);
                out = sign | (fexp << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            out = sign | 0x7F800000u | (mant << 13);
        } else {
            uint32_t fexp = exp + (127 - 15);
            out = sign | (fexp << 23) | (mant << 13);
        }

        float f;
        std::memcpy(&f, &out, sizeof(float));
        return f;
    }

    static float bf16_to_float(uint16_t b) {
        uint32_t u = static_cast<uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(float));
        return f;
    }

    static size_t dtype_size(const std::string& dtype) {
        if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
        if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
        if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
        if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
        throw std::runtime_error("Unsupported safetensors dtype: " + dtype);
    }

    static size_t product(const std::vector<size_t>& shape) {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }

    static const JsonValue& require_key(const JsonValue& obj, const char* key) {
        auto it = obj.object.find(std::string(key));
        if (it == obj.object.end()) throw std::runtime_error(std::string("Safetensors header missing key: ") + key);
        return it->second;
    }

public:
    SafeTensors() = default;

    explicit SafeTensors(const std::string& path) { load(path); }

    void load(const std::string& path) {
        tensors_.clear();
        payload_.clear();

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Could not open safetensors file: " + path);

        in.seekg(0, std::ios::end);
        std::streamoff file_size = in.tellg();
        in.seekg(0, std::ios::beg);
        if (file_size < 8) throw std::runtime_error("Invalid safetensors file: too small");

        std::vector<uint8_t> file(static_cast<size_t>(file_size));
        size_t bytes_to_read = static_cast<size_t>(file_size);
        size_t bytes_read = 0;
        const size_t chunk_size = 1024 * 1024 * 512; // Read in safe 512 MB chunks

        char* ptr = reinterpret_cast<char*>(file.data());
        while (bytes_read < bytes_to_read) {
            size_t chunk = std::min(chunk_size, bytes_to_read - bytes_read);
            in.read(ptr + bytes_read, chunk);
            if (!in) {
                throw std::runtime_error("Failed to read safetensors chunk at offset " + std::to_string(bytes_read));
            }
            bytes_read += chunk;
        }
        
        uint64_t header_len = read_le_u64(file.data());
        if (header_len > static_cast<uint64_t>(file.size() - 8)) {
            throw std::runtime_error("Invalid safetensors file: header length exceeds file size");
        }

        std::string header(reinterpret_cast<const char*>(file.data() + 8), static_cast<size_t>(header_len));
        payload_.assign(file.begin() + 8 + static_cast<std::ptrdiff_t>(header_len), file.end());

        JsonParser parser(header);
        JsonValue root = parser.parse_value();
        if (root.type != JsonValue::Type::Object) throw std::runtime_error("Safetensors header root must be an object");

        for (const auto& kv : root.object) {
            const std::string& name = kv.first;
            if (name == "__metadata__") continue;
            const JsonValue& obj = kv.second;
            if (obj.type != JsonValue::Type::Object) throw std::runtime_error("Tensor header entry must be an object: " + name);

            SafeTensorInfo info;
            info.name = name;

            const JsonValue& dtype = require_key(obj, "dtype");
            if (dtype.type != JsonValue::Type::String) throw std::runtime_error("dtype must be string for tensor: " + name);
            info.dtype = dtype.string;

            const JsonValue& shape = require_key(obj, "shape");
            if (shape.type != JsonValue::Type::Array) throw std::runtime_error("shape must be array for tensor: " + name);
            for (const JsonValue& d : shape.array) {
                if (d.type != JsonValue::Type::Number) throw std::runtime_error("shape dimensions must be integers for tensor: " + name);
                info.shape.push_back(static_cast<size_t>(d.number));
            }

            const JsonValue& offsets = require_key(obj, "data_offsets");
            if (offsets.type != JsonValue::Type::Array || offsets.array.size() != 2) {
                throw std::runtime_error("data_offsets must be [begin,end] for tensor: " + name);
            }
            info.begin = offsets.array[0].number;
            info.end = offsets.array[1].number;

            if (info.end < info.begin || info.end > payload_.size()) {
                throw std::runtime_error("Invalid data_offsets for tensor: " + name);
            }

            size_t expected_bytes = product(info.shape) * dtype_size(info.dtype);
            if (info.end - info.begin != expected_bytes) {
                std::ostringstream oss;
                oss << "Byte-size mismatch for tensor " << name << ": expected " << expected_bytes
                    << ", got " << (info.end - info.begin);
                throw std::runtime_error(oss.str());
            }

            tensors_.emplace(name, std::move(info));
        }
    }

    bool contains(const std::string& name) const { return tensors_.find(name) != tensors_.end(); }

    const SafeTensorInfo& info(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("Tensor not found in safetensors file: " + name);
        return it->second;
    }

    std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(tensors_.size());
        for (const auto& kv : tensors_) out.push_back(kv.first);
        std::sort(out.begin(), out.end());
        return out;
    }

    std::vector<double> tensor_as_double(const std::string& name) const {
        const SafeTensorInfo& t = info(name);
        size_t n = product(t.shape);
        std::vector<double> out(n);
        const uint8_t* p = payload_.data() + t.begin;

        if (t.dtype == "F64") {
            for (size_t i = 0; i < n; ++i) out[i] = read_f64(p + i * 8);
        } else if (t.dtype == "F32") {
            for (size_t i = 0; i < n; ++i) out[i] = read_f32(p + i * 4);
        } else if (t.dtype == "F16") {
            for (size_t i = 0; i < n; ++i) out[i] = fp16_to_float(read_le_u16(p + i * 2));
        } else if (t.dtype == "BF16") {
            for (size_t i = 0; i < n; ++i) out[i] = bf16_to_float(read_le_u16(p + i * 2));
        } else if (t.dtype == "I64") {
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(static_cast<int64_t>(read_le_u64_payload(p + i * 8)));
        } else if (t.dtype == "I32") {
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(static_cast<int32_t>(read_le_u32(p + i * 4)));
        } else if (t.dtype == "I16") {
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(static_cast<int16_t>(read_le_u16(p + i * 2)));
        } else if (t.dtype == "I8") {
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(static_cast<int8_t>(p[i]));
        } else if (t.dtype == "U8" || t.dtype == "BOOL") {
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(p[i]);
        } else {
            throw std::runtime_error("Unsupported dtype conversion: " + t.dtype);
        }
        return out;
    }

    void copy_to(const std::string& name,
                 std::vector<double>& destination,
                 const std::vector<size_t>& expected_shape,
                 bool transpose_2d = false) const {
        const SafeTensorInfo& t = info(name);
        std::vector<double> src = tensor_as_double(name);

        auto shape_product = [](const std::vector<size_t>& s) {
            size_t n = 1;
            for (size_t d : s) n *= d;
            return n;
        };

        if (!transpose_2d) {
            if (t.shape != expected_shape) {
                std::ostringstream oss;
                oss << "Shape mismatch loading " << name << ": file shape [";
                for (size_t i = 0; i < t.shape.size(); ++i) oss << (i ? "," : "") << t.shape[i];
                oss << "] does not match expected [";
                for (size_t i = 0; i < expected_shape.size(); ++i) oss << (i ? "," : "") << expected_shape[i];
                oss << "]";
                throw std::runtime_error(oss.str());
            }
            if (destination.size() != src.size()) destination.resize(src.size());
            destination = std::move(src);
            return;
        }

        if (t.shape.size() != 2 || expected_shape.size() != 2 ||
            t.shape[0] != expected_shape[1] || t.shape[1] != expected_shape[0]) {
            throw std::runtime_error("Transposed 2D shape mismatch loading tensor: " + name);
        }

        size_t rows = expected_shape[0];
        size_t cols = expected_shape[1];
        if (destination.size() != shape_product(expected_shape)) destination.assign(shape_product(expected_shape), 0.0);

        // File is [cols, rows]; destination is [rows, cols].
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                destination[r * cols + c] = src[c * rows + r];
            }
        }
    }
};
