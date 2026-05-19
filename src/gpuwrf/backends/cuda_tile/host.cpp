#include <cuda_runtime.h>
#include <zlib.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" cudaError_t launch_stencil_advdiff(
    const double* phi_initial,
    const float* u_face,
    const float* v_face,
    const float* w_face,
    double* phi_next,
    int nx,
    int ny,
    int nz,
    cudaStream_t stream);

extern "C" cudaError_t launch_column_thermo(
    const double* temperature_initial,
    const double* qv_initial,
    const double* pressure_initial,
    const double* saturation_qv,
    double* temperature_next,
    double* qv_next,
    double* pressure_next,
    double* mse_delta,
    int levels,
    cudaStream_t stream);

extern "C" cudaError_t stencil_theoretical_occupancy_pct(double* occupancy_pct, int nx, int ny);
extern "C" cudaError_t column_theoretical_occupancy_pct(double* occupancy_pct, int levels);

namespace {

struct Array {
    std::string descr;
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> bytes;

    std::size_t count() const {
        return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<std::size_t>());
    }
};

using ArrayMap = std::map<std::string, Array>;

[[noreturn]] void die_cuda(cudaError_t err, const std::string& context) {
    std::ostringstream oss;
    oss << context << ": " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

void check_cuda(cudaError_t err, const std::string& context) {
    if (err != cudaSuccess) {
        die_cuda(err, context);
    }
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open input file: " + path);
    }
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const std::vector<std::uint8_t>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot open output file: " + path);
    }
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

std::uint16_t le16(const std::vector<std::uint8_t>& data, std::size_t off) {
    return static_cast<std::uint16_t>(data.at(off)) | (static_cast<std::uint16_t>(data.at(off + 1)) << 8);
}

std::uint32_t le32(const std::vector<std::uint8_t>& data, std::size_t off) {
    return static_cast<std::uint32_t>(data.at(off)) | (static_cast<std::uint32_t>(data.at(off + 1)) << 8) |
           (static_cast<std::uint32_t>(data.at(off + 2)) << 16) | (static_cast<std::uint32_t>(data.at(off + 3)) << 24);
}

void put16(std::vector<std::uint8_t>& out, std::uint16_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xff));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
}

void put32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xff));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<std::uint8_t>((value >> 16) & 0xff));
    out.push_back(static_cast<std::uint8_t>((value >> 24) & 0xff));
}

std::vector<std::uint8_t> inflate_raw(const std::uint8_t* input, std::size_t compressed_size, std::size_t output_size) {
    std::vector<std::uint8_t> output(output_size);
    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input));
    stream.avail_in = static_cast<uInt>(compressed_size);
    stream.next_out = reinterpret_cast<Bytef*>(output.data());
    stream.avail_out = static_cast<uInt>(output.size());
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) {
        throw std::runtime_error("inflateInit2 failed");
    }
    const int rc = inflate(&stream, Z_FINISH);
    inflateEnd(&stream);
    if (rc != Z_STREAM_END || stream.total_out != output_size) {
        throw std::runtime_error("zip deflate decode failed");
    }
    return output;
}

std::string strip_npy_suffix(std::string name) {
    const std::string suffix = ".npy";
    if (name.size() >= suffix.size() && name.substr(name.size() - suffix.size()) == suffix) {
        name.resize(name.size() - suffix.size());
    }
    return name;
}

std::vector<std::string> split_shape(const std::string& text) {
    std::vector<std::string> parts;
    std::string current;
    for (char ch : text) {
        if (ch == ',') {
            parts.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        parts.push_back(current);
    }
    return parts;
}

Array parse_npy(const std::vector<std::uint8_t>& npy) {
    if (npy.size() < 16 || std::memcmp(npy.data(), "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy magic");
    }
    const int major = npy[6];
    std::size_t header_len = 0;
    std::size_t header_off = 0;
    if (major == 1) {
        header_len = le16(npy, 8);
        header_off = 10;
    } else if (major == 2 || major == 3) {
        header_len = le32(npy, 8);
        header_off = 12;
    } else {
        throw std::runtime_error("unsupported npy version");
    }
    if (header_off + header_len > npy.size()) {
        throw std::runtime_error("truncated npy header");
    }
    const std::string header(reinterpret_cast<const char*>(npy.data() + header_off), header_len);

    const auto descr_key = header.find("'descr'");
    const auto fortran_key = header.find("'fortran_order'");
    const auto shape_key = header.find("'shape'");
    if (descr_key == std::string::npos || fortran_key == std::string::npos || shape_key == std::string::npos) {
        throw std::runtime_error("npy header missing required keys");
    }
    const auto descr_first = header.find('\'', header.find(':', descr_key) + 1);
    const auto descr_second = header.find('\'', descr_first + 1);
    std::string descr = header.substr(descr_first + 1, descr_second - descr_first - 1);
    if (header.find("False", fortran_key) == std::string::npos) {
        throw std::runtime_error("fortran_order arrays are unsupported");
    }
    const auto lparen = header.find('(', shape_key);
    const auto rparen = header.find(')', lparen);
    const std::string shape_text = header.substr(lparen + 1, rparen - lparen - 1);
    std::vector<std::size_t> shape;
    for (std::string part : split_shape(shape_text)) {
        part.erase(std::remove_if(part.begin(), part.end(), [](unsigned char c) { return std::isspace(c); }), part.end());
        if (!part.empty()) {
            shape.push_back(static_cast<std::size_t>(std::stoull(part)));
        }
    }
    const std::size_t data_off = header_off + header_len;
    std::size_t item_size = 0;
    if (descr == "<f8" || descr == "|f8") {
        item_size = 8;
    } else if (descr == "<f4" || descr == "|f4") {
        item_size = 4;
    } else {
        throw std::runtime_error("unsupported dtype: " + descr);
    }
    const std::size_t nbytes = std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<std::size_t>()) * item_size;
    if (data_off + nbytes > npy.size()) {
        throw std::runtime_error("truncated npy data");
    }
    Array array{descr, shape, {}};
    array.bytes.assign(npy.begin() + static_cast<std::ptrdiff_t>(data_off), npy.begin() + static_cast<std::ptrdiff_t>(data_off + nbytes));
    return array;
}

ArrayMap read_npz(const std::string& path) {
    const std::vector<std::uint8_t> zip = read_file(path);
    if (zip.size() < 22) {
        throw std::runtime_error("zip too small");
    }
    std::size_t eocd = std::string::npos;
    for (std::size_t pos = zip.size() - 22;; --pos) {
        if (le32(zip, pos) == 0x06054b50u) {
            eocd = pos;
            break;
        }
        if (pos == 0) {
            break;
        }
    }
    if (eocd == std::string::npos) {
        throw std::runtime_error("zip end-of-central-directory not found");
    }
    const std::uint16_t entries = le16(zip, eocd + 10);
    std::size_t cd = le32(zip, eocd + 16);
    ArrayMap arrays;
    for (std::uint16_t i = 0; i < entries; ++i) {
        if (le32(zip, cd) != 0x02014b50u) {
            throw std::runtime_error("bad central directory signature");
        }
        const std::uint16_t method = le16(zip, cd + 10);
        const std::uint32_t compressed_size = le32(zip, cd + 20);
        const std::uint32_t uncompressed_size = le32(zip, cd + 24);
        const std::uint16_t name_len = le16(zip, cd + 28);
        const std::uint16_t extra_len = le16(zip, cd + 30);
        const std::uint16_t comment_len = le16(zip, cd + 32);
        const std::uint32_t local_off = le32(zip, cd + 42);
        const std::string filename(reinterpret_cast<const char*>(zip.data() + cd + 46), name_len);
        if (le32(zip, local_off) != 0x04034b50u) {
            throw std::runtime_error("bad local file signature");
        }
        const std::uint16_t local_name_len = le16(zip, local_off + 26);
        const std::uint16_t local_extra_len = le16(zip, local_off + 28);
        const std::size_t data_off = local_off + 30 + local_name_len + local_extra_len;
        std::vector<std::uint8_t> npy;
        if (method == 0) {
            npy.assign(zip.begin() + static_cast<std::ptrdiff_t>(data_off),
                       zip.begin() + static_cast<std::ptrdiff_t>(data_off + compressed_size));
        } else if (method == 8) {
            npy = inflate_raw(zip.data() + data_off, compressed_size, uncompressed_size);
        } else {
            throw std::runtime_error("unsupported zip compression method");
        }
        arrays[strip_npy_suffix(filename)] = parse_npy(npy);
        cd += 46 + name_len + extra_len + comment_len;
    }
    return arrays;
}

template <typename T>
std::vector<T> as_vector(const ArrayMap& arrays, const std::string& name, const std::vector<std::size_t>& shape) {
    const auto found = arrays.find(name);
    if (found == arrays.end()) {
        throw std::runtime_error("missing array: " + name);
    }
    const Array& arr = found->second;
    if (arr.shape != shape) {
        throw std::runtime_error("shape mismatch for " + name);
    }
    const std::string expected = sizeof(T) == 8 ? "<f8" : "<f4";
    if (arr.descr != expected && arr.descr != ("|f" + std::to_string(sizeof(T)))) {
        throw std::runtime_error("dtype mismatch for " + name);
    }
    std::vector<T> out(arr.count());
    std::memcpy(out.data(), arr.bytes.data(), arr.bytes.size());
    return out;
}

template <typename T>
Array make_array(const std::vector<T>& values, std::vector<std::size_t> shape) {
    Array arr;
    arr.descr = sizeof(T) == 8 ? "<f8" : "<f4";
    arr.shape = std::move(shape);
    arr.bytes.resize(values.size() * sizeof(T));
    std::memcpy(arr.bytes.data(), values.data(), arr.bytes.size());
    return arr;
}

std::vector<std::uint8_t> make_npy(const std::string& descr, const std::vector<std::size_t>& shape, const std::vector<std::uint8_t>& data) {
    std::ostringstream shape_stream;
    shape_stream << "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) {
            shape_stream << ", ";
        }
        shape_stream << shape[i];
    }
    if (shape.size() == 1) {
        shape_stream << ",";
    }
    shape_stream << ")";
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape_stream.str() + ", }";
    const std::size_t prefix = 10;
    const std::size_t padding = 16 - ((prefix + header.size() + 1) % 16);
    header.append(padding, ' ');
    header.push_back('\n');

    std::vector<std::uint8_t> npy;
    const std::uint8_t magic[] = {0x93u, 'N', 'U', 'M', 'P', 'Y', 1u, 0u};
    npy.insert(npy.end(), std::begin(magic), std::end(magic));
    put16(npy, static_cast<std::uint16_t>(header.size()));
    npy.insert(npy.end(), header.begin(), header.end());
    npy.insert(npy.end(), data.begin(), data.end());
    return npy;
}

void write_npz(const std::string& path, const ArrayMap& arrays) {
    struct CentralEntry {
        std::string filename;
        std::uint32_t crc;
        std::uint32_t size;
        std::uint32_t offset;
    };
    std::vector<std::uint8_t> out;
    std::vector<CentralEntry> central;
    for (const auto& [name, arr] : arrays) {
        const std::string filename = name + ".npy";
        const std::vector<std::uint8_t> npy = make_npy(arr.descr, arr.shape, arr.bytes);
        const std::uint32_t crc = crc32(0L, reinterpret_cast<const Bytef*>(npy.data()), static_cast<uInt>(npy.size()));
        const std::uint32_t offset = static_cast<std::uint32_t>(out.size());

        put32(out, 0x04034b50u);
        put16(out, 20);
        put16(out, 0);
        put16(out, 0);
        put16(out, 0);
        put16(out, 0);
        put32(out, crc);
        put32(out, static_cast<std::uint32_t>(npy.size()));
        put32(out, static_cast<std::uint32_t>(npy.size()));
        put16(out, static_cast<std::uint16_t>(filename.size()));
        put16(out, 0);
        out.insert(out.end(), filename.begin(), filename.end());
        out.insert(out.end(), npy.begin(), npy.end());
        central.push_back({filename, crc, static_cast<std::uint32_t>(npy.size()), offset});
    }

    const std::uint32_t cd_offset = static_cast<std::uint32_t>(out.size());
    for (const CentralEntry& entry : central) {
        put32(out, 0x02014b50u);
        put16(out, 20);
        put16(out, 20);
        put16(out, 0);
        put16(out, 0);
        put16(out, 0);
        put16(out, 0);
        put32(out, entry.crc);
        put32(out, entry.size);
        put32(out, entry.size);
        put16(out, static_cast<std::uint16_t>(entry.filename.size()));
        put16(out, 0);
        put16(out, 0);
        put16(out, 0);
        put16(out, 0);
        put32(out, 0);
        put32(out, entry.offset);
        out.insert(out.end(), entry.filename.begin(), entry.filename.end());
    }
    const std::uint32_t cd_size = static_cast<std::uint32_t>(out.size()) - cd_offset;
    put32(out, 0x06054b50u);
    put16(out, 0);
    put16(out, 0);
    put16(out, static_cast<std::uint16_t>(central.size()));
    put16(out, static_cast<std::uint16_t>(central.size()));
    put32(out, cd_size);
    put32(out, cd_offset);
    put16(out, 0);
    write_file(path, out);
}

template <typename T>
T* device_alloc_copy(const std::vector<T>& host, std::size_t& transfer_bytes) {
    T* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, host.size() * sizeof(T)), "cudaMalloc");
    check_cuda(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    transfer_bytes += host.size() * sizeof(T);
    return ptr;
}

template <typename T>
std::vector<T> copy_from_device(T* ptr, std::size_t count, std::size_t& transfer_bytes) {
    std::vector<T> host(count);
    check_cuda(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    transfer_bytes += count * sizeof(T);
    return host;
}

std::string json_escape(const std::string& text) {
    std::ostringstream out;
    for (char ch : text) {
        if (ch == '\\' || ch == '"') {
            out << '\\' << ch;
        } else if (ch == '\n') {
            out << "\\n";
        } else {
            out << ch;
        }
    }
    return out.str();
}

void print_result(const std::string& problem, double kernel_wall_s, std::size_t transfer_bytes, int kernel_launches, double occupancy_pct) {
    std::cout << "{\n"
              << "  \"problem\": \"" << json_escape(problem) << "\",\n"
              << "  \"wall_time_s\": " << std::setprecision(12) << kernel_wall_s << ",\n"
              << "  \"kernel_launches\": " << kernel_launches << ",\n"
              << "  \"host_device_transfer_bytes\": " << transfer_bytes << ",\n"
              << "  \"theoretical_occupancy_pct\": " << std::setprecision(8) << occupancy_pct << "\n"
              << "}\n";
}

void run_stencil(const std::string& input, const std::string& output) {
    constexpr int nx = 32;
    constexpr int ny = 16;
    constexpr int nz = 8;
    const ArrayMap arrays = read_npz(input);
    const auto phi_initial = as_vector<double>(arrays, "phi_initial", {nz, ny, nx});
    const auto u_face = as_vector<float>(arrays, "u_face", {nz, ny, nx + 1});
    const auto v_face = as_vector<float>(arrays, "v_face", {nz, ny + 1, nx});
    const auto w_face = as_vector<float>(arrays, "w_face", {nz + 1, ny, nx});

    std::size_t transfer_bytes = 0;
    double* d_phi_initial = device_alloc_copy(phi_initial, transfer_bytes);
    float* d_u_face = device_alloc_copy(u_face, transfer_bytes);
    float* d_v_face = device_alloc_copy(v_face, transfer_bytes);
    float* d_w_face = device_alloc_copy(w_face, transfer_bytes);
    double* d_phi_next = nullptr;
    check_cuda(cudaMalloc(&d_phi_next, phi_initial.size() * sizeof(double)), "cudaMalloc phi_next");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    check_cuda(launch_stencil_advdiff(d_phi_initial, d_u_face, d_v_face, d_w_face, d_phi_next, nx, ny, nz, nullptr),
               "launch_stencil_advdiff");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

    const auto phi_next = copy_from_device(d_phi_next, phi_initial.size(), transfer_bytes);
    ArrayMap out;
    out["phi_initial"] = make_array(phi_initial, {nz, ny, nx});
    out["phi_next"] = make_array(phi_next, {nz, ny, nx});
    out["u_face"] = make_array(u_face, {nz, ny, nx + 1});
    out["v_face"] = make_array(v_face, {nz, ny + 1, nx});
    out["w_face"] = make_array(w_face, {nz + 1, ny, nx});
    write_npz(output, out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_phi_initial);
    cudaFree(d_u_face);
    cudaFree(d_v_face);
    cudaFree(d_w_face);
    cudaFree(d_phi_next);
    double occupancy_pct = 0.0;
    check_cuda(stencil_theoretical_occupancy_pct(&occupancy_pct, nx, ny), "stencil_theoretical_occupancy_pct");
    print_result("stencil", static_cast<double>(ms) / 1000.0, transfer_bytes, 1, occupancy_pct);
}

void run_column(const std::string& input, const std::string& output) {
    constexpr int levels = 40;
    const ArrayMap arrays = read_npz(input);
    const auto temperature_initial = as_vector<double>(arrays, "temperature_initial", {levels});
    const auto qv_initial = as_vector<double>(arrays, "qv_initial", {levels});
    const auto pressure_initial = as_vector<double>(arrays, "pressure_initial", {levels});
    const auto saturation_qv = as_vector<double>(arrays, "saturation_qv", {levels});

    std::size_t transfer_bytes = 0;
    double* d_temperature_initial = device_alloc_copy(temperature_initial, transfer_bytes);
    double* d_qv_initial = device_alloc_copy(qv_initial, transfer_bytes);
    double* d_pressure_initial = device_alloc_copy(pressure_initial, transfer_bytes);
    double* d_saturation_qv = device_alloc_copy(saturation_qv, transfer_bytes);
    double* d_temperature_next = nullptr;
    double* d_qv_next = nullptr;
    double* d_pressure_next = nullptr;
    double* d_mse_delta = nullptr;
    check_cuda(cudaMalloc(&d_temperature_next, levels * sizeof(double)), "cudaMalloc temperature_next");
    check_cuda(cudaMalloc(&d_qv_next, levels * sizeof(double)), "cudaMalloc qv_next");
    check_cuda(cudaMalloc(&d_pressure_next, levels * sizeof(double)), "cudaMalloc pressure_next");
    check_cuda(cudaMalloc(&d_mse_delta, levels * sizeof(double)), "cudaMalloc mse_delta");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    check_cuda(launch_column_thermo(
                   d_temperature_initial,
                   d_qv_initial,
                   d_pressure_initial,
                   d_saturation_qv,
                   d_temperature_next,
                   d_qv_next,
                   d_pressure_next,
                   d_mse_delta,
                   levels,
                   nullptr),
               "launch_column_thermo");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

    const auto temperature_next = copy_from_device(d_temperature_next, levels, transfer_bytes);
    const auto qv_next = copy_from_device(d_qv_next, levels, transfer_bytes);
    const auto pressure_next = copy_from_device(d_pressure_next, levels, transfer_bytes);
    const auto mse_delta = copy_from_device(d_mse_delta, levels, transfer_bytes);
    ArrayMap out;
    out["mse_delta"] = make_array(mse_delta, {levels});
    out["pressure_initial"] = make_array(pressure_initial, {levels});
    out["pressure_next"] = make_array(pressure_next, {levels});
    out["qv_initial"] = make_array(qv_initial, {levels});
    out["qv_next"] = make_array(qv_next, {levels});
    out["saturation_qv"] = make_array(saturation_qv, {levels});
    out["temperature_initial"] = make_array(temperature_initial, {levels});
    out["temperature_next"] = make_array(temperature_next, {levels});
    write_npz(output, out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_temperature_initial);
    cudaFree(d_qv_initial);
    cudaFree(d_pressure_initial);
    cudaFree(d_saturation_qv);
    cudaFree(d_temperature_next);
    cudaFree(d_qv_next);
    cudaFree(d_pressure_next);
    cudaFree(d_mse_delta);
    double occupancy_pct = 0.0;
    check_cuda(column_theoretical_occupancy_pct(&occupancy_pct, levels), "column_theoretical_occupancy_pct");
    print_result("column", static_cast<double>(ms) / 1000.0, transfer_bytes, 1, occupancy_pct);
}

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    if (argc < 2) {
        throw std::runtime_error("usage: bench stencil|column --input path --output path");
    }
    args["problem"] = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::runtime_error("expected --key value argument");
        }
        args[key.substr(2)] = argv[++i];
    }
    if (!args.count("input") || !args.count("output")) {
        throw std::runtime_error("--input and --output are required");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        if (args.at("problem") == "stencil") {
            run_stencil(args.at("input"), args.at("output"));
        } else if (args.at("problem") == "column") {
            run_column(args.at("input"), args.at("output"));
        } else {
            throw std::runtime_error("unknown problem: " + args.at("problem"));
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "cuda_tile bench: " << exc.what() << "\n";
        return 1;
    }
}
