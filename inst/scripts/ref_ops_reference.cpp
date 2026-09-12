// Reference outputs for single-op models, produced by ONNX Runtime.
//
// Companion to ref_ops_vs_onnxruntime.R: that script writes one .onnx per
// operation plus its input, this runs the same files through ONNX Runtime and
// writes the outputs beside them for comparison.
//
// Every output is written, not just the first: the ops under test here are the
// ones that return more than one (TopK gives values and indices, and the
// indices are the half where a tie convention shows).
//
// Build is handled by the driver script; standalone it is:
//   g++ -O2 -std=c++17 ref_ops_reference.cpp -o ort_ops \
//       -I$ORT_DIR/include -L$ORT_DIR/lib -lonnxruntime -Wl,-rpath,$ORT_DIR/lib

#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<float> read_f32(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> v(static_cast<size_t>(n) / sizeof(float));
    f.read(reinterpret_cast<char *>(v.data()), n);
    return v;
}

static void write_f32(const std::string &path, const std::vector<float> &v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
}

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "inst/scripts/ref_data/ops";

    std::ifstream man(dir + "/manifest.tsv");
    if (!man) {
        std::printf("missing %s/manifest.tsv -- run the R side first\n", dir.c_str());
        return 1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "ops");
    Ort::AllocatorWithDefaultOptions alloc;
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::string line;
    while (std::getline(man, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        std::string name, dims_s;
        std::getline(ls, name, '\t');
        std::getline(ls, dims_s, '\t');

        std::vector<int64_t> shape;
        {
            std::istringstream ds(dims_s);
            std::string tok;
            while (std::getline(ds, tok, ',')) shape.push_back(std::stoll(tok));
        }

        std::printf("%-24s ", name.c_str());
        std::fflush(stdout);

        std::vector<float> input = read_f32(dir + "/" + name + ".in.bin");
        if (input.empty()) { std::printf("SKIP (no input)\n"); continue; }

        try {
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(1);
            Ort::Session session(env, (dir + "/" + name + ".onnx").c_str(), opts);

            auto in_held = session.GetInputNameAllocated(0, alloc);
            const char *in_names[] = { in_held.get() };
            Ort::Value in_val = Ort::Value::CreateTensor<float>(
                mem, input.data(), input.size(), shape.data(), shape.size());

            const size_t n_out = session.GetOutputCount();
            std::vector<Ort::AllocatedStringPtr> holders;
            std::vector<const char *> out_names;
            for (size_t k = 0; k < n_out; k++) {
                holders.push_back(session.GetOutputNameAllocated(k, alloc));
                out_names.push_back(holders.back().get());
            }

            auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in_val, 1,
                                    out_names.data(), out_names.size());

            for (size_t k = 0; k < n_out; k++) {
                auto info = outs[k].GetTensorTypeAndShapeInfo();
                const size_t n = info.GetElementCount();
                std::vector<float> v(n);
                switch (info.GetElementType()) {
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                        const float *p = outs[k].GetTensorData<float>();
                        v.assign(p, p + n);
                        break;
                    }
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                        const int64_t *p = outs[k].GetTensorData<int64_t>();
                        for (size_t i = 0; i < n; i++) v[i] = static_cast<float>(p[i]);
                        break;
                    }
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                        const int32_t *p = outs[k].GetTensorData<int32_t>();
                        for (size_t i = 0; i < n; i++) v[i] = static_cast<float>(p[i]);
                        break;
                    }
                    default:
                        std::printf("[out%zu type %d unsupported] ", k,
                                    (int)info.GetElementType());
                        continue;
                }
                const std::string suffix = k == 0 ? ".ort.bin"
                                                  : (".ort" + std::to_string(k) + ".bin");
                write_f32(dir + "/" + name + suffix, v);
                std::printf("%sout%zu n=%zu head=", k ? " | " : "", k, n);
                for (size_t i = 0; i < n && i < 4; i++) std::printf(" %g", v[i]);
            }
            std::printf("\n");
        } catch (const Ort::Exception &e) {
            std::printf("FAIL %s\n", e.what());
        }
    }
    return 0;
}
