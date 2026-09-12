// Reference outputs, produced by ONNX Runtime.
//
// The point of this program is to be a SECOND implementation: it shares no
// code with ggmlR, so agreement between the two is evidence that both are
// right, where ggmlR agreeing with itself is evidence of nothing.  It exists
// because a model can return output of the correct length and shape while
// every number in it is wrong -- roberta returned NaN for a long time and was
// reported as OK by a check that looked only at the length.
//
// Reads the inputs ggmlR wrote (float32, little-endian) plus its manifest, and
// writes its own first output beside them in the same format, for compare.R.
//
// Build (ORT_DIR = an unpacked onnxruntime-linux-x64 release):
//   g++ -O2 -std=c++17 ort_reference.cpp -o ort_reference \
//       -I$ORT_DIR/include -L$ORT_DIR/lib -lonnxruntime -Wl,-rpath,$ORT_DIR/lib
//
// Usage: ./ort_reference <models-dir> <data-dir>

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct InputSpec {
    std::string name;
    std::vector<int64_t> shape;
};

struct ModelSpec {
    std::string tag;
    std::string file;
    std::vector<InputSpec> inputs;
};

static std::vector<float> read_f32(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<float> v(static_cast<size_t>(n) / sizeof(float));
    f.read(reinterpret_cast<char *>(v.data()), n);
    return v;
}

static void write_f32(const std::string &path, const std::vector<float> &v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
}

static std::vector<int64_t> parse_shape(const std::string &s) {
    std::vector<int64_t> dims;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) dims.push_back(std::stoll(part));
    return dims;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <models-dir> <data-dir>\n", argv[0]);
        return 1;
    }
    const std::string models_dir = argv[1];
    const std::string dir        = argv[2];

    // Read the manifest, preserving the order models appear in it.
    std::vector<ModelSpec> specs;
    std::map<std::string, size_t> index;
    {
        std::ifstream mf(dir + "/manifest.tsv");
        if (!mf) { std::fprintf(stderr, "no manifest in %s\n", dir.c_str()); return 1; }
        std::string line;
        while (std::getline(mf, line)) {
            std::stringstream ss(line);
            std::string kind, tag, file, name, shape, isint;
            std::getline(ss, kind, '\t');
            std::getline(ss, tag,  '\t');
            std::getline(ss, file, '\t');
            if (kind != "in") continue;
            std::getline(ss, name,  '\t');
            std::getline(ss, shape, '\t');
            std::getline(ss, isint, '\t');
            auto it = index.find(tag);
            if (it == index.end()) {
                index[tag] = specs.size();
                specs.push_back(ModelSpec{tag, file, {}});
                it = index.find(tag);
            }
            specs[it->second].inputs.push_back(InputSpec{name, parse_shape(shape)});
        }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "ggmlr-reference");

    for (const ModelSpec &spec : specs) {
        std::printf("%-45s ", spec.file.c_str());
        std::fflush(stdout);
        try {
            Ort::SessionOptions opts;
            // One thread: the summation order is then at least stable between
            // reference runs.  It says nothing about ggmlR's order, which is
            // why the comparison uses a tolerance rather than equality.
            opts.SetIntraOpNumThreads(1);
            opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

            Ort::Session session(env, (models_dir + "/" + spec.file).c_str(), opts);
            Ort::AllocatorWithDefaultOptions alloc;

            auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            // Inputs are fed by NAME, matched against the session's own names,
            // so an ordering difference between the two sides cannot silently
            // swap two tensors of the same shape.
            size_t n_in = session.GetInputCount();
            std::vector<Ort::AllocatedStringPtr> in_holders;
            std::vector<const char *> in_names;
            std::vector<Ort::Value> in_values;
            std::vector<std::vector<float>>   f32_store;
            std::vector<std::vector<int64_t>> i64_store;
            f32_store.reserve(n_in);
            i64_store.reserve(n_in);

            bool missing = false;
            for (size_t i = 0; i < n_in; i++) {
                auto held = session.GetInputNameAllocated(i, alloc);
                std::string nm = held.get();

                const InputSpec *spec_in = nullptr;
                for (const auto &s : spec.inputs) if (s.name == nm) spec_in = &s;
                if (!spec_in) {
                    std::printf("SKIP (no dumped input '%s')\n", nm.c_str());
                    missing = true;
                    break;
                }

                std::vector<float> raw =
                    read_f32(dir + "/" + spec.tag + ".in." + nm + ".bin");
                if (raw.empty()) {
                    std::printf("SKIP (empty input '%s')\n", nm.c_str());
                    missing = true;
                    break;
                }

                auto elem = session.GetInputTypeInfo(i)
                                   .GetTensorTypeAndShapeInfo().GetElementType();
                std::vector<int64_t> shape = spec_in->shape;

                if (elem == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    i64_store.emplace_back();
                    auto &v = i64_store.back();
                    v.reserve(raw.size());
                    for (float x : raw) v.push_back(static_cast<int64_t>(x));
                    in_values.push_back(Ort::Value::CreateTensor<int64_t>(
                        mem, v.data(), v.size(), shape.data(), shape.size()));
                } else {
                    f32_store.push_back(std::move(raw));
                    auto &v = f32_store.back();
                    in_values.push_back(Ort::Value::CreateTensor<float>(
                        mem, v.data(), v.size(), shape.data(), shape.size()));
                }
                in_names.push_back(held.get());
                in_holders.push_back(std::move(held));
            }
            if (missing) continue;

            // Every output the model declares, not just the first.  A detector
            // splits its answer across them -- MaskRCNN returns boxes, labels,
            // scores and masks separately -- and a disagreement in the box list
            // alone cannot say whether the two runs found different objects or
            // ordered the same ones differently.  The extras go to their own
            // files; the first stays where compare.R expects it.
            size_t n_out = session.GetOutputCount();
            std::vector<Ort::AllocatedStringPtr> out_holders;
            std::vector<const char *> out_names;
            for (size_t k = 0; k < n_out; k++) {
                out_holders.push_back(session.GetOutputNameAllocated(k, alloc));
                out_names.push_back(out_holders.back().get());
            }

            auto outputs = session.Run(Ort::RunOptions{nullptr},
                                       in_names.data(), in_values.data(),
                                       in_values.size(),
                                       out_names.data(), out_names.size());

            for (size_t k = 1; k < n_out; k++) {
                auto ei = outputs[k].GetTensorTypeAndShapeInfo();
                size_t en = ei.GetElementCount();
                std::vector<float> ev(en);
                auto et = ei.GetElementType();
                if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    const float *p = outputs[k].GetTensorData<float>();
                    ev.assign(p, p + en);
                } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    const int64_t *p = outputs[k].GetTensorData<int64_t>();
                    for (size_t i = 0; i < en; i++) ev[i] = static_cast<float>(p[i]);
                } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                    const int32_t *p = outputs[k].GetTensorData<int32_t>();
                    for (size_t i = 0; i < en; i++) ev[i] = static_cast<float>(p[i]);
                } else {
                    std::printf("  [out%zu] unsupported type %d\n", k, (int)et);
                    continue;
                }
                write_f32(dir + "/" + spec.tag + ".out" + std::to_string(k) + ".ort.bin", ev);
                auto sh = ei.GetShape();
                std::printf("  [out%zu] '%s' n=%zu shape=[", k, out_names[k], en);
                for (size_t i = 0; i < sh.size(); i++)
                    std::printf("%lld%s", (long long)sh[i], i + 1 < sh.size() ? "," : "");
                std::printf("] head=");
                for (size_t i = 0; i < en && i < 4; i++) std::printf(" %.6g", ev[i]);
                std::printf("\n");
            }

            auto info = outputs[0].GetTensorTypeAndShapeInfo();
            size_t n = info.GetElementCount();
            std::vector<float> out(n);
            if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                const float *p = outputs[0].GetTensorData<float>();
                out.assign(p, p + n);
            } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                const int64_t *p = outputs[0].GetTensorData<int64_t>();
                for (size_t i = 0; i < n; i++) out[i] = static_cast<float>(p[i]);
            } else {
                std::printf("SKIP (output type %d)\n", (int)info.GetElementType());
                continue;
            }

            write_f32(dir + "/" + spec.tag + ".ort.bin", out);
            std::printf("OK  out=%zu  head=", n);
            for (size_t i = 0; i < n && i < 3; i++) std::printf(" %.6g", out[i]);
            std::printf("\n");
        } catch (const Ort::Exception &e) {
            std::printf("FAIL %s\n", e.what());
        }
    }
    return 0;
}
