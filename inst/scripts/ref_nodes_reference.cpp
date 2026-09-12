// Reference values for named edges inside a whole model, from ONNX Runtime.
//
// Companion to ref_patch_outputs.R and the ONNX_DUMP_NODES dump in
// onnx_ggml.c: the patcher adds internal edges to a model's graph.output, this
// runs the patched model and writes each output where the comparison expects
// it, and ggmlR writes its own side of the same edges.
//
// Separate from ref_ops_reference.cpp rather than a flag on it.  That one
// answers "which op disagrees" with one-node models it generates itself: one
// input named by position, one shape, synthetic values.  This one runs a real
// model on real inputs, with several named inputs of different element types,
// and the two have almost no code in common beyond reading floats.
//
// Element type comes from the model, not from the caller: a detector's inputs
// are float, but its internal edges carry int64 indices and int32 shapes, and
// an index read as float is off by whatever the bit pattern happens to mean.
// Everything is WRITTEN as float32 though -- the dump side does the same, and
// one dtype keeps the comparison to a single code path.
//
// Build:
//   g++ -O2 -std=c++17 ref_nodes_reference.cpp -o ort_nodes \
//       -I$ORT_DIR/include -L$ORT_DIR/lib -lonnxruntime -Wl,-rpath,$ORT_DIR/lib
//
// Usage:
//   ort_nodes <model.onnx> <data_dir> <tag> [name=d,d,d ...]
//
// Reads  <data_dir>/<tag>.in.<name>.bin  for every model input.
// Writes <data_dir>/<edge>.ort.bin       for every model output.
//
// Input shapes are given on the command line when the model does not state
// them.  A detector declares its image input fully dynamic -- every dimension
// free -- and an element count cannot resolve three unknowns, so guessing is
// not on the table: 150528 floats are equally [3,224,224] and [1,3,224,224]
// and [224,224,3], and the last two are a different picture.

#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <cstring>
#include <fstream>
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

// An edge name may contain '/', which would turn the dump path into a
// directory that does not exist.  The dump side rewrites the same character,
// so both halves land on the same filename.
static std::string safe_name(const std::string &s) {
    std::string o = s;
    for (char &c : o) if (c == '/') c = '_';
    return o;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::printf("usage: ort_nodes <model.onnx> <data_dir> <tag>\n");
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string dir        = argv[2];
    const std::string tag        = argv[3];

    // name=d,d,d overrides for inputs the model leaves dynamic.
    std::vector<std::pair<std::string, std::vector<int64_t>>> shape_args;
    for (int a = 4; a < argc; a++) {
        const std::string s = argv[a];
        const size_t eq = s.find('=');
        if (eq == std::string::npos) continue;
        std::vector<int64_t> dims;
        size_t p = eq + 1;
        while (p < s.size()) {
            size_t c = s.find(',', p);
            if (c == std::string::npos) c = s.size();
            dims.push_back(std::stoll(s.substr(p, c - p)));
            p = c + 1;
        }
        shape_args.emplace_back(s.substr(0, eq), dims);
    }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "nodes");
    Ort::AllocatorWithDefaultOptions alloc;
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    try {
        Ort::SessionOptions opts;
        // One thread: a reference that changes with the thread count is not a
        // reference.  Reductions accumulate in a different order per split.
        opts.SetIntraOpNumThreads(1);
        Ort::Session session(env, model_path.c_str(), opts);

        // ── inputs ──────────────────────────────────────────────
        const size_t n_in = session.GetInputCount();
        std::vector<Ort::AllocatedStringPtr> in_held;
        std::vector<const char *>            in_names;
        std::vector<Ort::Value>              in_vals;
        // The float buffers must outlive Run: CreateTensor does not copy, it
        // borrows. Holding them in a vector that is only appended to is not
        // enough -- a reallocation moves the data out from under the tensors --
        // so it is reserved up front.
        std::vector<std::vector<float>>      f32_store;
        std::vector<std::vector<int64_t>>    i64_store;
        std::vector<std::vector<int64_t>>    shape_store;
        f32_store.reserve(n_in);
        i64_store.reserve(n_in);
        shape_store.reserve(n_in);

        for (size_t k = 0; k < n_in; k++) {
            in_held.push_back(session.GetInputNameAllocated(k, alloc));
            const std::string nm = in_held.back().get();
            in_names.push_back(in_held.back().get());

            auto info  = session.GetInputTypeInfo(k).GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            // A free dimension is -1 in the model; the dump was made at a
            // concrete size, so the element count decides it. Only one such
            // dimension can be resolved this way, which is all these models need.
            shape_store.push_back(shape);
            std::vector<int64_t> &sh = shape_store.back();

            const std::string p = dir + "/" + tag + ".in." + nm + ".bin";
            std::vector<float> raw = read_f32(p);
            if (raw.empty()) {
                std::printf("missing input dump %s\n", p.c_str());
                return 1;
            }

            for (const auto &sa : shape_args)
                if (sa.first == nm) { sh = sa.second; break; }

            int64_t known = 1;
            int neg = -1;
            for (size_t d = 0; d < sh.size(); d++) {
                if (sh[d] < 0) neg = (int)d; else known *= sh[d];
            }
            if (neg >= 0 && known > 0) sh[neg] = (int64_t)raw.size() / known;

            int64_t want = 1;
            for (int64_t d : sh) want *= d;
            if (sh.empty() || want <= 0) {
                std::printf("input '%s' has no usable shape -- pass %s=d,d,d\n",
                            nm.c_str(), nm.c_str());
                return 1;
            }

            const ONNXTensorElementDataType et = info.GetElementType();
            if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                // Token ids and edge indices are int64 in the model and float
                // in the dump, because the dump format is one dtype throughout.
                i64_store.emplace_back(raw.begin(), raw.end());
                in_vals.push_back(Ort::Value::CreateTensor<int64_t>(
                    mem, i64_store.back().data(), i64_store.back().size(),
                    sh.data(), sh.size()));
            } else {
                f32_store.push_back(std::move(raw));
                in_vals.push_back(Ort::Value::CreateTensor<float>(
                    mem, f32_store.back().data(), f32_store.back().size(),
                    sh.data(), sh.size()));
            }
            std::printf("  in  %-20s n=%zu\n", nm.c_str(),
                        et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                            ? i64_store.back().size() : f32_store.back().size());
        }

        // ── outputs: whatever the model declares, patched ones included ──
        const size_t n_out = session.GetOutputCount();
        std::vector<Ort::AllocatedStringPtr> out_held;
        std::vector<const char *>            out_names;
        for (size_t k = 0; k < n_out; k++) {
            out_held.push_back(session.GetOutputNameAllocated(k, alloc));
            out_names.push_back(out_held.back().get());
        }
        std::printf("  running, %zu input(s), %zu output(s)\n", n_in, n_out);

        auto outs = session.Run(Ort::RunOptions{nullptr},
                                in_names.data(), in_vals.data(), n_in,
                                out_names.data(), n_out);

        for (size_t k = 0; k < n_out; k++) {
            if (!outs[k].IsTensor()) {
                std::printf("  out %-20s SKIP (not a tensor)\n", out_names[k]);
                continue;
            }
            auto info = outs[k].GetTensorTypeAndShapeInfo();
            const size_t n = info.GetElementCount();
            std::vector<float> v(n);
            switch (info.GetElementType()) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                    const float *p = outs[k].GetTensorData<float>();
                    v.assign(p, p + n); break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                    const int64_t *p = outs[k].GetTensorData<int64_t>();
                    for (size_t i = 0; i < n; i++) v[i] = (float)p[i]; break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                    const int32_t *p = outs[k].GetTensorData<int32_t>();
                    for (size_t i = 0; i < n; i++) v[i] = (float)p[i]; break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                    const uint8_t *p = outs[k].GetTensorData<uint8_t>();
                    for (size_t i = 0; i < n; i++) v[i] = (float)p[i]; break;
                }
                // Quantised weights are int8, and a conv's weights are exactly
                // what an int32 reference implementation needs to read.
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
                    const int8_t *p = outs[k].GetTensorData<int8_t>();
                    for (size_t i = 0; i < n; i++) v[i] = (float)p[i]; break;
                }
                default:
                    std::printf("  out %-20s SKIP (type %d)\n", out_names[k],
                                (int)info.GetElementType());
                    continue;
            }

            const std::string path = dir + "/" + safe_name(out_names[k]) + ".ort.bin";
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char *>(v.data()),
                    (std::streamsize)(v.size() * sizeof(float)));

            auto shape = info.GetShape();
            std::printf("  out %-20s n=%zu shape=[", out_names[k], n);
            for (size_t d = 0; d < shape.size(); d++)
                std::printf("%s%lld", d ? "," : "", (long long)shape[d]);
            std::printf("] head=");
            for (size_t i = 0; i < n && i < 4; i++) std::printf(" %g", v[i]);
            std::printf("\n");
        }
    } catch (const Ort::Exception &e) {
        std::printf("FAIL %s\n", e.what());
        return 1;
    }
    return 0;
}
