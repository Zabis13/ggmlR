// ggmlR Tensor Parallelism (P2P) — Vulkan split buffer type.
//
// NOT part of upstream ggml. Upstream implements a tensor-split buffer type only
// for CUDA/SYCL; this is a Vulkan port. It row-splits a weight matrix across N
// Vulkan devices so each device holds a horizontal slice of the rows, enabling
// true tensor parallelism (as opposed to layer-split / replication).
//
// This file is #included into ggml-vulkan.cpp as a single translation unit (like
// the other ggml-vulkan-*.cpp parts); all functions are static. It must be
// included AFTER ggml-vulkan-graph.cpp so ggml_vk_get_device_count() and the
// vk_instance / vk_buffer machinery are visible.
//
// Stage E2 scope (this commit): the row-split MATH (get_row_split /
// get_row_rounding / nbytes_split) is pure arithmetic and is unit-tested on a
// single GPU via R_ggml_vk_split_row_range. The buffer_type SCAFFOLD (context,
// iface, init/set/get_tensor) is present but its multi-device allocation path
// cannot be exercised without >=2 GPUs — see TODO.md / memory.

// ---------------------------------------------------------------------------
// Transport abstraction (architectural hook, per NVLink discussion).
// The way result slices are gathered across devices is deliberately abstracted
// so a device-group (VK_KHR_device_group / LDA, NVLink-capable) transport can be
// added later as a second implementation without touching the row-split math.
// opaque-fd is the portable default (AMD/RADV + any PCIe topology).
// ---------------------------------------------------------------------------
enum vk_split_transport {
    VK_SPLIT_TRANSPORT_OPAQUE_FD = 0,  // default: portable, PCIe P2P via external_memory_fd
    VK_SPLIT_TRANSPORT_DEVICE_GROUP,   // experimental: NVIDIA LDA, may route over NVLink
};

// Pad each device's row slice so the last row is a multiple of this many
// elements, matching the CUDA split buffer (avoids out-of-bounds in matmul).
#define VK_SPLIT_MATRIX_ROW_PADDING 512

// ---------------------------------------------------------------------------
// Row-split math (pure; unit-tested on a single GPU).
// tensor_split is a cumulative fraction array: tensor_split[i] is the fraction
// of rows *before* device i (so tensor_split[0] == 0.0), matching upstream CUDA.
// ---------------------------------------------------------------------------

// Row rounding granularity. Upstream CUDA derives this from the MMQ tile height
// per-device; the Vulkan matmul path aligns slice boundaries to the matrix row
// padding, which is a safe (>=) choice for correctness of the split.
static int64_t ggml_vk_split_row_rounding(int n_devices) {
    GGML_UNUSED(n_devices);
    return VK_SPLIT_MATRIX_ROW_PADDING;
}

// Compute [row_low, row_high) for device `id` given a tensor's row count.
// Mirrors ggml-cuda.cu get_row_split. Boundaries are rounded down to a multiple
// of the rounding granularity; the last device always covers up to nrows.
static void ggml_vk_split_row_range(int64_t nrows, const float * tensor_split,
                                    int n_devices, int id,
                                    int64_t * row_low, int64_t * row_high) {
    const int64_t rounding = ggml_vk_split_row_rounding(n_devices);

    *row_low = (id == 0) ? 0 : (int64_t)(nrows * tensor_split[id]);
    *row_low -= *row_low % rounding;

    if (id == n_devices - 1) {
        *row_high = nrows;
    } else {
        *row_high = (int64_t)(nrows * tensor_split[id + 1]);
        *row_high -= *row_high % rounding;
    }

    // Clamp defensively so a degenerate tensor_split never yields an inverted or
    // out-of-range range (the pure test relies on this being total & monotone).
    if (*row_low  < 0)     *row_low  = 0;
    if (*row_high > nrows) *row_high = nrows;
    if (*row_high < *row_low) *row_high = *row_low;
}

// Bytes for `nrows_split` rows of a tensor (row size derived from ne[0] & type).
// Unlike upstream CUDA this does not static_assert GGML_MAX_DIMS==4: the ggmlR
// tree uses GGML_MAX_DIMS==5, and this quantity depends only on ne[0] & type.
static size_t ggml_vk_split_nbytes(const struct ggml_tensor * tensor, int64_t nrows_split) {
    return (size_t) nrows_split * ggml_row_size(tensor->type, tensor->ne[0]);
}

// Normalize a caller-provided per-device weight vector into the cumulative
// fraction array the split math expects (out[0]==0, monotone non-decreasing,
// out has n_devices entries). If `weights` is NULL or sums to ~0, split evenly.
static void ggml_vk_split_normalize(const float * weights, int n_devices, float * out /*[n_devices]*/) {
    float sum = 0.0f;
    if (weights) {
        for (int i = 0; i < n_devices; i++) sum += weights[i] > 0.0f ? weights[i] : 0.0f;
    }
    if (!weights || sum <= 0.0f) {
        for (int i = 0; i < n_devices; i++) out[i] = (float) i / (float) n_devices;
        return;
    }
    float acc = 0.0f;
    for (int i = 0; i < n_devices; i++) {
        out[i] = acc / sum;
        acc += weights[i] > 0.0f ? weights[i] : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Split buffer type — SCAFFOLD.
// Multi-device allocation/set/get is unverifiable on a single GPU; it is written
// to the CUDA pattern but gated behind n_devices and left for target hardware.
// ---------------------------------------------------------------------------

struct ggml_backend_vk_split_buffer_type_context {
    int                              main_device;
    int                              n_devices;
    std::vector<float>               tensor_split;   // cumulative fractions, size n_devices
    vk_split_transport               transport;
    std::string                      name;
};

// Per-tensor slices: one vk_buffer per device (empty where the slice is 0 rows).
struct ggml_backend_vk_split_tensor_extra {
    std::vector<vk_buffer> slices;   // size n_devices
};

struct ggml_backend_vk_split_buffer_context {
    std::vector<ggml_backend_vk_split_tensor_extra *> tensor_extras;
    ~ggml_backend_vk_split_buffer_context() {
        for (auto * e : tensor_extras) {
            delete e;   // vk_buffer is shared_ptr; slices free themselves
        }
    }
};

static void ggml_backend_vk_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_vk_split_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_vk_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000;  // dummy: real pointers live in the per-tensor extra
}

static enum ggml_status ggml_backend_vk_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src == nullptr && "views of split tensors are not supported");
    GGML_ASSERT(ggml_is_contiguous(tensor)  && "split buffers only support contiguous tensors");

    auto * ctx      = (ggml_backend_vk_split_buffer_context *)      buffer->context;
    auto * buft_ctx = (ggml_backend_vk_split_buffer_type_context *) buffer->buft->context;

    const int64_t nrows = ggml_nrows(tensor);
    const int64_t ne0   = tensor->ne[0];

    auto * extra = new ggml_backend_vk_split_tensor_extra{};
    extra->slices.resize(buft_ctx->n_devices);
    ctx->tensor_extras.push_back(extra);

    for (int id = 0; id < buft_ctx->n_devices; id++) {
        int64_t row_low, row_high;
        ggml_vk_split_row_range(nrows, buft_ctx->tensor_split.data(),
                                buft_ctx->n_devices, id, &row_low, &row_high);
        const int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;   // this device owns no rows of this tensor
        }

        size_t size = ggml_vk_split_nbytes(tensor, nrows_split);
        // Pad the last row up to the matrix row padding (matches CUDA).
        if (ne0 % VK_SPLIT_MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, VK_SPLIT_MATRIX_ROW_PADDING - ne0 % VK_SPLIT_MATRIX_ROW_PADDING);
        }

        // Allocate the slice on device `id`. Exportable via opaque-fd so the
        // matmul result-gather transport can share it across devices.
        vk_device slice_dev = ggml_vk_get_device((size_t) id);
        const bool want_export = (buft_ctx->transport == VK_SPLIT_TRANSPORT_OPAQUE_FD)
                                 && slice_dev->external_memory_fd;
        extra->slices[id] = ggml_vk_create_buffer(
            slice_dev, size,
            { vk::MemoryPropertyFlagBits::eDeviceLocal },
            /*import_ptr=*/nullptr, /*export_fd=*/want_export);
    }

    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_vk_split_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                    const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0 && "split tensors must be set in their entirety");
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only support contiguous tensors");

    auto * buft_ctx = (ggml_backend_vk_split_buffer_type_context *) buffer->buft->context;
    auto * extra    = (ggml_backend_vk_split_tensor_extra *)        tensor->extra;

    const int64_t nrows = ggml_nrows(tensor);
    const size_t  nb1   = tensor->nb[1];

    for (int id = 0; id < buft_ctx->n_devices; id++) {
        int64_t row_low, row_high;
        ggml_vk_split_row_range(nrows, buft_ctx->tensor_split.data(),
                                buft_ctx->n_devices, id, &row_low, &row_high);
        const int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0 || !extra->slices[id]) {
            continue;
        }
        const char * src = (const char *) data + row_low * nb1;
        const size_t sz  = (size_t) nrows_split * nb1;
        ggml_vk_buffer_write(extra->slices[id], 0, src, sz);
    }
}

static void ggml_backend_vk_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                                    void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0 && "split tensors must be read in their entirety");
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only support contiguous tensors");

    auto * buft_ctx = (ggml_backend_vk_split_buffer_type_context *) buffer->buft->context;
    auto * extra    = (ggml_backend_vk_split_tensor_extra *)        tensor->extra;

    const int64_t nrows = ggml_nrows(tensor);
    const size_t  nb1   = tensor->nb[1];

    for (int id = 0; id < buft_ctx->n_devices; id++) {
        int64_t row_low, row_high;
        ggml_vk_split_row_range(nrows, buft_ctx->tensor_split.data(),
                                buft_ctx->n_devices, id, &row_low, &row_high);
        const int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0 || !extra->slices[id]) {
            continue;
        }
        char * dst = (char *) data + row_low * nb1;
        const size_t sz = (size_t) nrows_split * nb1;
        ggml_vk_buffer_read(extra->slices[id], 0, dst, sz);
    }
}

static void ggml_backend_vk_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
    // Split buffers hold weights that are always fully set via set_tensor; there
    // is no meaningful whole-buffer clear across devices. No-op (matches CUDA).
}

static const ggml_backend_buffer_i ggml_backend_vk_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_vk_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_vk_split_buffer_clear,
    /* .reset           = */ NULL,
};

// ---------------------------------------------------------------------------
// Public C entry point (not upstream): pure row-split math for unit tests.
// No Vulkan device is touched — this is the arithmetic verified on a single GPU.
// ---------------------------------------------------------------------------
extern "C" int ggml_backend_vk_split_row_ranges(int64_t nrows, const float * weights,
                                                int n_devices,
                                                int64_t * row_low, int64_t * row_high) {
    if (n_devices <= 0 || nrows < 0 || !row_low || !row_high) {
        return -1;
    }
    std::vector<float> split(n_devices);
    ggml_vk_split_normalize(weights, n_devices, split.data());
    for (int id = 0; id < n_devices; id++) {
        ggml_vk_split_row_range(nrows, split.data(), n_devices, id,
                                &row_low[id], &row_high[id]);
    }
    return 0;
}
