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

// Extra headers for the P2P self-test (chrono for timing, cstring for memcmp/
// snprintf, unistd for close()). This file is part of the ggml-vulkan.cpp
// translation unit; these are additive and idempotent if already pulled in.
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Transport abstraction (architectural hook, per NVLink discussion).
// The way result slices are gathered across devices is deliberately abstracted
// so alternative cross-device transports can be swapped in without touching the
// row-split math. All cross-device copies go through ggml_vk_p2p_copy() below.
//
// Empirically determined on 4x Tesla P100 (NVIDIA proprietary driver, 2026-07):
//   * OPAQUE_FD works for loopback (same device) but does NOT share memory
//     cross-device — an imported fd reads back as all zeros even with a dedicated
//     allocation bound to the exporter's exact memory type. NVIDIA opaque-fd does
//     not alias VRAM between two separate VkDevices here.
//   * DEVICE_GROUP is unavailable: the driver reports every P100 in its own
//     single-device group (no LDA / NVLink peer path via Vulkan).
// Therefore HOST_STAGING (device -> host -> device) is the portable default: it
// is correct everywhere (NVIDIA and AMD/RADV), needs no external dependency, and
// is bandwidth-limited by PCIe + one RAM round-trip. OPAQUE_FD is kept for AMD,
// where cross-device dma-buf sharing may actually alias. See memory / TODO.
// ---------------------------------------------------------------------------
enum vk_split_transport {
    VK_SPLIT_TRANSPORT_HOST_STAGING = 0,  // default: portable, correct everywhere (device->host->device)
    VK_SPLIT_TRANSPORT_OPAQUE_FD,         // PCIe P2P via external_memory_fd (AMD/RADV; broken cross-device on NVIDIA)
    VK_SPLIT_TRANSPORT_DEVICE_GROUP,      // experimental: NVIDIA LDA, may route over NVLink (unavailable on P100)
};

// ggmlR TP: single point for a cross-device buffer copy of `bytes` from
// `src_buf` (on its device) into `dst_buf` (on its device), starting at the given
// offsets. This is the transport Stage E3's result-gather (and activation
// broadcast) routes through. Only HOST_STAGING is wired up today; the other
// transports fall back to it until they are verified on their target hardware.
//
// HOST_STAGING: read src slice into a host bounce buffer, then write it into dst.
// ggml_vk_buffer_read / ggml_vk_buffer_write are each internally synchronous
// (they submit and wait on their device's fence), so no cross-device semaphore is
// needed — the read has fully completed on src before the write begins on dst.
static void ggml_vk_p2p_copy(vk_buffer & dst_buf, size_t dst_offset,
                             vk_buffer & src_buf, size_t src_offset,
                             size_t bytes, vk_split_transport transport) {
    GGML_UNUSED(transport);  // only HOST_STAGING implemented; others fall back here
    if (bytes == 0 || !src_buf || !dst_buf) {
        return;
    }
    std::vector<uint8_t> bounce(bytes);
    ggml_vk_buffer_read(src_buf, src_offset, bounce.data(), bytes);
    ggml_vk_buffer_write(dst_buf, dst_offset, bounce.data(), bytes);
}

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
// P2P self-test (ggmlR TP, not upstream).
//
// Exercises the opaque-fd export/import transport that Stage E3 will use to move
// weight/activation slices between Vulkan devices. Two modes, selected by whether
// src_dev == dst_dev:
//
//   loopback (src == dst): export an fd on a device and import it back on the
//     SAME device. Sanity-checks that the fd mechanism itself works (allocation
//     is exportable, getMemoryFdKHR succeeds, ImportMemoryFdInfoKHR binds). Does
//     NOT exercise any device<->device link.
//
//   cross-device (src != dst): export on src, import on dst, then vkCmdCopyBuffer
//     from the imported (remote) buffer into a local dst buffer on the dst queue.
//     This is the path whose routing (NVLink vs PCIe) the driver decides; we can
//     only MEASURE the achieved bandwidth, we cannot query the route from Vulkan.
//
// Correctness: a byte pattern written on src is read back from the dst-local copy
// and compared. Bandwidth: the copy is repeated `iters` times under a fence and
// timed; GB/s = bytes * iters / seconds (1 GB = 1e9 bytes, to match nvidia-smi).
//
// IMPORTANT (reporting): a measured bandwidth above the PCIe 3.0 x16 ceiling
// (~16 GB/s) is EMPIRICAL evidence that a faster link (e.g. NVLink) carried the
// bytes — it is NOT a claim that Vulkan used an NVLink API. There is no Vulkan
// call that reports the physical route; the conclusion is inferred from the rate.
//
// Returns 0 on success (data verified), <0 on failure. `out_gbps` receives the
// measured cross-device bandwidth (0 for loopback / on failure). `report` gets a
// human-readable summary.
// Append a printf-style line to a fixed-size report buffer (bounded, never
// overflows). Marked with the printf format attribute so -Wformat-security is
// satisfied even for calls with no variadic arguments.
static void ggml_vk_report_append(char * report, size_t report_size, const char * fmt, ...)
    __attribute__((format(printf, 3, 4)));
static void ggml_vk_report_append(char * report, size_t report_size, const char * fmt, ...) {
    size_t len = strlen(report);
    if (len >= report_size) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(report + len, report_size - len, fmt, ap);
    va_end(ap);
}

static int ggml_vk_p2p_selftest_impl(int src_dev, int dst_dev, size_t bytes, int iters,
                                     vk_split_transport transport,
                                     double * out_gbps, char * report, size_t report_size) {
    if (out_gbps) *out_gbps = 0.0;
    #define say(...) ggml_vk_report_append(report, report_size, __VA_ARGS__)

    const int n_dev = ggml_vk_get_device_count();
    if (src_dev < 0 || dst_dev < 0 || src_dev >= n_dev || dst_dev >= n_dev) {
        say("p2p_selftest: device index out of range (have %d device(s))\n", n_dev);
        return -1;
    }
    if (bytes == 0 || iters <= 0) {
        say("p2p_selftest: bytes and iters must be > 0\n");
        return -1;
    }

    const bool loopback = (src_dev == dst_dev);
    vk_device src = ggml_vk_get_device((size_t) src_dev);
    vk_device dst = ggml_vk_get_device((size_t) dst_dev);

    say("p2p_selftest: %s  src=dev%d (%s)  dst=dev%d (%s)  %zu bytes x%d  transport=%s\n",
        loopback ? "LOOPBACK" : "CROSS-DEVICE",
        src_dev, src->name.c_str(), dst_dev, dst->name.c_str(), bytes, iters,
        transport == VK_SPLIT_TRANSPORT_HOST_STAGING ? "host-staging"
        : transport == VK_SPLIT_TRANSPORT_OPAQUE_FD  ? "opaque-fd" : "device-group");

    // ---------------------------------------------------------------------
    // HOST_STAGING path (ggmlR TP): the portable, correct-everywhere transport.
    // No fd export/import — just device-local buffers on each side and a
    // device->host->device copy via ggml_vk_p2p_copy. This is exactly the
    // transport Stage E3 uses, so a green result here validates the real path.
    // ---------------------------------------------------------------------
    if (transport == VK_SPLIT_TRANSPORT_HOST_STAGING) {
        const auto dev_local = vk::MemoryPropertyFlagBits::eDeviceLocal;
        vk_buffer src_buf, dst_buf;
        int rc = 0;
        try {
            src_buf = ggml_vk_create_buffer(src, bytes, { dev_local });
            dst_buf = ggml_vk_create_buffer(dst, bytes, { dev_local });
            if (!src_buf || !dst_buf) {
                say("  FAIL: could not allocate device-local buffers\n");
                return -3;
            }
            std::vector<uint8_t> pattern(bytes);
            for (size_t i = 0; i < bytes; i++) pattern[i] = (uint8_t)((i * 131u + 7u) & 0xFF);
            ggml_vk_buffer_write(src_buf, 0, pattern.data(), bytes);

            // Correctness: one staged copy, then read back and compare.
            ggml_vk_p2p_copy(dst_buf, 0, src_buf, 0, bytes, transport);
            std::vector<uint8_t> readback(bytes);
            ggml_vk_buffer_read(dst_buf, 0, readback.data(), bytes);
            if (memcmp(readback.data(), pattern.data(), bytes) != 0) {
                size_t first = 0;
                while (first < bytes && readback[first] == pattern[first]) first++;
                say("  FAIL: data mismatch at byte %zu (got %u, want %u)\n",
                    first, readback[first], pattern[first]);
                return -6;
            }
            say("  OK: %zu bytes verified via host-staging (device->host->device)\n", bytes);

            // Bandwidth: `iters` staged copies, timed as one batch. GB/s counts the
            // bytes moved device-to-device (the host round-trip is the cost).
            if (!loopback) {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iters; i++) {
                    ggml_vk_p2p_copy(dst_buf, 0, src_buf, 0, bytes, transport);
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double secs = std::chrono::duration<double>(t1 - t0).count();
                double gbps = secs > 0.0 ? (double) bytes * (double) iters / secs / 1e9 : 0.0;
                if (out_gbps) *out_gbps = gbps;
                say("  bandwidth: %.2f GB/s (%d x %zu bytes, host-staged)\n", gbps, iters, bytes);
                say("  => host-staging is PCIe + RAM bounded by design; NVLink is not used.\n");
            } else {
                say("  loopback: bandwidth not meaningful (same-device staging)\n");
            }
        } catch (const vk::SystemError & e) {
            say("  FAIL: Vulkan exception: %s\n", e.what());
            rc = -7;
        }
        return rc;
    }

    // ---------------------------------------------------------------------
    // OPAQUE_FD path (below): requires external_memory_fd on both devices.
    // ---------------------------------------------------------------------
    if (!src->external_memory_fd) {
        say("  FAIL: src device does not support VK_KHR_external_memory_fd\n");
        return -2;
    }
    if (!dst->external_memory_fd) {
        say("  FAIL: dst device does not support VK_KHR_external_memory_fd\n");
        return -2;
    }

    const auto dev_local = vk::MemoryPropertyFlagBits::eDeviceLocal;

    vk_buffer src_buf, imported, dst_local;
    int fd = -1;
    int rc = 0;
    try {
        // 1) Exportable source buffer on src device, filled with a known pattern.
        src_buf = ggml_vk_create_buffer(src, bytes, { dev_local }, nullptr, /*export_fd=*/true);
        if (!src_buf || !src_buf->exportable) {
            say("  FAIL: could not allocate exportable src buffer\n");
            return -3;
        }
        std::vector<uint8_t> pattern(bytes);
        for (size_t i = 0; i < bytes; i++) pattern[i] = (uint8_t)((i * 131u + 7u) & 0xFF);
        ggml_vk_buffer_write(src_buf, 0, pattern.data(), bytes);

        // 2) Export an opaque fd for the src allocation.
        fd = ggml_vk_buffer_export_fd(src_buf);
        if (fd < 0) {
            say("  FAIL: getMemoryFdKHR returned no fd\n");
            return -4;
        }

        // 3) Import that fd on the dst device. The driver takes ownership of the
        //    fd on success; do not close it afterwards. Hand over the exporter's
        //    memory type index: getMemoryFdPropertiesKHR is unreliable on NVIDIA,
        //    and the import must bind to the SAME type the exporter used or it
        //    silently reads back as zeros. All devices here are the same model,
        //    so the index is valid on the dst device too. (ggmlR TP)
        imported = ggml_vk_create_buffer(dst, bytes, { dev_local }, nullptr,
                                         /*export_fd=*/false, /*import_fd=*/fd,
                                         /*import_type_index=*/(int) src_buf->memory_type_index);
        if (!imported) {
            say("  FAIL: ImportMemoryFdInfoKHR failed on dst device\n");
            return -5;
        }
        fd = -1;  // consumed by the driver

        // 4) Local destination buffer on dst; copy imported -> dst_local on the
        //    dst transfer queue. This is the transfer whose route the driver picks.
        dst_local = ggml_vk_create_buffer(dst, bytes, { dev_local });

        // Warm-up copy (first submit pays one-off costs) + correctness copy.
        {
            std::lock_guard<std::recursive_mutex> guard(dst->mutex);
            vk_context subctx = ggml_vk_create_temporary_context(dst->transfer_queue.cmd_pool);
            ggml_vk_ctx_begin(dst, subctx);
            ggml_vk_buffer_copy_async(subctx, dst_local, 0, imported, 0, bytes);
            ggml_vk_ctx_end(subctx);
            ggml_vk_submit(subctx, dst->fence);
            VK_CHECK(dst->device.waitForFences({ dst->fence }, true, UINT64_MAX), "p2p_selftest warmup");
            dst->device.resetFences({ dst->fence });
            ggml_vk_queue_command_pools_cleanup(dst);
        }

        // 5) Correctness: read dst_local back and compare to the source pattern.
        std::vector<uint8_t> readback(bytes);
        ggml_vk_buffer_read(dst_local, 0, readback.data(), bytes);
        if (memcmp(readback.data(), pattern.data(), bytes) != 0) {
            size_t first = 0;
            while (first < bytes && readback[first] == pattern[first]) first++;
            size_t diff = 0, nonzero = 0;
            for (size_t i = 0; i < bytes; i++) {
                if (readback[i] != pattern[i]) diff++;
                if (readback[i] != 0)          nonzero++;
            }
            say("  FAIL: data mismatch at byte %zu (got %u, want %u)\n",
                first, readback[first], pattern[first]);
            say("        %zu/%zu bytes differ; %zu bytes non-zero "
                "(all-zero readback => import bound to wrong memory)\n",
                diff, bytes, nonzero);
            return -6;
        }
        say("  OK: %zu bytes verified across the fd-imported buffer\n", bytes);

        // 6) Bandwidth: time `iters` device->device copies under a single fence.
        double gbps = 0.0;
        if (!loopback) {
            std::lock_guard<std::recursive_mutex> guard(dst->mutex);
            vk_context subctx = ggml_vk_create_temporary_context(dst->transfer_queue.cmd_pool);
            ggml_vk_ctx_begin(dst, subctx);
            for (int i = 0; i < iters; i++) {
                ggml_vk_buffer_copy_async(subctx, dst_local, 0, imported, 0, bytes);
            }
            ggml_vk_ctx_end(subctx);

            auto t0 = std::chrono::high_resolution_clock::now();
            ggml_vk_submit(subctx, dst->fence);
            VK_CHECK(dst->device.waitForFences({ dst->fence }, true, UINT64_MAX), "p2p_selftest bench");
            auto t1 = std::chrono::high_resolution_clock::now();
            dst->device.resetFences({ dst->fence });
            ggml_vk_queue_command_pools_cleanup(dst);

            double secs = std::chrono::duration<double>(t1 - t0).count();
            if (secs > 0.0) {
                gbps = (double) bytes * (double) iters / secs / 1e9;
            }
            if (out_gbps) *out_gbps = gbps;

            const double PCIE3_X16_GBPS = 16.0;
            say("  bandwidth: %.2f GB/s (%d x %zu bytes)\n", gbps, iters, bytes);
            if (gbps > PCIE3_X16_GBPS) {
                say("  => exceeds PCIe 3.0 x16 ceiling (~16 GB/s): empirically a faster\n");
                say("     link (e.g. NVLink) carried the bytes. NOT a Vulkan NVLink-API claim;\n");
                say("     the route is inferred from the measured rate, not queried.\n");
            } else {
                say("  => at/below PCIe 3.0 x16 ceiling: consistent with a PCIe route\n");
                say("     (NVLink present in topology may still exist but was not used here).\n");
            }
        } else {
            say("  loopback: bandwidth not meaningful (same-device import)\n");
        }
    } catch (const vk::SystemError & e) {
        say("  FAIL: Vulkan exception: %s\n", e.what());
        rc = -7;
    }

    // Buffers are shared_ptr (vk_buffer); they free on scope exit. An un-consumed
    // fd (only on an early failure path) must be closed to avoid a leak.
    if (fd >= 0) {
        close(fd);
    }
    return rc;
    #undef say
}

// ---------------------------------------------------------------------------
// Public C entry point (not upstream): opaque-fd P2P self-test (correctness +
// cross-device bandwidth). See ggml_vk_p2p_selftest_impl for semantics.
// ---------------------------------------------------------------------------
// `transport`: 0 = host-staging (default, portable), 1 = opaque-fd, 2 = device-group.
extern "C" int ggml_backend_vk_p2p_selftest(int src_dev, int dst_dev,
                                            size_t bytes, int iters, int transport,
                                            double * out_gbps,
                                            char * report, size_t report_size) {
    if (report && report_size) report[0] = '\0';
    vk_split_transport t = VK_SPLIT_TRANSPORT_HOST_STAGING;
    if (transport == 1) t = VK_SPLIT_TRANSPORT_OPAQUE_FD;
    else if (transport == 2) t = VK_SPLIT_TRANSPORT_DEVICE_GROUP;
    return ggml_vk_p2p_selftest_impl(src_dev, dst_dev, bytes, iters, t,
                                     out_gbps, report, report_size);
}

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
