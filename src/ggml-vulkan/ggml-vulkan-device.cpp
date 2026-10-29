#include "ggml-vulkan.h"
#include <vulkan/vulkan_core.h>

// See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
// We use VULKAN_HPP_DEFAULT_DISPATCHER, but not VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
// to avoid conflicts with applications or other libraries who might use it.
#if VK_HEADER_VERSION >= 301
namespace vk::detail { class DispatchLoaderDynamic; }
using vk::detail::DispatchLoaderDynamic;
#else
namespace vk { class DispatchLoaderDynamic; }
using vk::DispatchLoaderDynamic;
#endif
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
#define VULKAN_HPP_DEFAULT_DISPATCHER ggml_vk_default_dispatcher()

#include <vulkan/vulkan.hpp>

#if __has_include(<spirv/unified1/spirv.hpp>)
#    include <spirv/unified1/spirv.hpp>
#elif __has_include(<spirv-headers/spirv.hpp>)
#    include <spirv-headers/spirv.hpp>
#elif __has_include(<spirv.hpp>)
#    include <spirv.hpp>
#else
     // ggmlR: no SPIRV-Headers (plain distro Vulkan without the SDK) -- use our
     // own copy of the few constants the SPIR-V patches need.
#    include "ggml-vulkan-spv.h"
#endif

#include <algorithm>
// The Vulkan backend is one translation unit: this file is the root, and
// ggml-vulkan.cpp #includes the rest on top of it. memcpy/assert are used
// throughout those parts and used to arrive transitively through the C++
// headers below -- clang 23 no longer provides them that way, so they are
// requested explicitly here rather than in each part.
#include <cassert>
#include <cstring>
#include <cmath>
#include <cstdarg>  // ggmlR TP: va_list/vsnprintf for device-group diagnostic report
#include <iomanip>
#include <iostream>

// R package: redirect std::cerr to a no-op stream to avoid CRAN NOTE
// about '_ZSt4cerr' symbol in compiled code.
// std::cerr -> std::r_vk_null_ostream() via macro on 'cerr' token.
#if defined(GGML_R_PACKAGE)
#include <streambuf>
namespace std {
    inline std::ostream & r_vk_null_ostream() {
        struct null_buf : std::streambuf {
            int overflow(int c) override { return c; }
        };
        static null_buf buf;
        static std::ostream stream(&buf);
        return stream;
    }
}
#define cerr r_vk_null_ostream()
#endif

#include <tuple>
#include <vector>
#include <deque>
#include <sstream>
#include <utility>
#include <memory>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <thread>

#if defined(_MSC_VER)
# define NOMINMAX 1
# include <windows.h>
# define YIELD() YieldProcessor()
#elif defined(__clang__) || defined(__GNUC__)
# if defined(__x86_64__) ||defined(__i386__)
#  include <immintrin.h>
#  define YIELD() _mm_pause()
# elif defined(__arm__) || defined(__aarch64__)
#  if defined(__clang__)
#   include <arm_acle.h>
#   define YIELD() __yield()
#  else
#   define YIELD() asm volatile("yield")
#  endif
# endif
#endif

#if !defined(YIELD)
#define YIELD()
#endif

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

// ggmlR TP: crash-survivable diagnostic logger (GGMLR_TP_TRACE=path). Uses
// fopen/fwrite/fclose (NOT macro-redirected in this TU), so lines survive a
// segfault/abort during teardown. No-op when the env var is unset. See
// r_dbg_filelog.h. Used to localize P2P/teardown crashes; keep the trace points
// but they stay silent in normal use.
#include "../r_dbg_filelog.h"

#include "ggml-vulkan-shaders.hpp"

// R package: re-redirect exit() after <cstdlib> undoes the r_ggml_compat.h macro
#if defined(GGML_R_PACKAGE) && !defined(R_GGML_IO_IMPL)
#undef exit
#define exit(status) r_ggml_exit(status)
#endif

// remove this once it's more widely available in the SDK
#if !defined(VK_KHR_shader_bfloat16)

#define VK_KHR_shader_bfloat16 1
#define VK_KHR_SHADER_BFLOAT16_SPEC_VERSION                          1
#define VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME                        "VK_KHR_shader_bfloat16"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR ((VkStructureType)1000141000)
#define VK_COMPONENT_TYPE_BFLOAT16_KHR                               ((VkComponentTypeKHR)1000141000)

typedef struct VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    VkStructureType                       sType;
    void*                                 pNext;
    VkBool32                              shaderBFloat16Type;
    VkBool32                              shaderBFloat16DotProduct;
    VkBool32                              shaderBFloat16CooperativeMatrix;
} VkPhysicalDeviceShaderBfloat16FeaturesKHR;
#endif

#ifndef VK_NV_cooperative_matrix_decode_vector
#define VK_NV_cooperative_matrix_decode_vector 1
#define VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME "VK_NV_cooperative_matrix_decode_vector"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV ((VkStructureType)1000689000)
typedef struct VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           cooperativeMatrixDecodeVector;
} VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV;

#define VK_VALVE_shader_mixed_float_dot_product 1
#define VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_SPEC_VERSION 1
#define VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_EXTENSION_NAME "VK_VALVE_shader_mixed_float_dot_product"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE ((VkStructureType)1000673000)
typedef struct VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           shaderMixedFloatDotProductFloat16AccFloat32;
    VkBool32           shaderMixedFloatDotProductFloat16AccFloat16;
    VkBool32           shaderMixedFloatDotProductBFloat16Acc;
    VkBool32           shaderMixedFloatDotProductFloat8AccFloat32;
} VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE;
#endif

#if !defined(VK_EXT_shader_ocp_microscaling_types)
#define VK_EXT_shader_ocp_microscaling_types 1
#define VK_EXT_SHADER_OCP_MICROSCALING_TYPES_SPEC_VERSION 1
#define VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME "VK_EXT_shader_ocp_microscaling_types"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT ((VkStructureType)1000672000)
typedef struct VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           shaderFloat4;
    VkBool32           shaderFloat6;
    VkBool32           shaderFloat8UnsignedE8M0;
    VkBool32           shaderMXInt8;
} VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT;
#endif

#if !defined(VK_EXT_shader_float8)
#define VK_EXT_shader_float8 1
#define VK_EXT_SHADER_FLOAT8_SPEC_VERSION 1
#define VK_EXT_SHADER_FLOAT8_EXTENSION_NAME "VK_EXT_shader_float8"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT ((VkStructureType)1000567000)
typedef struct VkPhysicalDeviceShaderFloat8FeaturesEXT {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           shaderFloat8;
    VkBool32           shaderFloat8CooperativeMatrix;
} VkPhysicalDeviceShaderFloat8FeaturesEXT;
#endif

#ifndef VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME
#define VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME "VK_KHR_internally_synchronized_queues"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR ((VkStructureType)1000504000)
#define VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR ((VkDeviceQueueCreateFlagBits)0x00000004)

// Compile-time constant guaranteed; no runtime initialization overhead
static constexpr vk::DeviceQueueCreateFlagBits eInternallySynchronizedKHR =
    static_cast<vk::DeviceQueueCreateFlagBits>(0x00000004);

typedef struct VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR {
    VkStructureType    sType;
    void* pNext;
    VkBool32           internallySynchronizedQueues;
} VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR;
#else
static constexpr vk::DeviceQueueCreateFlagBits eInternallySynchronizedKHR = vk::DeviceQueueCreateFlagBits::eInternallySynchronizedKHR;
#endif

#define ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))

#define ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))
// No intermediate M + N - 1: with N taken from a device limit (RADV reports
// maxComputeWorkGroupCount[0] = UINT32_MAX) that sum wraps and the result is 0.
template <typename M, typename N>
static inline auto ggml_vk_ceil_div(M m, N n) {
    GGML_ASSERT(n != 0 && "CEIL_DIV by zero");
    return m / n + (m % n != 0);
}
#define CEIL_DIV(M, N) ggml_vk_ceil_div((M), (N))
static bool is_pow2(uint32_t x) { return x > 1 && (x & (x-1)) == 0; }

#define VK_VENDOR_ID_AMD 0x1002
#define VK_VENDOR_ID_APPLE 0x106b
#define VK_VENDOR_ID_INTEL 0x8086
#define VK_VENDOR_ID_NVIDIA 0x10de

#define VK_VENDOR_ID_QUALCOMM 0x5143

#define VK_VENDOR_ID_SAMSUNG 0x144d

#define VK_DEVICE_DESCRIPTOR_POOL_SIZE 256

#define GGML_VK_MAX_NODES 8192

// Upstream 0.27 passes the device as a 3rd argument (device-lost report, then throw);
// ggmlR keeps GGML_ABORT and ignores it: a C++ exception must not cross .Call.
#define VK_CHECK(err, msg, ...)                                     \
    do {                                                            \
        vk::Result err_ = (err);                                    \
        if (err_ != vk::Result::eSuccess) {                         \
            /* GGML_ABORT, not exit(1): both reach R as an error, but  \
             * this one carries the Vulkan result into the message. */ \
            GGML_ABORT("ggml_vulkan: %s error %s",                  \
                #err, to_string(err_).c_str());                     \
        }                                                           \
    } while (0)

#define VK_LOG_DEBUG(msg) ((void) 0)

struct ggml_backend_vk_context;

struct vk_semaphore {
    vk::Semaphore s;
    uint64_t value;
};

#define MAX_PARAMETER_COUNT 12
// Max number of adds that can be fused without exceeding MAX_PARAMETER_COUNT.
#define MAX_FUSED_ADDS (MAX_PARAMETER_COUNT - 3)

typedef std::shared_ptr<struct vk_pipeline_struct> vk_pipeline;

struct vk_pipeline_struct {
    std::string name;
    vk::ShaderModule shader_module;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t push_constant_size;
    uint32_t parameter_count;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
    // true if fields have been set by ggml_vk_create_pipeline
    bool initialized {};
    // true while a compile is in flight, used to dedupe concurrent claims.
    // Protected by device->compile_mutex.
    bool compile_pending {};
    // set to true when the shader has been compiled
    std::atomic<bool> compiled {};
    // number of registers used, extracted from pipeline executable properties
    uint32_t register_count {};

#if defined(VK_EXT_shader_64bit_indexing)
    bool is_64b_indexing {};
#endif
    // linked list of pipelines for multiple compilation variants.
    // currently only used to compile a 64-bit indexing variant.
    vk_pipeline next;
};

typedef std::weak_ptr<vk_pipeline_struct> vk_pipeline_ref;

// ggml 0.27 keeps every matmul pipeline in one map. ggmlR adds two key fields
// for its own variants (both default to false, so upstream's brace-initialised
// keys still mean what they meant upstream):
//   prec32     - f32 x f32 with exact f32 shared-memory tiles, selected when
//                the node asks for GGML_PREC_F32 (on fp16/coopmat devices the
//                regular f32 pipeline keeps its tiles in f16).
//   sg_shuffle - q8_1 MMQ built with the subgroup-shuffle (no shared memory)
//                variant, used for Q4_K/Q5_K/Q6_K on wave64 devices.
struct vk_matmul_pipeline_key {
    ggml_type type_a;
    ggml_type type_b;
    bool mul_mat_id;
    bool f16acc;
    bool prec32     = false;
    bool sg_shuffle = false;

    bool operator<(const vk_matmul_pipeline_key & o) const {
        return std::tie(type_a, type_b, mul_mat_id, f16acc, prec32, sg_shuffle)
             < std::tie(o.type_a, o.type_b, o.mul_mat_id, o.f16acc, o.prec32, o.sg_shuffle);
    }
};

struct vk_matmul_pipeline_pair {
    vk_pipeline unaligned;
    vk_pipeline aligned;
    uint32_t align;
};

struct vk_tile_config {
    std::vector<uint32_t> warptile;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
};

using matmul_tile_selector_t = std::function<uint32_t(
    uint32_t m, uint32_t n, uint32_t k, uint32_t shader_core_count,
    const std::vector<vk_matmul_pipeline_pair>& configs)>;

struct vk_device_struct;

static void ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);

struct vk_matmul_pipeline_struct {
    vk_pipeline l, m, s;
    vk_pipeline a_l, a_m, a_s;
    // Returns true when all unaligned pipelines are null.
    // We only check for unaligned variants since one of the unaligned pipelines must exist
    // while aligned pipelines are optional
    bool is_empty() const {
        return l == nullptr && m == nullptr && s == nullptr;
    }
};
typedef std::shared_ptr<vk_matmul_pipeline_struct> vk_matmul_pipeline;

struct vk_matmul_pipeline2 {
    vk_matmul_pipeline2() {
        f16acc = std::make_shared<vk_matmul_pipeline_struct>();
        f32acc = std::make_shared<vk_matmul_pipeline_struct>();
    }
    vk_matmul_pipeline f32acc;
    vk_matmul_pipeline f16acc;
};

struct vk_device_struct;
typedef std::shared_ptr<vk_device_struct> vk_device;
typedef std::weak_ptr<vk_device_struct> vk_device_ref;

struct vk_buffer_struct;
typedef std::shared_ptr<vk_buffer_struct> vk_buffer;
typedef std::weak_ptr<vk_buffer_struct> vk_buffer_ref;

struct ggml_backend_vk_buffer_type_context {
    std::string name;
    vk_device device;
};

struct vk_queue;

struct vk_command_buffer {
    vk::CommandBuffer buf;
    uint64_t use_counter = 0;
    bool in_use = false;
};

// Stores command pool/buffers. There's an instance of this
// for each (context,queue) pair and for each (device,queue) pair.
struct vk_command_pool {
    void init(vk_device& device, vk_queue *q_);
    void destroy(vk::Device& device);

    vk::CommandPool pool;
    // Using deque so the pointers to command buffers
    // remain valid even if we add more
    std::deque<vk_command_buffer> cmd_buffers;

    vk_queue *q;

    size_t buffers_in_use() const {
        return std::count_if(cmd_buffers.begin(), cmd_buffers.end(),
            [](const auto& cb) { return cb.in_use; });
    }
};

struct vk_queue_handle {
    vk::Queue queue;
    vk_device_ref device;
    std::mutex * device_submit_mutex = nullptr;
    virtual void submit(vk::ArrayProxy<const vk::SubmitInfo> submits, vk::Fence fence) = 0;
    virtual void lock()   {}   // no-op by default (internally synchronized case)
    virtual void unlock() {}
    virtual ~vk_queue_handle() = default;
};

struct vk_queue_handle_synchronized : vk_queue_handle {
    std::mutex mutex;
    void submit(vk::ArrayProxy<const vk::SubmitInfo> submits, vk::Fence fence) override;

    void lock()   override { mutex.lock(); }
    void unlock() override { mutex.unlock(); }
};

struct vk_queue_handle_unsynchronized : vk_queue_handle {
    void submit(vk::ArrayProxy<const vk::SubmitInfo> submits, vk::Fence fence) override;

    // lock()/unlock() inherited no-ops
};

struct vk_queue {
    uint32_t queue_family_index;
    std::shared_ptr<vk_queue_handle> handle;

    vk_command_pool cmd_pool;

    vk::PipelineStageFlags stage_flags;

    bool transfer_only;
};

static const char * ggml_backend_vk_buffer_type_name(ggml_backend_buffer_type_t buft);
static ggml_backend_buffer_t ggml_backend_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
static size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft);
static size_t ggml_backend_vk_buffer_type_get_max_size(ggml_backend_buffer_type_t buft);
static size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor);
static ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ NULL,
};

class vk_perf_logger;
static void ggml_vk_destroy_buffer(vk_buffer& buf);
static void ggml_vk_synchronize(ggml_backend_vk_context * ctx);

static constexpr uint32_t mul_mat_vec_max_cols = 8;
static constexpr uint32_t p021_max_gqa_ratio = 8;

enum vk_device_architecture {
    OTHER,
    AMD_GCN,
    AMD_RDNA1,
    AMD_RDNA2,
    AMD_RDNA3,
    AMD_RDNA4,
    INTEL_XE1,
    INTEL_XE2,
    NVIDIA_PRE_TURING,
    NVIDIA_TURING,
    QUALCOMM_ADRENO,
};

static vk_device_architecture get_device_architecture(const vk::PhysicalDevice& device) {
    vk::PhysicalDeviceProperties props = device.getProperties();

    if (props.vendorID == VK_VENDOR_ID_AMD) {
        const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        bool amd_shader_core_properties = false;
        bool integer_dot_product = false;
        bool subgroup_size_control = false;

        for (const auto& properties : ext_props) {
            if (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
                amd_shader_core_properties = true;
            } else if (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_product = true;
            } else if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_size_control = true;
            }
        }

        if (!amd_shader_core_properties || !integer_dot_product || !subgroup_size_control) {
            return vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
        vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR integer_dot_props;
        vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;

        props2.pNext = &shader_core_props_amd;
        shader_core_props_amd.pNext = &integer_dot_props;
        integer_dot_props.pNext = &subgroup_size_control_props;

        device.getProperties2(&props2);

        if (subgroup_size_control_props.maxSubgroupSize == 64 && subgroup_size_control_props.minSubgroupSize == 64) {
            return vk_device_architecture::AMD_GCN;
        }
        if (subgroup_size_control_props.maxSubgroupSize == 64 && subgroup_size_control_props.minSubgroupSize == 32) {
            // RDNA1/2/3/4: distinguished by wavefrontsPerSimd
            if (shader_core_props_amd.wavefrontsPerSimd == 20) {
                return vk_device_architecture::AMD_RDNA1;
            }
            if (shader_core_props_amd.wavefrontsPerSimd == 16) {
                // RDNA4 (GFX12xx): 16 wavefronts per SIMD
                return vk_device_architecture::AMD_RDNA4;
            }
            if (integer_dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated) {
                return vk_device_architecture::AMD_RDNA3;
            }
            return vk_device_architecture::AMD_RDNA2;
        }
    } else if (props.vendorID == VK_VENDOR_ID_INTEL) {
        const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        bool subgroup_size_control = false;
        bool integer_dot_product = false;

        for (const auto& properties : ext_props) {
            if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_size_control = true;
            } else if (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_product = true;
            }
        }

        if (!subgroup_size_control || !integer_dot_product) {
            return vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;
        vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR integer_dot_props;

        props2.pNext = &subgroup_size_control_props;
        subgroup_size_control_props.pNext = &integer_dot_props;
        device.getProperties2(&props2);

        if (subgroup_size_control_props.minSubgroupSize == 16) {
            // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
            // Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
            // https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
            // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
            return vk_device_architecture::INTEL_XE2;
        } else if (subgroup_size_control_props.minSubgroupSize == 8 &&
                 integer_dot_product && integer_dot_props.integerDotProduct4x8BitPackedSignedAccelerated) {
            return vk_device_architecture::INTEL_XE1;
        }
    } else if (props.vendorID == VK_VENDOR_ID_NVIDIA) {
        const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        bool cooperative_matrix = false;
        bool sm_builtins = false;

        // Detect "pre-turing" based on lack of coopmat support.
        for (const auto& properties : ext_props) {
            if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_matrix = true;
            } else if (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = true;
            }
        }

        if (!cooperative_matrix) {
            return vk_device_architecture::NVIDIA_PRE_TURING;
        }

        if (sm_builtins) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;

            props2.pNext = &sm_props;

            device.getProperties2(&props2);

            // Turing has 32, following architectures have 48
            if (sm_props.shaderWarpsPerSM == 32) {
                return vk_device_architecture::NVIDIA_TURING;
            }
        }
    } else if(props.vendorID == VK_VENDOR_ID_QUALCOMM){
        const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        bool cooperative_matrix = false;
        bool cooperative_matrix_conversion = false;

        for (const auto& properties : ext_props) {
            if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_matrix = true;
            } else if (strcmp("VK_QCOM_cooperative_matrix_conversion", properties.extensionName) == 0) {
                cooperative_matrix_conversion = true;
            }
        }

        if (cooperative_matrix && cooperative_matrix_conversion) {
            return vk_device_architecture::QUALCOMM_ADRENO;
        }
    }
    return vk_device_architecture::OTHER;
}

static uint32_t ggml_vk_intel_shader_core_count(const vk::PhysicalDevice& vkdev) {
    VkPhysicalDeviceProperties2 props = vkdev.getProperties2();

    if (props.properties.vendorID != VK_VENDOR_ID_INTEL) {
        return 0;
    }

    const uint32_t device_id = props.properties.deviceID;

    switch (device_id) {
    case 0x56A6:  // A310
        return 6;
    case 0x5693:  // A370M
    case 0x56A5:  // A380
    case 0x56B1:  // Pro A40/A50
        return 8;
    case 0x5697:  // A530M
        return 12;
    case 0x5692:  // A550M
    case 0x56B3:  // Pro A60
        return 16;
    case 0x56A2:  // A580
        return 24;
    case 0x5691:  // A730M
    case 0x56A1:  // A750
        return 28;
    case 0x56A0:  // A770
    case 0x5690:  // A770M
        return 32;
    case 0xE212:  // Pro B50
        return 16;
    case 0xE20C:  // B570
        return 18;
    case 0xE20B:  // B580
    case 0xE211:  // Pro B60
    case 0xE222:  // Pro B65
        return 20;
    case 0xE223:  // Pro B70
        return 32;
    case 0xB080:  // PTL Xe3 LPG 2x6 (12 subslices)
        return 12;
    default:
        return 0;
    }
}

enum vk_conv_shapes {
    CONV_SHAPE_128x128,
    CONV_SHAPE_64x32,
    CONV_SHAPE_32x256,
    CONV_SHAPE_64x128,
    CONV_SHAPE_COUNT,
};

struct vk_conv_block_size {
    uint32_t K;
    uint32_t NPQ;
    uint32_t CRS;
};

vk_conv_block_size vk_conv_block_sizes[CONV_SHAPE_COUNT] = {
    // K   NPQ  CRS
    { 128, 128, 16 }, // CONV_SHAPE_128x128
    {  64,  32, 32 }, // CONV_SHAPE_64x32
    {  32, 256, 16 }, // CONV_SHAPE_32x256
    {  64, 128, 16 }, // CONV_SHAPE_64x128
};

enum dmmv_wg_sizes {
    DMMV_WG_SIZE_SUBGROUP,
    DMMV_WG_SIZE_LARGE,
    DMMV_WG_SIZE_COUNT,
};

enum FaCodePath {
    FA_SCALAR,
    FA_COOPMAT1,
    FA_COOPMAT2,
};

struct vk_fa_tuning_params {
    FaCodePath path;
    uint32_t workgroup_size;
    uint32_t subgroup_size;
    uint32_t block_rows;
    uint32_t block_cols;
    uint32_t d_split;
    uint32_t row_split;
    bool shmem_staging;
    bool disable_subgroups;
    uint32_t limit_occupancy_shmem;
};

struct vk_fa_pipeline_state {
    uint32_t HSK, HSV;
    uint32_t Br, Bc;
    uint32_t D_split, row_split;
    bool shmem_staging;
    FaCodePath path;
    uint32_t workgroup_size, subgroup_size;
    bool aligned;
    bool f32acc;
    uint32_t flags;
    uint32_t limit_occupancy_shmem;
    ggml_type k_type;
    ggml_type v_type;

    bool operator<(const vk_fa_pipeline_state &b) const {
        return std::tie(HSK, HSV, Br, Bc, D_split, row_split, shmem_staging, path, workgroup_size, subgroup_size, aligned, f32acc, flags, limit_occupancy_shmem, k_type, v_type) <
               std::tie(b.HSK, b.HSV, b.Br, b.Bc, b.D_split, b.row_split, b.shmem_staging, b.path, b.workgroup_size, b.subgroup_size, b.aligned, b.f32acc, b.flags, b.limit_occupancy_shmem, b.k_type, b.v_type);
    }
};

struct vk_conv2d_pipeline_state {
    vk_conv2d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t p0, uint32_t p1, uint32_t d0, uint32_t d1, uint32_t KW, uint32_t KH, uint32_t aligned)
        : s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), KW(KW), KH(KH), aligned(aligned) {}

    uint32_t s0, s1, p0, p1, d0, d1, KW, KH;
    // when set, shader can skip K/CRS/NPQ bounds checks and address clamps
    uint32_t aligned;

    bool operator<(const vk_conv2d_pipeline_state &b) const {
        return std::tie(s0, s1, p0, p1, d0, d1, KW, KH, aligned) <
               std::tie(b.s0, b.s1, b.p0, b.p1, b.d0, b.d1, b.KW, b.KH, b.aligned);
    }
};

struct vk_conv3d_pipeline_state {
    vk_conv3d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t p0, uint32_t p1, uint32_t p2,
                             uint32_t d0, uint32_t d1, uint32_t d2, uint32_t KW, uint32_t KH, uint32_t KD, uint32_t aligned)
        : s0(s0), s1(s1), s2(s2), p0(p0), p1(p1), p2(p2), d0(d0), d1(d1), d2(d2), KW(KW), KH(KH), KD(KD), aligned(aligned) {}

    uint32_t s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD;
    uint32_t aligned;

    bool operator<(const vk_conv3d_pipeline_state &b) const {
        return std::tie(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned) <
               std::tie(b.s0, b.s1, b.s2, b.p0, b.p1, b.p2, b.d0, b.d1, b.d2, b.KW, b.KH, b.KD, b.aligned);
    }
};

struct vk_solve_tri_pipeline_state {
    vk_solve_tri_pipeline_state(uint32_t N, uint32_t K)
        : N(N), K(K) {}

    uint32_t N, K;

    bool operator<(const vk_solve_tri_pipeline_state &b) const {
        return std::tie(N, K) <
               std::tie(b.N, b.K);
    }
};

enum shader_reduction_mode {
    SHADER_REDUCTION_MODE_SHMEM,
    SHADER_REDUCTION_MODE_HYBRID,
    SHADER_REDUCTION_MODE_SUBGROUP,
    SHADER_REDUCTION_MODE_COUNT,
};

// argsort pipelines for up to 1<<10 invocations per workgroup
static constexpr uint32_t num_argsort_pipelines = 11;
static constexpr uint32_t num_topk_moe_pipelines = 10;
static constexpr uint32_t num_topk_pipelines = 11;

static constexpr std::initializer_list<ggml_op> topk_moe_early_softmax_norm{ GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                                                             GGML_OP_SUM_ROWS, GGML_OP_CLAMP,    GGML_OP_DIV,
                                                                             GGML_OP_RESHAPE };

static constexpr std::initializer_list<ggml_op> topk_moe_sigmoid_norm_bias{ GGML_OP_UNARY,    GGML_OP_RESHAPE,  GGML_OP_ADD,
                                                                            GGML_OP_ARGSORT,  GGML_OP_VIEW,     GGML_OP_GET_ROWS,
                                                                            GGML_OP_RESHAPE,  GGML_OP_SUM_ROWS, GGML_OP_CLAMP,
                                                                            GGML_OP_DIV,      GGML_OP_RESHAPE };

static constexpr std::initializer_list<ggml_op> topk_moe_sqrt_softplus_norm_bias{ GGML_OP_UNARY,    GGML_OP_SQRT,
                                                                                  GGML_OP_RESHAPE,  GGML_OP_ADD,
                                                                                  GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                                  GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                                                                  GGML_OP_SUM_ROWS, GGML_OP_CLAMP,
                                                                                  GGML_OP_DIV,      GGML_OP_RESHAPE };

static constexpr std::initializer_list<ggml_op> topk_moe_early_softmax     { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_ROWS };

static constexpr std::initializer_list<ggml_op> topk_moe_late_softmax      { GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                             GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                                                             GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

static constexpr std::initializer_list<ggml_op> snake_pattern              { GGML_OP_MUL,      GGML_OP_SIN,
                                                                             GGML_OP_SQR,      GGML_OP_MUL,
                                                                             GGML_OP_ADD };

static constexpr std::initializer_list<ggml_op> topk_qsa_pattern { GGML_OP_GET_ROWS, GGML_OP_PERMUTE,
                                                                   GGML_OP_CONT,     GGML_OP_CPY,
                                                                   GGML_OP_RESHAPE,  GGML_OP_ADD,
                                                                   GGML_OP_TOP_K };

static constexpr std::initializer_list<std::array<int, 3>> topk_qsa_edges {
    { 1, 0, 0 }, // permute->src[0] == get_rows
    { 2, 0, 1 }, // cont->src[0]    == permute
    { 4, 0, 3 }, // reshape->src[0] == cpy (mask cast)
    { 5, 0, 2 }, // add->src[0]     == cont
    { 5, 1, 4 }, // add->src[1]     == reshape
    { 6, 0, 5 }, // top_k->src[0]   == add
};

static constexpr std::initializer_list<ggml_op> rms_norm_mul_add_mul_pattern { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_MUL };

static constexpr std::initializer_list<ggml_op> rms_norm_mul_add_pattern     { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };

static constexpr std::initializer_list<ggml_op> rms_norm_mul_rope_view_set_rows_pattern { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

static constexpr std::initializer_list<ggml_op> rms_norm_view_set_rows_pattern { GGML_OP_RMS_NORM, GGML_OP_VIEW, GGML_OP_SET_ROWS };

static constexpr std::initializer_list<ggml_op> rope_view_set_rows_pattern { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };
// scale_out*sigmoid(scale_in*x) as the hc_post weights (qwen4exp hc_combine)
static constexpr std::initializer_list<ggml_op> hc_post_gate_pattern { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

static constexpr std::initializer_list<std::array<int, 3>> hc_post_gate_edges {
    { 1, 0, 0 }, // sigmoid->src[0] == scale
    { 2, 0, 1 }, // scale->src[0]   == sigmoid
    { 3, 2, 2 }, // hc_post->src[2] == scale (post)
};

static constexpr std::array<ggml_type, 9> lightning_indexer_k_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_0,
    GGML_TYPE_IQ4_NL,
};

static bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type) {
    return std::find(lightning_indexer_k_types.begin(), lightning_indexer_k_types.end(), type) != lightning_indexer_k_types.end();
}

static constexpr std::initializer_list<std::array<int, 3>> topk_moe_early_softmax_norm_edges {
    { 1, 0, 0 }, // reshape->src[0]  == softmax
    { 2, 0, 0 }, // argsort->src[0]  == softmax
    { 3, 0, 2 }, // view->src[0]     == argsort
    { 4, 0, 1 }, // get_rows->src[0] == reshape
    { 4, 1, 3 }, // get_rows->src[1] == view
    { 5, 0, 4 }, // reshape->src[0]  == get_rows
    { 6, 0, 5 }, // sum_rows->src[0] == reshape
    { 7, 0, 6 }, // clamp->src[0]    == sum_rows
    { 8, 0, 5 }, // div->src[0]      == reshape
    { 8, 1, 7 }, // div->src[1]      == clamp
    { 9, 0, 8 }, // reshape->src[0]  == div
};

static constexpr std::initializer_list<std::array<int, 3>> topk_moe_sigmoid_norm_bias_edges {
    { 1, 0, 0 }, // reshape->src[0]  == sigmoid
    { 2, 0, 0 }, // add->src[0]      == sigmoid
    { 3, 0, 2 }, // argsort->src[0]  == add
    { 4, 0, 3 }, // view->src[0]     == argsort
    { 5, 0, 1 }, // get_rows->src[0] == reshape
    { 5, 1, 4 }, // get_rows->src[1] == view
    { 6, 0, 5 }, // reshape->src[0]  == get_rows
    { 7, 0, 6 }, // sum_rows->src[0] == reshape
    { 8, 0, 7 }, // clamp->src[0]    == sum_rows
    { 9, 0, 6 }, // div->src[0]      == reshape
    { 9, 1, 8 }, // div->src[1]      == clamp
    {10, 0, 9 }, // reshape->src[0]  == div
};

static constexpr std::initializer_list<std::array<int, 3>> topk_moe_sqrt_softplus_norm_bias_edges {
    { 1, 0, 0 }, // sqrt->src[0]     == softplus
    { 2, 0, 1 }, // reshape->src[0]  == sqrt
    { 3, 0, 1 }, // add->src[0]      == sqrt
    { 4, 0, 3 }, // argsort->src[0]  == add
    { 5, 0, 4 }, // view->src[0]     == argsort
    { 6, 0, 2 }, // get_rows->src[0] == reshape
    { 6, 1, 5 }, // get_rows->src[1] == view
    { 7, 0, 6 }, // reshape->src[0]  == get_rows
    { 8, 0, 7 }, // sum_rows->src[0] == reshape
    { 9, 0, 8 }, // clamp->src[0]    == sum_rows
    {10, 0, 7 }, // div->src[0]      == reshape
    {10, 1, 9 }, // div->src[1]      == clamp
    {11, 0,10 }, // reshape->src[0]  == div
};

static constexpr std::initializer_list<std::array<int, 3>> topk_moe_early_softmax_edges {
    { 1, 0, 0 }, // reshape->src[0]  == softmax
    { 2, 0, 0 }, // argsort->src[0]  == softmax
    { 3, 0, 2 }, // view->src[0]     == argsort
    { 4, 0, 1 }, // get_rows->src[0] == reshape
    { 4, 1, 3 }, // get_rows->src[1] == view
};

static constexpr std::initializer_list<std::array<int, 3>> topk_moe_late_softmax_edges {
    { 1, 0, 0 }, // view->src[0]     == argsort
    { 2, 1, 1 }, // get_rows->src[1] == view
    { 3, 0, 2 }, // reshape->src[0]  == get_rows
    { 4, 0, 3 }, // soft_max->src[0] == reshape
    { 5, 0, 4 }, // reshape->src[0]  == soft_max
};

enum topk_moe_mode {
    TOPK_MOE_EARLY_SOFTMAX,
    TOPK_MOE_EARLY_SOFTMAX_NORM,
    TOPK_MOE_LATE_SOFTMAX,
    TOPK_MOE_SIGMOID_NORM_BIAS,
    TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS,
    TOPK_MOE_COUNT,
};

enum rms_norm_mode {
    RMS_NORM_MUL,
    RMS_NORM_MUL_ADD,
    RMS_NORM_MUL_ADD_MUL,
    RMS_NORM_MUL_ROPE,
    RMS_NORM_MUL_ROPE_VIEW_SET_ROWS,
    RMS_NORM_VIEW_SET_ROWS,
    RMS_NORM_COUNT,
};

static constexpr std::initializer_list<std::array<int, 3>> rope_view_set_rows_edges {
    { 1, 0, 0 }, // view->src[0]     == rope
    { 2, 0, 1 }, // set_rows->src[0] == view
};

static constexpr std::initializer_list<std::array<int, 3>> rms_norm_mul_rope_view_set_rows_edges {
    { 1, 0, 0 }, // mul->src[0]      == rms
    { 2, 0, 1 }, // rope->src[0]     == mul
    { 3, 0, 2 }, // view->src[0]     == rope
    { 4, 0, 3 }, // set_rows->src[0] == view
};

static constexpr std::initializer_list<std::array<int, 3>> rms_norm_view_set_rows_edges {
    { 1, 0, 0 }, // view->src[0]     == rms_norm
    { 2, 0, 1 }, // set_rows->src[0] == view
};

struct vk_device_struct {
    std::recursive_mutex mutex;
    std::mutex queue_submit_mutex;

    // Guards compile_pending, all_pipelines, and the dynamic pipeline maps
    // (flash_attn, fa_mask_opt, solve_tri, conv2d, etc). The actual compile
    // runs with no lock held, so different pipelines can compile in parallel.
    // Lock order is device->mutex -> compile_mutex, never the reverse.
    std::mutex compile_mutex;
    std::condition_variable compile_cv;

    uint32_t debug_cmdbuf_idx {};

    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties properties;
    std::string name;
    uint64_t max_memory_allocation_size;
    uint64_t max_buffer_size;
    uint64_t suballocation_block_size;
    uint64_t min_imported_host_pointer_alignment;
    bool external_memory_host {};
    // ggmlR Tensor Parallelism (P2P): opaque-fd export/import for device<->device
    // weight-split buffers. Not in upstream ggml — see TODO.md "Vulkan Tensor Parallelism".
    bool external_memory_fd {};
    bool fp16;
    bool bf16;
    bool pipeline_robustness;
    bool memory_priority;
    vk::Device device;
    uint32_t vendor_id;
    vk::DriverId driver_id;
    vk_device_architecture architecture;
    std::unique_ptr<vk_queue> compute_queue;
    std::unique_ptr<vk_queue> transfer_queue;
    bool single_queue;
    bool support_async;
    bool async_use_transfer_queue;
    bool has_internally_synchronized_queues = false;
    uint32_t subgroup_size;
    uint32_t subgroup_size_log2;
    uint32_t shader_core_count;
    bool uma;
    bool prefer_host_memory;
    bool float_controls_rte_fp16;
    bool float_controls_denorm_preserve_fp16;
    bool subgroup_basic;
    bool subgroup_arithmetic;
    bool subgroup_shuffle;
    bool subgroup_ballot;
    bool subgroup_clustered;
    bool subgroup_vote;
    bool multi_add;
    bool shader_int64;
    bool buffer_device_address;
    bool vulkan_memory_model;

    bool add_rms_fusion;
    uint32_t partials_binding_alignment;
    uint32_t max_nodes_per_submit;

    bool shader_64b_indexing;

    bool integer_dot_product;

    // ggmlR extension: VK_EXT_shader_atomic_float with shaderBufferFloat32AtomicAdd.
    // Backward kernels that accumulate into shared gradients (ssm_scan_back, and
    // scatter_elements in add mode) need it; supports_op refuses those ops rather
    // than dispatching a shader whose atomicAdd would not compile or, worse,
    // would silently drop the accumulation.
    bool atomic_float_add;
    // 0: default, 1: force mmvq, -1: disable mmvq
    int32_t mmvq_mode;

    bool subgroup_size_control;
    uint32_t subgroup_min_size;
    uint32_t subgroup_max_size;
    bool subgroup_require_full_support;
    uint32_t wavefronts_per_simd;  // AMD only, from VK_AMD_shader_core_properties

    // floor(log2(maxComputeWorkGroupInvocations))
    uint32_t max_workgroup_size_log2 {};

    bool coopmat_support;
    bool coopmat_acc_f32_support {};
    bool coopmat_acc_f16_support {};
    bool coopmat_bf16_support {};
    bool coopmat_support_16x16x16_f16acc {};
    bool coopmat_support_16x16x16_f32acc {};
    bool coopmat1_fa_support {};
    uint32_t coopmat_m;
    uint32_t coopmat_n;
    uint32_t coopmat_k;

    bool coopmat_int_support;
    uint32_t coopmat_int_m;
    uint32_t coopmat_int_n;
    uint32_t coopmat_int_k;

    bool coopmat2;
    bool coopmat2_bf16_support {};
    bool coopmat2_decode_vector;

    bool dot2_f16 {};
    bool ocp_fp4 {};

    bool pipeline_executable_properties_support {};
    bool push_descriptors {};

    bool supports_256_push_constants {};

    bool device_fault {};
    PFN_vkGetDeviceFaultInfoEXT pfn_vkGetDeviceFaultInfoEXT {};

    bool serialize_submissions {};

    const ggml_cgraph * diag_cgraph {};
    int diag_prev_start = -1;
    int diag_prev_end = -1;

    size_t idx;

    bool mul_mat_l[GGML_TYPE_COUNT];
    bool mul_mat_m[GGML_TYPE_COUNT];
    bool mul_mat_s[GGML_TYPE_COUNT];
    bool mul_mat_id_l[GGML_TYPE_COUNT];
    bool mul_mat_id_m[GGML_TYPE_COUNT];
    bool mul_mat_id_s[GGML_TYPE_COUNT];

    // Separate flags for the q8_1 (integer dot) mmq path, whose shader uses
    // a different shared-memory layout than the float matmul shaders.
    bool mul_mat_l_int[GGML_TYPE_COUNT];
    bool mul_mat_m_int[GGML_TYPE_COUNT];
    bool mul_mat_s_int[GGML_TYPE_COUNT];
    bool mul_mat_id_l_int[GGML_TYPE_COUNT];
    bool mul_mat_id_m_int[GGML_TYPE_COUNT];
    bool mul_mat_id_s_int[GGML_TYPE_COUNT];

    vk::DescriptorSetLayout dsl;

    std::map<vk_matmul_pipeline_key, std::vector<vk_matmul_pipeline_pair>> pipeline_matmul;
    matmul_tile_selector_t matmul_tile_selector;
    matmul_tile_selector_t matmul_id_tile_selector;

    vk_pipeline pipeline_matmul_split_k_reduce;
    vk_pipeline pipeline_quantize_q8_1_x4;

    vk_pipeline pipeline_dequant[GGML_TYPE_COUNT];
    vk_pipeline pipeline_dequant_transpose[GGML_TYPE_COUNT]; // fused dequant+transpose for FA quant-KV
    vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][mul_mat_vec_max_cols];
    vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][mul_mat_vec_max_cols];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT];

    vk_pipeline pipeline_dequant_mul_mat_vec_q8_1_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][mul_mat_vec_max_cols];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_q8_1_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT];

    vk_pipeline pipeline_mul_mat_vec_p021_f16_f32[p021_max_gqa_ratio];
    vk_pipeline pipeline_mul_mat_vec_nc_f16_f32;
    vk_pipeline pipeline_get_rows[GGML_TYPE_COUNT];
    vk_pipeline pipeline_get_rows_f32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_acc_f32;
    vk_pipeline pipeline_set_f32;
    vk_pipeline pipeline_scatter_elements_none;
    vk_pipeline pipeline_scatter_elements_add;

    // [src0 0=fp32,1=fp16][src1 0=fp32,1=fp16][dst 0=fp32,1=fp16]
    vk_pipeline pipeline_add[2][2][2];
    vk_pipeline pipeline_add_norepeat[2][2][2];
    vk_pipeline pipeline_sub[2][2][2];
    vk_pipeline pipeline_sub_norepeat[2][2][2];
    vk_pipeline pipeline_mul[2][2][2];
    vk_pipeline pipeline_mul_norepeat[2][2][2];
    vk_pipeline pipeline_div[2][2][2];
    vk_pipeline pipeline_div_norepeat[2][2][2];
    vk_pipeline pipeline_add_rms[2][2][2];
    vk_pipeline pipeline_add_rms_norepeat[2][2][2];

    // indexed by num_additional_fused_ops == num_adds - 1
    vk_pipeline pipeline_multi_add[MAX_FUSED_ADDS];
    vk_pipeline pipeline_multi_add_rms[MAX_FUSED_ADDS];

    vk_pipeline pipeline_add_id_f32;

    vk_pipeline pipeline_concat_i8, pipeline_concat_i16, pipeline_concat_i32, pipeline_concat_i64;
    vk_pipeline pipeline_upscale_nearest_f32, pipeline_upscale_bilinear_f32, pipeline_upscale_bicubic_f32, pipeline_upscale_bilinear_antialias_f32;
    vk_pipeline pipeline_scale[2];
    vk_pipeline pipeline_sqr[2];
    vk_pipeline pipeline_sqrt[2];
    vk_pipeline pipeline_sin[2];
    vk_pipeline pipeline_cos[2];
    vk_pipeline pipeline_log[2];
    vk_pipeline pipeline_tri[2];
    vk_pipeline pipeline_diag[2];
    vk_pipeline pipeline_clamp[2];
    vk_pipeline pipeline_pad_f32;
    vk_pipeline pipeline_pad_reflect_1d_f32;
    vk_pipeline pipeline_roll_f32;
    vk_pipeline pipeline_repeat_i32, pipeline_repeat_back_f32;
    vk_pipeline pipeline_repeat_i16;
    vk_pipeline pipeline_cpy_32, pipeline_cpy_f32_f16, pipeline_cpy_16, pipeline_cpy_f16_f32, pipeline_cpy_f32_bf16, pipeline_cpy_bf16_f32, pipeline_cpy_bf16_f16, pipeline_cpy_f32_i32, pipeline_cpy_i32_f32;
    vk_pipeline pipeline_contig_cpy_32, pipeline_contig_cpy_f32_f16, pipeline_contig_cpy_16, pipeline_contig_cpy_f16_f32, pipeline_contig_cpy_f32_bf16, pipeline_contig_cpy_bf16_f32, pipeline_contig_cpy_bf16_f16, pipeline_contig_cpy_f32_i32, pipeline_contig_cpy_i32_f32;
    vk_pipeline pipeline_cpy_f32_quant[GGML_TYPE_COUNT];
    vk_pipeline pipeline_cpy_quant_f32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_cpy_transpose_16, pipeline_cpy_transpose_32;
    vk_pipeline pipeline_cpy_transpose_02_16, pipeline_cpy_transpose_02_32;
    // [src0 0=fp32,1=fp16][dst]
    vk_pipeline pipeline_set_rows_i32[2][GGML_TYPE_COUNT];
    vk_pipeline pipeline_set_rows_i64[2][GGML_TYPE_COUNT];
    vk_pipeline pipeline_norm_f32;
    vk_pipeline pipeline_group_norm_f32;
    vk_pipeline pipeline_rms_norm_f32;
    vk_pipeline pipeline_rms_norm_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_add_f32;
    vk_pipeline pipeline_rms_norm_mul_add_mul_f32;
    vk_pipeline pipeline_rms_norm_mul_add_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_add_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_set_rows_f32_f32;
    vk_pipeline pipeline_rms_norm_set_rows_f32_f16;
    vk_pipeline pipeline_rms_norm_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f16;
    vk_pipeline pipeline_rms_norm_back_f32;
    // DIVERGENCE from upstream: LayerNorm backward, see ggml_norm_back().
    vk_pipeline pipeline_norm_back_f32;
    vk_pipeline pipeline_l2_norm_f32;

    // [src/dst 0=fp32,1=fp16]
    vk_pipeline pipeline_exp[2];
    vk_pipeline pipeline_expm1[2];
    vk_pipeline pipeline_gelu[2];
    vk_pipeline pipeline_gelu_erf[2];
    vk_pipeline pipeline_gelu_quick[2];
    vk_pipeline pipeline_silu[2];
    vk_pipeline pipeline_relu[2];
    vk_pipeline pipeline_elu[2];
    vk_pipeline pipeline_xielu[2];
    vk_pipeline pipeline_neg[2];
    vk_pipeline pipeline_tanh[2];
    vk_pipeline pipeline_sigmoid[2];
    vk_pipeline pipeline_hardsigmoid[2];
    vk_pipeline pipeline_hardswish[2];
    vk_pipeline pipeline_abs[2];
    vk_pipeline pipeline_softplus[2];
    vk_pipeline pipeline_step[2];
    vk_pipeline pipeline_round[2];
    vk_pipeline pipeline_round_even[2];
    vk_pipeline pipeline_ceil[2];
    vk_pipeline pipeline_floor[2];
    vk_pipeline pipeline_trunc[2];
    vk_pipeline pipeline_sgn[2];

    // fused UNARY+MUL pipelines: [op][f16][norepeat][op_on_b]
    vk_pipeline pipeline_unary_mul[4][2][2][2];

    vk_pipeline pipeline_add1_f16_f16;
    vk_pipeline pipeline_add1_f16_f32;
    vk_pipeline pipeline_add1_f32_f32;

    vk_pipeline pipeline_arange_f32;

    vk_pipeline pipeline_rel_pos_bias_f32;

    vk_pipeline pipeline_umap_sgd;
    vk_pipeline pipeline_pairwise_dist;
    vk_pipeline pipeline_roi_align;
    vk_pipeline pipeline_qmatmul_i32;
    vk_pipeline pipeline_qconv_i32;
    vk_pipeline pipeline_nms;
    vk_pipeline pipeline_knn_tiled;
    vk_pipeline pipeline_matmul_f64;
    vk_pipeline pipeline_sparse_lognorm;

    vk_pipeline pipeline_fill_f32;
    vk_pipeline pipeline_fill_f16;

    vk_pipeline pipeline_geglu[2];
    vk_pipeline pipeline_reglu[2];
    vk_pipeline pipeline_swiglu[2];
    vk_pipeline pipeline_swiglu_oai[2];
    vk_pipeline pipeline_swiglu_clamp[2];
    vk_pipeline pipeline_geglu_erf[2];
    vk_pipeline pipeline_geglu_quick[2];

    vk_pipeline pipeline_leaky_relu[2];
    vk_pipeline pipeline_silu_back_f32;
    vk_pipeline pipeline_gelu_back_f32;
    vk_pipeline pipeline_diag_mask_inf_f32;
    vk_pipeline pipeline_soft_max_f32, pipeline_soft_max_f32_f16;
    vk_pipeline pipeline_soft_max_f32_wg128, pipeline_soft_max_f32_f16_wg128;
    vk_pipeline pipeline_soft_max_f32_wg512, pipeline_soft_max_f32_f16_wg512;
    vk_pipeline pipeline_soft_max_f16, pipeline_soft_max_f16_wg128, pipeline_soft_max_f16_wg512;
    vk_pipeline pipeline_soft_max_back_f32;

    vk_pipeline pipeline_soft_max_large1_f32, pipeline_soft_max_large1_f32_f16;
    vk_pipeline pipeline_soft_max_large2_f32, pipeline_soft_max_large2_f32_f16;
    vk_pipeline pipeline_soft_max_large3_f32, pipeline_soft_max_large3_f32_f16;

    vk_pipeline pipeline_rope_norm_f32, pipeline_rope_norm_f16, pipeline_rope_norm_f32_f16;
    vk_pipeline pipeline_rope_neox_f32, pipeline_rope_neox_f16, pipeline_rope_neox_f32_f16;
    vk_pipeline pipeline_rope_multi_f32, pipeline_rope_multi_f16, pipeline_rope_multi_f32_f16;
    vk_pipeline pipeline_rope_vision_f32, pipeline_rope_vision_f16;
    vk_pipeline pipeline_argsort_f32[num_argsort_pipelines];
    vk_pipeline pipeline_argsort_large_f32[num_argsort_pipelines];
    vk_pipeline pipeline_topk_f32[num_topk_pipelines];
    vk_pipeline pipeline_sum_rows[2];
    vk_pipeline pipeline_topk_radix_f32;
    vk_pipeline pipeline_topk_radix_qsa; // qwen4 QSA indexer fusion (f16 mask)
    vk_pipeline pipeline_cross_entropy_loss_f32, pipeline_cross_entropy_loss_f32_wg512;
    vk_pipeline pipeline_fwht_f32[4];
    vk_pipeline pipeline_cumsum_f32;
    vk_pipeline pipeline_cumsum_small_f32;
    vk_pipeline pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline_cumsum_multipass2_f32;
    vk_pipeline pipeline_argmax_f32;
    vk_pipeline pipeline_count_equal_i32;
    vk_pipeline pipeline_dsv4_hc_comb_f32;
    vk_pipeline pipeline_dsv4_hc_pre_f32;
    vk_pipeline pipeline_dsv4_hc_pre_gated_f32;
    vk_pipeline pipeline_dsv4_hc_post_f32;
    vk_pipeline pipeline_dsv4_hc_post_nocomb_f32;
    vk_pipeline pipeline_lightning_indexer_f32[GGML_TYPE_COUNT];
    std::map<vk_solve_tri_pipeline_state, vk_pipeline> pipeline_solve_tri_f32;
    vk_pipeline pipeline_im2col_f32, pipeline_im2col_f32_f16;
    vk_pipeline pipeline_im2col_3d_f32, pipeline_im2col_3d_f32_f16;
    vk_pipeline pipeline_timestep_embedding_f32;
    vk_pipeline pipeline_conv_transpose_1d_f32;
    vk_pipeline pipeline_col2im_1d_f32;
    vk_pipeline pipeline_col2im_1d_f16;
    vk_pipeline pipeline_col2im_1d_bf16;
    vk_pipeline pipeline_snake_f32;
    vk_pipeline pipeline_snake_f16;
    vk_pipeline pipeline_snake_bf16;
    vk_pipeline pipeline_pool1d_f32;
    vk_pipeline pipeline_pool2d_f32;
    vk_pipeline pipeline_pool2d_subgroup_f32;
    vk_pipeline pipeline_rwkv_wkv6_f32;
    vk_pipeline pipeline_rwkv_wkv7_f32;
    vk_pipeline pipeline_gated_linear_attn_f32;
    // [size_idx][kda] where size_idx: 0=d16, 1=d32, 2=d64, 3=d128
    vk_pipeline pipeline_gated_delta_net[4][2];
    vk_pipeline pipeline_ssm_scan_f32_d128;
    vk_pipeline pipeline_ssm_scan_f32_d256;
    vk_pipeline pipeline_ssm_conv_f32;
    // ggmlR extension: backward of ssm_conv (no upstream Vulkan shader).
    vk_pipeline pipeline_ssm_conv_back_f32;
    vk_pipeline pipeline_ssm_scan_back_f32_d128;
    vk_pipeline pipeline_ssm_scan_back_f32_d256;
    vk_pipeline pipeline_out_prod_f32;   // ggmlR: backward of mul_mat
    vk_pipeline pipeline_cross_entropy_loss_back_f32;   // ggmlR
    vk_pipeline pipeline_flash_attn_back_f32;   // ggmlR: backward of flash attention
    vk_pipeline pipeline_im2col_back_f32;   // ggmlR: backward of convolution
    vk_pipeline pipeline_get_rows_back_f32;   // ggmlR: backward of embedding lookup
    // ggmlR: stage-truncated copies of the same shader, for profiling only.
    // [0] is unused; [1..4] stop after phase 1..4. Built alongside the real
    // pipeline so a profiling run needs no rebuild, and never used by the graph.
    vk_pipeline pipeline_flash_attn_back_f32_prof[5];
    vk_pipeline pipeline_ssm_conv_silu_f32;
    vk_pipeline pipeline_ssm_conv_bias_silu_f32;
    vk_pipeline pipeline_opt_step_adamw_f32;
    vk_pipeline pipeline_opt_step_sgd_f32;
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f32[CONV_SHAPE_COUNT];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f16_f32[CONV_SHAPE_COUNT];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f32[CONV_SHAPE_COUNT];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f16_f32[CONV_SHAPE_COUNT];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f32_cm1[CONV_SHAPE_COUNT];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f16_f32_cm1[CONV_SHAPE_COUNT];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f32_cm1[CONV_SHAPE_COUNT];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f16_f32_cm1[CONV_SHAPE_COUNT];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f32[CONV_SHAPE_COUNT];
    std::map<vk_conv3d_pipeline_state, vk_pipeline> pipeline_conv3d_f16_f32[CONV_SHAPE_COUNT];
    vk_pipeline pipeline_conv2d_dw_whcn_f32, pipeline_conv2d_dw_whcn_f16_f32;
    vk_pipeline pipeline_conv2d_dw_cwhn_f32, pipeline_conv2d_dw_cwhn_f16_f32;

    std::map<vk_fa_pipeline_state, vk_pipeline> pipeline_flash_attn_f32_f16;

    std::map<std::pair<uint32_t, uint32_t>, vk_pipeline> pipeline_fa_mask_opt;

    vk_pipeline pipeline_fa_sparse_compact;
    vk_pipeline pipeline_fa_sparse_compact_subgroup;
    bool fa_sparse_compact_use_subgroups;

    vk_pipeline pipeline_flash_attn_split_k_reduce;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, std::pair<vk_pipeline, vk_pipeline>> pipeline_xe_fa_decode_dual_phases;
    vk_pipeline pipeline_count_experts;

    // [2] is for whether to take n_experts from spec constant (0) or push constant (1)
    vk_pipeline pipeline_topk_moe[num_topk_moe_pipelines][2];

    std::vector<vk_pipeline_ref> all_pipelines;

    vk::Fence fence;
    vk_buffer sync_staging;

    ggml_backend_buffer_type buffer_type;

    bool disable_fusion;
    bool disable_descriptor_reuse;
    std::atomic<uint64_t> buffer_destroy_count {};
    bool disable_host_visible_vidmem;
    bool allow_sysmem_fallback;
    bool disable_graph_optimize;


    ~vk_device_struct() {
        // the build failed before the device was created, nothing to destroy
        if (device == VK_NULL_HANDLE) {
            return;
        }

        VK_LOG_DEBUG("destroy device " << name);

        // ggmlR TP trace (silent unless GGMLR_TP_TRACE set): step-by-step device
        // teardown so a segfault here names the exact failing step. sync_staging is
        // of particular interest — the P2P self-test is the first code to allocate
        // it on a non-main device.
        r_tp_tracef("~vk_device_struct enter name=%s staging=%s",
                    name.c_str(), sync_staging ? "present" : "null");

        r_tp_tracef("  destroyFence...");
        device.destroyFence(fence);

        r_tp_tracef("  destroy sync_staging...");
        ggml_vk_destroy_buffer(sync_staging);

        r_tp_tracef("  destroy cmd_pools...");
        if (compute_queue) compute_queue->cmd_pool.destroy(device);
        if (transfer_queue) transfer_queue->cmd_pool.destroy(device);

        // Explicitly clear to ensure queues drop their shared_ptrs to handles
        // before the Vulkan logical device instance is destroyed
        compute_queue.reset();
        transfer_queue.reset();

        r_tp_tracef("  destroy pipelines...");
        for (auto& pipeline : all_pipelines) {
            if (pipeline.expired()) {
                continue;
            }

            vk_pipeline pl = pipeline.lock();
            ggml_vk_destroy_pipeline(device, pl);
        }
        all_pipelines.clear();

        r_tp_tracef("  destroyDescriptorSetLayout...");
        device.destroyDescriptorSetLayout(dsl);

        r_tp_tracef("  device.destroy()...");
        device.destroy();
        r_tp_tracef("~vk_device_struct done name=%s", name.c_str());
    }
};


void vk_command_pool::init(vk_device& device, vk_queue *q_) {
    q = q_;

    vk::CommandPoolCreateInfo command_pool_create_info(vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT), q->queue_family_index);
    pool = device->device.createCommandPool(command_pool_create_info);
}

void vk_command_pool::destroy(vk::Device& device) {
    device.destroyCommandPool(pool);
    pool = nullptr;
    cmd_buffers.clear();
}

struct vk_buffer_struct {
    vk::Buffer buffer = VK_NULL_HANDLE;
    vk::DeviceMemory device_memory = VK_NULL_HANDLE;
    vk::MemoryPropertyFlags memory_property_flags;
    void * ptr;
    size_t size = 0;
    vk::DeviceAddress bda_addr {};

    // ggmlR Tensor Parallelism (P2P): true when this buffer's memory was
    // allocated with opaque-fd export capability (VkExportMemoryAllocateInfo),
    // so its device memory can be shared to another Vulkan device. Not upstream.
    bool exportable = false;

    // ggmlR Tensor Parallelism (P2P): the VkPhysicalDeviceMemoryProperties index
    // actually used for this allocation. Recorded so an exporter can hand it to
    // the importer: getMemoryFdPropertiesKHR returns ErrorUnknown on some NVIDIA
    // drivers, and the imported fd MUST bind to the SAME memory type as the
    // exporter — the two GPUs are identical, so the index is valid on both. Not
    // upstream. (uint32_t)-1 means "not tracked / choose from the mask".
    uint32_t memory_type_index = (uint32_t)-1;

    vk_device device;

    ~vk_buffer_struct() {
        if (size == 0) {
            return;
        }
        VK_LOG_DEBUG("~vk_buffer_struct(" << buffer << ", " << size << ")");

        // ggmlR TP trace (silent unless GGMLR_TP_TRACE set): localizes teardown
        // crashes — the last line written before a segfault names the failing step.
        r_tp_tracef("~vk_buffer_struct enter size=%zu dev=%p mem_type=%u",
                    size, (void *) device.get(), memory_type_index);

        // ggmlR TP: guard teardown. A P2P import that bound to an unusable memory
        // type (e.g. a failed cross-device self-test) can leave device_memory in a
        // state where freeMemory faults. Swallow driver errors here so a diagnostic
        // FAIL path unwinds cleanly instead of taking the process down. Not upstream.
        try {
            // bump before destroying, so a thread that sees the buffer gone also sees the new count
            device->buffer_destroy_count.fetch_add(1, std::memory_order_release);
            r_tp_tracef("  freeMemory...");
            device->device.freeMemory(device_memory);
            r_tp_tracef("  destroyBuffer...");
            device->device.destroyBuffer(buffer);
            r_tp_tracef("~vk_buffer_struct done");
        } catch (const vk::SystemError &) {
            // best-effort release; nothing actionable at destruction time
            r_tp_tracef("~vk_buffer_struct caught vk::SystemError");
        }
    }
};

struct vk_subbuffer {
    vk_buffer buffer;
    uint64_t offset;
    uint64_t size;

    operator vk::DescriptorBufferInfo() const {
        return { buffer->buffer, offset, size };
    }
};

// vk_event is used for the event-related backend interfaces. It uses 'event' for
// event_wait and 'fence' for event_synchronize. Polling on an event for
// event_synchronize wouldn't be sufficient to wait for command buffers to complete,
// and would lead to validation errors.
struct vk_event {
    std::vector<vk::Event> events_free; // Events available for reuse
    std::vector<vk::Event> events_submitted; // Events that are fully submitted and can be reused on next synchronize
    vk::Event event;
    bool has_event;

    vk_semaphore tl_semaphore;
    vk_command_buffer* cmd_buffer = nullptr;
    uint64_t cmd_buffer_use_counter = 0;
};

struct vk_submission {
    vk_command_buffer* buffer = nullptr;
    std::vector<vk_semaphore> wait_semaphores;
    std::vector<vk_semaphore> signal_semaphores;
};

typedef std::vector<vk_submission> vk_sequence;

struct vk_quantize_q8_1_push_constants {
    uint32_t ne;
    uint32_t num_blocks;
};

struct vk_mat_mat_push_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t base_work_group_z; uint32_t num_batches;
    uint32_t k_split;
    uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
    uint32_t padded_N;
};

#define MAT_VEC_FUSION_FLAGS_BIAS0 0x1
#define MAT_VEC_FUSION_FLAGS_BIAS1 0x2
#define MAT_VEC_FUSION_FLAGS_SCALE0 0x4
#define MAT_VEC_FUSION_FLAGS_SCALE1 0x8

struct vk_mat_vec_push_constants {
    uint32_t ncols;
    uint32_t stride_a;
    uint32_t stride_b;
    uint32_t stride_d;
    uint32_t batch_stride_a;
    uint32_t batch_stride_b;
    uint32_t batch_stride_d;
    uint32_t fusion_flags;
    uint32_t base_work_group_y;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t broadcast2;
    uint32_t broadcast3;
};

struct vk_mat_vec_p021_push_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t nchannels_x;
    uint32_t nchannels_y;
    uint32_t b_offset;
    uint32_t d_offset;
    uint32_t fusion_flags;
};

struct vk_mat_vec_nc_push_constants {
    uint32_t ncols_x;
    uint32_t nrows_x;
    uint32_t row_stride_x;
    uint32_t channel_stride_x;
    uint32_t channel_stride_y;
    uint32_t channel_x_divisor;
    uint32_t ne12;
    uint32_t b_offset;
    uint32_t d_offset;
    uint32_t nb03;
    uint32_t nb13;
    uint32_t nb23;
    uint32_t fusion_flags;
};

struct vk_mat_mat_id_push_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t nei1; uint32_t nbi1; uint32_t ne11;
    uint32_t n_experts;
    uint32_t row_ids_offset;
};
struct vk_mat_vec_id_push_constants {
    uint32_t ncols;
    uint32_t stride_a;
    uint32_t stride_b;
    uint32_t stride_d;
    uint32_t batch_stride_a;
    uint32_t batch_stride_b;
    uint32_t batch_stride_d;
    uint32_t fusion_flags;
    uint32_t nei0;
    uint32_t ne11;
    uint32_t expert_i1;
    uint32_t nbi1;
};

struct vk_flash_attn_push_constants {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
};

struct vk_op_flash_attn_split_k_reduce_push_constants {
    uint32_t D;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t k_num;
    uint32_t sinks;
};
static_assert(sizeof(vk_flash_attn_push_constants) <= 128, "sizeof(vk_flash_attn_push_constants) must be <= 128");

struct vk_op_push_constants {
    uint32_t KX;
    uint32_t KY;
    float param1;
    float param2;
    float param3;
    float param4;
};

struct vk_op_count_experts_push_constants {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t nb00;
    uint32_t nb01;
    uint32_t a_offset;
    uint32_t n_experts;
    uint32_t ne00mp;
    uint32_t ne00L;
    uint32_t row_tile_size;
    uint32_t row_ids_offset;
};

struct vk_op_glu_push_constants {
    uint32_t N;
    uint32_t ne00;
    uint32_t ne20;
    uint32_t mode;  // 0: default, 1: swapped, 2: split
    float alpha; // for swiglu_oai
    float limit;
    uint32_t nb00;
    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb10;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb20;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;
    uint32_t ne21;
    uint32_t ne22;
    uint32_t misalign_offsets;
    uint32_t ne2_012mp; uint32_t ne2_012L;
    uint32_t ne2_01mp;  uint32_t ne2_01L;
    uint32_t ne2_0mp;   uint32_t ne2_0L;
};

struct vk_op_unary_push_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t ne04;
    uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03; uint32_t nb04;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t ne14;
    uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13; uint32_t nb14;
    uint32_t misalign_offsets;
    float param1; float param2; float param3; float param4;
    uint32_t ne0_0123mp; uint32_t ne0_0123L;
    uint32_t ne0_012mp;  uint32_t ne0_012L;
    uint32_t ne0_01mp;   uint32_t ne0_01L;
    uint32_t ne0_0mp;    uint32_t ne0_0L;
    uint32_t ne1_0123mp; uint32_t ne1_0123L;
    uint32_t ne1_012mp;  uint32_t ne1_012L;
    uint32_t ne1_01mp;   uint32_t ne1_01L;
    uint32_t ne1_0mp;    uint32_t ne1_0L;
};

struct vk_op_dsv4_hc_comb_push_constants {
    uint32_t n_tokens;

    uint32_t nbm0; uint32_t nbm1;
    uint32_t nbs0;
    uint32_t nbb0;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t m_offset;
    uint32_t s_offset;
    uint32_t b_offset;
    uint32_t d_offset;

    float eps;
    uint32_t n_iter;
};

struct vk_op_dsv4_hc_pre_push_constants {
    uint32_t n_embd;
    uint32_t n_tokens;

    uint32_t nbx0; uint32_t nbx1; uint32_t nbx2;
    uint32_t nbw0; uint32_t nbw1; uint32_t nbw2;
    uint32_t nbd0; uint32_t nbd1;

    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t d_offset;

    float scale;
};

struct vk_op_dsv4_hc_post_push_constants {
    uint32_t n_embd;
    uint32_t n_tokens;

    uint32_t nbx0; uint32_t nbx1;
    uint32_t nbr0; uint32_t nbr1; uint32_t nbr2;
    uint32_t nbp0; uint32_t nbp1;
    uint32_t nbc0; uint32_t nbc1; uint32_t nbc2;
    uint32_t nbd0; uint32_t nbd1; uint32_t nbd2;

    uint32_t x_offset;
    uint32_t r_offset;
    uint32_t p_offset;
    uint32_t c_offset;
    uint32_t d_offset;

    uint32_t gate;
    float    gate_scale_in;
    float    gate_scale_out;
};

struct vk_op_lightning_indexer_push_constants {
    uint32_t n_kv;
    uint32_t n_heads;
    uint32_t n_tokens;
    uint32_t n_masks;
    uint32_t q_nb1;
    uint32_t q_nb2;
    uint32_t q_nb3;
    uint32_t k_nb2;
    uint32_t k_nb3;
    uint32_t w_nb1;
    uint32_t w_nb3;
    uint32_t m_nb1;
    uint32_t m_nb3;
    uint32_t d_nb1;
    uint32_t d_nb3;
};

static_assert(sizeof(vk_op_lightning_indexer_push_constants) <= 128);

struct vk_fa_xe_opt_push_constants {
    uint32_t kv_seq_len;
    uint32_t activation_length;
    uint32_t q_head;
    uint32_t kv_head;
    uint32_t qk_ratio;
    uint32_t qk_sub_groups;
    uint32_t flag;
    uint32_t nbkv_tok;
    uint32_t nbkv_head;
    uint32_t batch_stride_q;
    uint32_t batch_stride_k;
    uint32_t batch_stride_v;
    uint32_t batch_stride_m;
    uint32_t batch_stride_o;
    float softmax_scale;
};

struct vk_op_fwht_push_constants {
    uint32_t n_rows;
    uint32_t src_offset;
    uint32_t dst_offset;
    float scale;
};

struct vk_op_topk_radix_push_constants {
    uint32_t ncols;
    uint32_t k;
    uint32_t nrows;
    uint32_t n_tps;    // QSA only
    uint32_t n_blocks; // QSA only
    uint32_t n_stream; // QSA only
};

struct vk_op_col2im_1d_push_constants {
    uint32_t T_out;
    uint32_t OC;
    uint32_t K_OC;
    uint32_t T_in;
    uint32_t K;
    int32_t  stride;
    int32_t  p0;
};

struct vk_op_snake_push_constants {
    uint32_t ne0;
    uint32_t ne1;
};

struct vk_op_pool1d_push_constants {
    uint32_t IL;
    uint32_t OL;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0;
    int32_t s0;
    int32_t p0;
};

struct vk_op_gated_linear_attn_push_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
    float scale;
};

struct vk_op_conv3d_push_constants {
    uint32_t OC;
    uint32_t IC;
    uint32_t N;

    uint32_t IW;
    uint32_t IH;
    uint32_t ID;
    uint32_t OW;
    uint32_t OH;
    uint32_t OD;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;

    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;

    uint32_t nb1;
    uint32_t nb2;
    uint32_t nb3;

    uint32_t OWmp;     uint32_t OWL;
    uint32_t OWOHmp;   uint32_t OWOHL;
    uint32_t OWOHODmp; uint32_t OWOHODL;

    uint32_t knl_offset;
    uint32_t src_offset;
    uint32_t dst_offset;
};

struct vk_op_flash_attn_sparse_compact_push_constants {
    uint32_t KV;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nbm1;
    uint32_t nbm2;
    uint32_t nbm3;
    uint32_t n_kv_max;
};
static_assert(sizeof(vk_op_unary_push_constants) <= 256, "sizeof(vk_op_unary_push_constants) must be <= 256");

static vk_op_unary_push_constants vk_op_unary_push_constants_init(const ggml_tensor * src0, const ggml_tensor * dst, int64_t ne = 0) {
    GGML_ASSERT(ne != 0 || (ggml_nelements(src0) == ggml_nelements(dst)));
    ne = ne != 0 ? ne : ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_unary_push_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_size(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.ne04 = (uint32_t)src0->ne[4];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);
    p.nb04 = (uint32_t)(src0->nb[4] / src0_tsize);

    size_t dst_tsize = ggml_type_size(dst->type);
    p.ne10 = (uint32_t)dst->ne[0];
    p.ne11 = (uint32_t)dst->ne[1];
    p.ne12 = (uint32_t)dst->ne[2];
    p.ne13 = (uint32_t)dst->ne[3];
    p.ne14 = (uint32_t)dst->ne[4];
    p.nb10 = (uint32_t)(dst->nb[0] / dst_tsize);
    p.nb11 = (uint32_t)(dst->nb[1] / dst_tsize);
    p.nb12 = (uint32_t)(dst->nb[2] / dst_tsize);
    p.nb13 = (uint32_t)(dst->nb[3] / dst_tsize);
    p.nb14 = (uint32_t)(dst->nb[4] / dst_tsize);

    return p; // offsets are initialized later in ggml_vk_op
}

struct vk_op_pad_push_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t misalign_offsets;
    uint32_t circular;

    uint32_t lp0; uint32_t rp0;
    uint32_t lp1; uint32_t rp1;
    uint32_t lp2; uint32_t rp2;
    uint32_t lp3; uint32_t rp3;
};

static vk_op_pad_push_constants vk_op_pad_push_constants_init(const ggml_tensor * src0, const ggml_tensor * dst) {
    int64_t ne = ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_pad_push_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_size(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)(src0->ne[3] * src0->ne[4]); // collapse 5D → 4D
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_size(dst->type);
    p.ne10 = (uint32_t)dst->ne[0];
    p.ne11 = (uint32_t)dst->ne[1];
    p.ne12 = (uint32_t)dst->ne[2];
    p.ne13 = (uint32_t)(dst->ne[3] * dst->ne[4]); // collapse 5D → 4D
    p.nb10 = (uint32_t)(dst->nb[0] / dst_tsize);
    p.nb11 = (uint32_t)(dst->nb[1] / dst_tsize);
    p.nb12 = (uint32_t)(dst->nb[2] / dst_tsize);
    p.nb13 = (uint32_t)(dst->nb[3] / dst_tsize);

    p.lp0 = dst->op_params[0];
    p.rp0 = dst->op_params[1];
    p.lp1 = dst->op_params[2];
    p.rp1 = dst->op_params[3];
    p.lp2 = dst->op_params[4];
    p.rp2 = dst->op_params[5];
    p.lp3 = dst->op_params[6];
    p.rp3 = dst->op_params[7];
    p.circular = dst->op_params[8];

    return p; // fastdiv values and offsets are initialized later in ggml_vk_op
}

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static void init_fastdiv_values(uint32_t d, uint32_t &mp, uint32_t &L)
{
    // compute L = ceil(log2(d));
    L = 0;
    while (L < 32 && (uint32_t{1} << L) < d) {
        L++;
    }

    mp = (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
}

template <typename T> void init_pushconst_fastdiv(T &p) {
    GGML_UNUSED(p);
    static_assert(!std::is_const<T>::value, "unexpected type");
}

template <> inline void init_pushconst_fastdiv(vk_op_unary_push_constants &p) {
    init_fastdiv_values(p.ne03*p.ne02*p.ne01*p.ne00, p.ne0_0123mp,   p.ne0_0123L);
    init_fastdiv_values(p.ne02*p.ne01*p.ne00,         p.ne0_012mp,    p.ne0_012L);
    init_fastdiv_values(p.ne01*p.ne00,                p.ne0_01mp,     p.ne0_01L);
    init_fastdiv_values(p.ne00,                       p.ne0_0mp,      p.ne0_0L);
    init_fastdiv_values(p.ne13*p.ne12*p.ne11*p.ne10,  p.ne1_0123mp,   p.ne1_0123L);
    init_fastdiv_values(p.ne12*p.ne11*p.ne10,          p.ne1_012mp,    p.ne1_012L);
    init_fastdiv_values(p.ne11*p.ne10,                 p.ne1_01mp,     p.ne1_01L);
    init_fastdiv_values(p.ne10,                        p.ne1_0mp,      p.ne1_0L);
}

template <> inline void init_pushconst_fastdiv(vk_op_glu_push_constants &p) {
    // GLU linearizes over dst, then uses dst coordinates for src0/src1.
    init_fastdiv_values(p.ne22*p.ne21*p.ne20,  p.ne2_012mp,    p.ne2_012L);
    init_fastdiv_values(p.ne21*p.ne20,         p.ne2_01mp,     p.ne2_01L);
    init_fastdiv_values(p.ne20,                p.ne2_0mp,      p.ne2_0L);
}

struct vk_op_binary_push_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t ne04; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03; uint32_t nb04;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t ne14; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13; uint32_t nb14;
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23; uint32_t ne24; uint32_t nb20; uint32_t nb21; uint32_t nb22; uint32_t nb23; uint32_t nb24;
    uint32_t misalign_offsets;
    float param1; float param2; int32_t param3;
};
static_assert(sizeof(vk_op_binary_push_constants) <= 256, "sizeof(vk_op_binary_push_constants) must be <= 256");

struct vk_op_concat_push_constants : vk_op_binary_push_constants {};

static_assert(sizeof(vk_op_concat_push_constants) == sizeof(vk_op_binary_push_constants));

static_assert(std::is_standard_layout_v<vk_op_concat_push_constants>);

struct vk_op_rel_pos_bias_push_constants {
    uint32_t H;
    uint32_t W;
    uint32_t B;
    uint32_t C;
    uint32_t rel_h;
    uint32_t rel_w;
};
static_assert(sizeof(vk_op_rel_pos_bias_push_constants) <= 256, "sizeof(vk_op_rel_pos_bias_push_constants) must be <= 256");

// UMAP SGD layout step push constants. Layout MUST match the `parameter` block
// in vulkan-shaders/umap_sgd.comp exactly (4 uints then 4 floats).
struct vk_op_umap_sgd_push_constants {
    uint32_t n;
    uint32_t ne;
    uint32_t n_neg;
    uint32_t seed;
    float    alpha;
    float    a;
    float    b;
    float    gamma;
    uint32_t epoch;      // 1-based; drives the shader's weighted-sampling test
    uint32_t row_stride; // edges per Y-row of the 2D dispatch grid (see umap.cpp)
};
static_assert(sizeof(vk_op_umap_sgd_push_constants) <= 256, "sizeof(vk_op_umap_sgd_push_constants) must be <= 256");

// Pairwise squared-distance push constants. Must match the `parameter` block in
// vulkan-shaders/pairwise_dist.comp exactly (two uints).
struct vk_op_pairwise_dist_push_constants {
    uint32_t n;
    uint32_t dims;
};
static_assert(sizeof(vk_op_pairwise_dist_push_constants) <= 256, "sizeof(vk_op_pairwise_dist_push_constants) must be <= 256");

// RoiAlign push constants. Must match the `parameter` block in
// vulkan-shaders/roi_align.comp exactly: eight 4-byte scalars then one float.
// sampling_ratio is SIGNED -- 0 means "adaptive" and the shader tests it with
// `> 0`, which an unsigned copy would still pass but for the wrong reason if a
// caller ever handed it a negative value.
struct vk_op_roi_align_push_constants {
    uint32_t W;
    uint32_t H;
    uint32_t C;
    uint32_t num_rois;
    uint32_t ow;
    uint32_t oh;
    int32_t  sampling_ratio;
    uint32_t mode;
    float    spatial_scale;
};
static_assert(sizeof(vk_op_roi_align_push_constants) <= 256, "sizeof(vk_op_roi_align_push_constants) must be <= 256");

// QLinearMatMul (i32 accumulator) push constants. Must match the `parameter`
// block in vulkan-shaders/qmatmul_i32.comp exactly. b_zp_any is computed on the
// host so the shader does not scan the zero-point table in every thread.
struct vk_op_qmatmul_i32_push_constants {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    float    a_scale;
    float    y_scale;
    int32_t  a_zp;
    int32_t  y_zp;
    uint32_t n_b_scale;
    uint32_t n_b_zp;
    uint32_t b_zp_any;
    float    out_lo;
    float    out_hi;
};
static_assert(sizeof(vk_op_qmatmul_i32_push_constants) <= 256, "sizeof(vk_op_qmatmul_i32_push_constants) must be <= 256");

// QLinearConv (i32 accumulator) push constants. Must match the `parameter`
// block in vulkan-shaders/qconv_i32.comp exactly.
struct vk_op_qconv_i32_push_constants {
    uint32_t W_in;
    uint32_t H_in;
    uint32_t C_in;
    uint32_t W_out;
    uint32_t H_out;
    uint32_t C_out;
    uint32_t KW;
    uint32_t KH;
    int32_t  stride_w;
    int32_t  stride_h;
    int32_t  pad_w;
    int32_t  pad_h;
    int32_t  dil_w;
    int32_t  dil_h;
    float    x_scale;
    float    y_scale;
    int32_t  x_zp;
    int32_t  y_zp;
    uint32_t n_w_scale;
    uint32_t n_w_zp;
    uint32_t has_bias;
    float    out_lo;
    float    out_hi;
    // w_zero_point is optional in the operator spec. Its binding is filled
    // with another tensor when it is absent, because an unwritten descriptor
    // is not a legal source -- so the shader needs to be told not to read it,
    // rather than inferring absence from a length.
    uint32_t has_w_zp;
    // Diagnostic, GGMLR_QCONV_DEBUG_ACC=1: write the integer accumulator
    // instead of the requantised value, so a per-node trace can tell a
    // summation difference from a rounding one.
    uint32_t debug_acc;
    // Images in the batch (dst->ne[3]). Walked by the shader: a detector's mask
    // head convolves one feature map PER DETECTION, so this is 51 rather than 1
    // on MaskRCNN-12-int8. Leaving it out computed image 0 and left the rest of
    // the output untouched -- masks summed 139.5 against ONNX Runtime's 7088.1.
    uint32_t N_batch;
};
static_assert(sizeof(vk_op_qconv_i32_push_constants) <= 256, "sizeof(vk_op_qconv_i32_push_constants) must be <= 256");

// NonMaxSuppression push constants. Must match the `parameter` block in
// vulkan-shaders/nms.comp exactly.
struct vk_op_nms_push_constants {
    uint32_t num_boxes;
    uint32_t num_classes;
    uint32_t N;
    uint32_t max_per_pair;
    int32_t  max_output;
    float    iou_thresh;
    float    score_thresh;
    uint32_t have_score_thresh;
    uint32_t center_point_box;
};
static_assert(sizeof(vk_op_nms_push_constants) <= 256, "sizeof(vk_op_nms_push_constants) must be <= 256");

// Tiled fused k-NN push constants. Must match the `parameter` block in
// vulkan-shaders/knn_tiled.comp exactly (three uints). k must equal the K
// specialization constant the pipeline was created with.
struct vk_op_knn_tiled_push_constants {
    uint32_t n;
    uint32_t dims;
    uint32_t k;
};
static_assert(sizeof(vk_op_knn_tiled_push_constants) <= 256, "sizeof(vk_op_knn_tiled_push_constants) must be <= 256");

// FP64 matmul push constants (PoC). Must match the `parameter` block in
// vulkan-shaders/matmul_f64.comp exactly (three uints: M, N, K).
struct vk_op_matmul_f64_push_constants {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};
static_assert(sizeof(vk_op_matmul_f64_push_constants) <= 256, "sizeof(vk_op_matmul_f64_push_constants) must be <= 256");

// Sparse LogNormalize push constants. Must match the `parameter` block in
// vulkan-shaders/sparse_lognorm.comp exactly (one uint: the nnz count).
struct vk_op_sparse_lognorm_push_constants {
    uint32_t nnz;
    uint32_t row_stride;  // elements per Y-row of the 2D dispatch grid (see sparse.cpp)
};
static_assert(sizeof(vk_op_sparse_lognorm_push_constants) <= 256, "sizeof(vk_op_sparse_lognorm_push_constants) must be <= 256");

static vk_op_binary_push_constants vk_op_binary_push_constants_init(
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
    uint32_t ne_override = 0,
    float param1 = 0.0f, float param2 = 0.0f, int32_t param3 = 0)
{
    const uint32_t src0_tsize = (uint32_t)ggml_type_size(src0->type);
    const uint32_t src1_tsize = (uint32_t)ggml_type_size(src1 ? src1->type : src0->type);
    const uint32_t dst_tsize  = (uint32_t)ggml_type_size(dst->type);

    vk_op_binary_push_constants p{};
    p.ne     = ne_override ? ne_override : (uint32_t)ggml_nelements(src0);
    p.ne00 = (uint32_t)src0->ne[0]; p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2]; p.ne03 = (uint32_t)src0->ne[3]; p.ne04 = (uint32_t)src0->ne[4];
    p.nb00 = (uint32_t)(src0->nb[0]/src0_tsize); p.nb01 = (uint32_t)(src0->nb[1]/src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2]/src0_tsize); p.nb03 = (uint32_t)(src0->nb[3]/src0_tsize); p.nb04 = (uint32_t)(src0->nb[4]/src0_tsize);
    if (src1) {
        p.ne10 = (uint32_t)src1->ne[0]; p.ne11 = (uint32_t)src1->ne[1];
        p.ne12 = (uint32_t)src1->ne[2]; p.ne13 = (uint32_t)src1->ne[3]; p.ne14 = (uint32_t)src1->ne[4];
        p.nb10 = (uint32_t)(src1->nb[0]/src1_tsize); p.nb11 = (uint32_t)(src1->nb[1]/src1_tsize);
        p.nb12 = (uint32_t)(src1->nb[2]/src1_tsize); p.nb13 = (uint32_t)(src1->nb[3]/src1_tsize); p.nb14 = (uint32_t)(src1->nb[4]/src1_tsize);
    }
    p.ne20 = (uint32_t)dst->ne[0]; p.ne21 = (uint32_t)dst->ne[1];
    p.ne22 = (uint32_t)dst->ne[2]; p.ne23 = (uint32_t)dst->ne[3]; p.ne24 = (uint32_t)dst->ne[4];
    p.nb20 = (uint32_t)(dst->nb[0]/dst_tsize); p.nb21 = (uint32_t)(dst->nb[1]/dst_tsize);
    p.nb22 = (uint32_t)(dst->nb[2]/dst_tsize); p.nb23 = (uint32_t)(dst->nb[3]/dst_tsize); p.nb24 = (uint32_t)(dst->nb[4]/dst_tsize);
    p.param1 = param1; p.param2 = param2; p.param3 = param3;
    return p; // misalign_offsets set later in ggml_vk_op
}

// Upstream builds this with a 4D brace initializer; ours goes through the 5D
// vk_op_binary_push_constants_init so ne*4/nb*4 stay in place.
static vk_op_binary_push_constants ggml_vk_rms_norm_push_constants(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
        float eps, uint32_t num_partials) {
    return vk_op_binary_push_constants_init(src0, src1, dst, 0, eps, 0.0f, (int32_t)num_partials);
}

/* No vk_op_scatter_elements_push_constants: SCATTER_ELEMENTS sends
 * vk_op_binary_push_constants like the other binary ops, and the pipeline is
 * declared with that size. A second, smaller struct for the same shader is
 * what made the pipeline's push constant range too short to cover the fields
 * the shader reads -- see the note at ggml_vk_create_pipeline for it. */

struct vk_op_multi_add_push_constants {
    // shape for dst
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23;

    // strides for srcs+dst
    uint32_t nb[MAX_PARAMETER_COUNT][4];

    uint32_t rms_partials;
};
// update multi_add.comp if this changes
static_assert(MAX_PARAMETER_COUNT == 12);
static_assert(sizeof(vk_op_multi_add_push_constants) <= 256);

struct vk_op_topk_moe_push_constants {
    uint32_t n_rows;
    uint32_t n_experts_push;
    uint32_t n_expert_used;
    float clamp_min;
    float clamp_max;
    uint32_t gating_func;
    uint32_t has_bias;
    uint32_t with_norm;
    float output_scale;
    float output_bias;
};

struct vk_op_add_id_push_constants {
    uint32_t ne0;
    uint32_t ne1;
    uint32_t s01;
    uint32_t s02;
    uint32_t s11;
    uint32_t s21;
};

struct vk_op_diag_mask_push_constants {
    uint32_t ncols;
    uint32_t rows_per_channel;
    int32_t n_past;
};

struct vk_op_rope_push_constants {
    uint32_t rope_mode;
    uint32_t nrows;
    uint32_t n_dims;
    uint32_t n_offs;
    float freq_scale;
    float freq_base;
    float ext_factor;
    float attn_factor;
    float corr_dims[2];
    float theta_scale;
    uint32_t has_ff;
    int32_t sections[4];
    uint32_t is_imrope;
    uint32_t is_back;
    uint32_t set_rows_stride;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t a_offset;
    uint32_t d_offset;
};

// For fused rms_norm+mul+rope(+view+set_rows)
struct vk_op_rms_norm_mul_rope_push_constants {
    vk_op_binary_push_constants bin;
    vk_op_rope_push_constants rope;
};

struct vk_op_soft_max_push_constants {
    uint32_t KX;
    uint32_t KY;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t ne13;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    float scale;
    float max_bias;
    float m0;
    float m1;
    uint32_t n_head_log2;
    uint32_t nrows_x;
    uint32_t has_sinks;
};

struct vk_op_argsort_push_constants {
    uint32_t ncols;
    uint32_t ncols_padded;
    uint32_t ncols_padded_log2;
    uint32_t nrows;
    uint32_t order;
    uint32_t outer_start;
    uint32_t outer_end;
    uint32_t inner_start;
    uint32_t inner_end;
};

struct vk_op_topk_push_constants {
    uint32_t orig_ncols;
    uint32_t ncols_input;
    uint32_t ncols_output;
    uint32_t k;
    uint32_t nrows;
    uint32_t first_pass;
    uint32_t last_pass;
};

struct vk_op_im2col_push_constants {
    uint64_t dst_addr;
    uint32_t batch_offset; uint32_t offset_delta;
    uint32_t IC;
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t KW; uint32_t KH;
    uint32_t OH_batch;
    uint32_t CHW;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
    int32_t d0; int32_t d1;
    uint32_t batch_IC;
};

struct vk_op_im2col_3d_push_constants {
    uint64_t dst_addr;
    uint32_t nb10;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t s0;
    uint32_t s1;
    uint32_t s2;
    uint32_t p0;
    uint32_t p1;
    uint32_t p2;
    uint32_t d0;
    uint32_t d1;
    uint32_t d2;
    uint32_t IW;
    uint32_t IH;
    uint32_t ID;
    uint32_t IC;
    uint32_t KW;
    uint32_t OH;
    uint32_t KD_KH_KW;
    uint32_t KH_KW;
    uint32_t IC_KD_KH_KW;
    uint32_t N_OD_OH;
    uint32_t OD_OH;
    uint32_t OD_OH_OW_IC_KD_KH_KW;
    uint32_t OH_OW_IC_KD_KH_KW;
    uint32_t OW_IC_KD_KH_KW;
    uint32_t misalign_offsets;
};

struct vk_op_timestep_embedding_push_constants {
    uint32_t nb1;
    uint32_t dim;
    uint32_t max_period;
};

struct vk_op_conv_transpose_1d_push_constants {
    uint32_t Cout;
    uint32_t Cin;
    uint32_t K;
    uint32_t L;
    uint32_t KL;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb11;
    uint32_t nb1;

    int32_t s0;
};

struct vk_op_pool2d_push_constants {
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0; int32_t k1;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
};

struct vk_op_rwkv_wkv6_push_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

struct vk_op_rwkv_wkv7_push_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};
struct vk_op_gated_delta_net_push_constants {
    uint32_t H;
    uint32_t n_tokens;
    uint32_t n_seqs;
    uint32_t s_off;
    uint32_t sq1, sq2, sq3;
    uint32_t sv1, sv2, sv3;
    uint32_t sb1, sb2, sb3;
    uint32_t neq1, rq3;
    float scale;
    uint32_t K;
};
struct vk_op_flash_attn_mask_opt_push_constants {
    uint32_t nem0;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nbm1;
    uint32_t nbm2;
    uint32_t nbm3;
    uint32_t nbd1;
    uint32_t nbd2;
    uint32_t nbd3;
};

struct vk_op_ssm_scan_push_constants {
    uint32_t nb02, nb03, nb12, nb13;
    uint32_t nb21, nb22, nb31;
    uint32_t nb42, nb43, nb52, nb53;
    uint32_t s_off;
    uint32_t n_head, d_head, n_group, n_tok;
    uint32_t n_seq, K;
};
struct vk_op_ssm_conv_push_constants {
    uint32_t nb01, nb02;
    uint32_t nb11;
    uint32_t dst_nb0, dst_nb1, dst_nb2;
    uint32_t nc, ncs, nr, n_t, n_s;
};

// ggmlR extension: backward of ssm_conv. Strides are byte counts; the shader
// divides by sizeof(float). d_c_off is where the packed [d_sx | d_c] output
// switches from one gradient to the other.
struct vk_op_ssm_conv_back_push_constants {
    uint32_t nb01, nb02;
    uint32_t nb11;
    uint32_t g_nb1, g_nb2;
    uint32_t nc, ncs, nr, n_t, n_s;
    uint32_t d_c_off;
};

// ggmlR extension: backward of ssm_scan. Strides are byte counts (the shader
// divides by sizeof(float)); the *_off fields are ELEMENT offsets into the
// packed six-gradient output, which is why they are not scaled.
struct vk_op_ssm_scan_back_push_constants {
    uint32_t s0_nb2, s0_nb3;
    uint32_t x_nb2, x_nb3;
    uint32_t dt_nb1, dt_nb2;
    uint32_t B_nb2, B_nb3;
    uint32_t C_nb2, C_nb3;
    uint32_t nh, nr, ng, nt, ns;
    uint32_t d_s_off, d_x_off, d_dt_off, d_A_off, d_B_off, d_C_off;
    uint32_t g_s_off;
};

// ggmlR extension: GGML_OP_OUT_PROD on the GPU (no upstream Vulkan shader).
// Strides are byte counts; the shader divides by sizeof(float).
struct vk_op_out_prod_push_constants {
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t ne01;
    uint32_t nb01, nb02, nb03;
    uint32_t nb10, nb11, nb12, nb13;
    uint32_t nb1, nb2, nb3;
    uint32_t dps2, dps3;
};

// ggmlR extension: GGML_OP_GET_ROWS_BACK on the GPU. Strides are element
// counts, not bytes.
struct vk_op_get_rows_back_push_constants {
    uint32_t nc;
    uint32_t nr;
    uint32_t dst_row;
    uint32_t src_row;
    uint32_t ne_total;
};

// ggmlR extension: GGML_OP_IM2COL_BACK on the GPU. Mirrors the CPU kernel's
// parameters; ne_total lets the shader bound-check without recomputing it.
struct vk_op_im2col_back_push_constants {
    uint32_t N;
    uint32_t IC;
    uint32_t IH; uint32_t IW;
    uint32_t OH; uint32_t OW;
    uint32_t KH; uint32_t KW;
    int32_t  s0; int32_t  s1;
    int32_t  p0; int32_t  p1;
    int32_t  d0; int32_t  d1;
    uint32_t is_2D;
    uint32_t ne_total;
};

// ggmlR extension: GGML_OP_FLASH_ATTN_EXT backward on the GPU. Strides are
// element counts, not bytes -- the host divides by sizeof(float) so the shader
// can index directly. dq/dk/dv offsets locate each gradient inside the packed
// output, and must match the arithmetic ggml-graph.c uses to view them.
struct vk_op_flash_attn_back_push_constants {
    uint32_t DK, DV, N, M, H, B;
    uint32_t rk2, rk3;
    uint32_t q_s1, q_s2, q_s3;
    uint32_t k_s1, k_s2, k_s3;
    uint32_t v_s1, v_s2, v_s3;
    uint32_t d_s1, d_s2, d_s3;
    uint32_t m_s1, m_s2, m_s3;
    uint32_t m_ne2, m_ne3;
    uint32_t dq_off, dk_off, dv_off;
    float    scale;
    uint32_t has_mask;
};

// ggmlR extension: GGML_OP_CROSS_ENTROPY_LOSS_BACK on the GPU (no upstream
// Vulkan shader). inv_nr is 1/nrows, folded in on the host so the shader does
// not repeat the division per row.
struct vk_op_cross_entropy_loss_back_push_constants {
    uint32_t nc;
    uint32_t nr;
    float    inv_nr;
};

struct vk_op_conv2d_push_constants {
    uint32_t Cout;
    uint32_t Cin;
    uint32_t N;

    uint32_t W;
    uint32_t H;
    uint32_t OW;
    uint32_t OH;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;

    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;

    uint32_t nb1;
    uint32_t nb2;
    uint32_t nb3;

    // init_fastdiv_values constants for dividing by OW, OW*OH
    uint32_t OWmp;   uint32_t OWL;
    uint32_t OWOHmp; uint32_t OWOHL;

    uint32_t knl_offset;
    uint32_t src_offset;
    uint32_t dst_offset;
};

template <> inline void init_pushconst_fastdiv(vk_op_conv2d_push_constants &p) {
    // Compute magic values to divide by OW, OW*OH
    init_fastdiv_values(p.OW,       p.OWmp,    p.OWL);
    init_fastdiv_values(p.OW*p.OH,  p.OWOHmp,  p.OWOHL);
}

template <> inline void init_pushconst_fastdiv(vk_op_conv3d_push_constants &p) {
    init_fastdiv_values(p.OW,             p.OWmp,     p.OWL);
    init_fastdiv_values(p.OW*p.OH,        p.OWOHmp,   p.OWOHL);
    init_fastdiv_values(p.OW*p.OH*p.OD,   p.OWOHODmp, p.OWOHODL);
}

template <> inline void init_pushconst_fastdiv(vk_op_count_experts_push_constants &p) {
    init_fastdiv_values(p.ne00, p.ne00mp, p.ne00L);
}

struct vk_op_conv2d_dw_push_constants {
    uint32_t ne;
    uint32_t batches;
    uint32_t channels;
    uint32_t dst_w;
    uint32_t dst_h;
    uint32_t src_w;
    uint32_t src_h;
    uint32_t knl_w;
    uint32_t knl_h;
    int32_t stride_x;
    int32_t stride_y;
    int32_t pad_x;
    int32_t pad_y;
    int32_t dilation_x;
    int32_t dilation_y;
};

struct vk_op_upscale_push_constants {
    uint32_t ne; uint32_t a_offset; uint32_t d_offset;
    uint32_t ne00; uint32_t ne01;
    uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13;
    float sf0; float sf1; float sf2; float sf3;
    float pixel_offset;
};

struct vk_op_sum_rows_push_constants
{
    uint32_t n_cols;
    uint32_t ne01, ne02;
    uint32_t nb01, nb02, nb03;
    uint32_t nb11, nb12, nb13;
    float weight;
    uint32_t misalign_offsets;
    uint32_t ne0_12mp, ne0_12L;
    uint32_t ne0_1mp, ne0_1L;
};

static vk_op_sum_rows_push_constants vk_op_sum_rows_push_constants_init(const ggml_tensor * src, const ggml_tensor * dst, int64_t n_cols) {
    uint32_t type_size = (uint32_t)ggml_type_size(src->type);
    vk_op_sum_rows_push_constants p = {};
    p.n_cols = (uint32_t)n_cols;
    p.ne01 = (uint32_t)src->ne[1];
    p.ne02 = (uint32_t)src->ne[2];
    p.nb01 = (uint32_t)src->nb[1] / type_size;
    p.nb02 = (uint32_t)src->nb[2] / type_size;
    p.nb03 = (uint32_t)src->nb[3] / type_size;
    p.nb11 = (uint32_t)dst->nb[1] / type_size;
    p.nb12 = (uint32_t)dst->nb[2] / type_size;
    p.nb13 = (uint32_t)dst->nb[3] / type_size;
    p.weight = 1.0f;
    return p;
}

template <> inline void init_pushconst_fastdiv(vk_op_sum_rows_push_constants &p) {
    init_fastdiv_values(p.ne01*p.ne02, p.ne0_12mp, p.ne0_12L);
    init_fastdiv_values(p.ne01,        p.ne0_1mp,  p.ne0_1L);
}

// Allow pre-recording command buffers
struct vk_staging_memcpy {
    vk_staging_memcpy(void * _dst, const void * _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

    void * dst;
    const void * src;
    size_t n;
};

struct vk_staging_memset {
    vk_staging_memset(void * _dst, uint32_t _val, size_t _n) : dst(_dst), val(_val), n(_n) {}

    void * dst;
    uint32_t val;
    size_t n;
};

struct vk_context_struct {
    vk_submission * s;
    std::vector<vk_sequence> seqs;

    int exit_tensor_idx;

    std::vector<vk_staging_memcpy> in_memcpys;
    std::vector<vk_staging_memcpy> out_memcpys;
    std::vector<vk_staging_memset> memsets;

    std::vector<std::string> debug_labels;

    vk_command_pool * p {};
};
typedef std::shared_ptr<vk_context_struct> vk_context;
typedef std::weak_ptr<vk_context_struct> vk_context_ref;

struct ggml_vk_garbage_collector {
    std::vector<vk_semaphore> tl_semaphores;
    std::vector<vk_semaphore> semaphores;
    std::vector<vk::Event> events;
    std::vector<vk_context> contexts;
};

static void ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
static void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested = nullptr);
static void ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);

#define VK_LOG_MEMORY(msg) ((void) 0)

static uint64_t ggml_vk_get_node_flops(const ggml_tensor * node) {
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        const uint64_t m     = node->ne[0];
        const uint64_t n     = node->ne[1];
        const uint64_t k     = node->src[1]->ne[0];
        const uint64_t batch = node->ne[2] * node->ne[3];
        return m * n * (k + (k - 1)) * batch;
    }
    if (node->op == GGML_OP_CONV_2D || node->op == GGML_OP_CONV_TRANSPOSE_2D) {
        const ggml_tensor * knl = node->src[0];
        const uint64_t Cout  = node->ne[2];
        const uint64_t size_K = node->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        const uint64_t size_N = node->ne[3] * node->ne[0] * node->ne[1];
        return Cout * size_N * (size_K + (size_K - 1));
    }
    if (node->op == GGML_OP_CONV_3D) {
        const ggml_tensor * knl = node->src[0];
        const uint64_t OC     = ggml_get_op_params_i32(node, 11);
        const uint64_t IC     = ggml_get_op_params_i32(node, 9);
        const uint64_t size_K = IC * knl->ne[0] * knl->ne[1] * knl->ne[2];
        const uint64_t size_N = node->ne[3] / OC * node->ne[0] * node->ne[1] * node->ne[2];
        return OC * size_N * (size_K + (size_K - 1));
    }
    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
        const ggml_tensor * q = node->src[0];
        const ggml_tensor * k = node->src[1];
        const ggml_tensor * v = node->src[2];
        return 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
    }
    return 0;
}

static bool vk_perf_logger_enabled = false;
static bool vk_perf_logger_concurrent = false;
static bool vk_enable_sync_logger = false;
// number of calls between perf logger prints
static uint32_t vk_perf_logger_frequency = 1;

class vk_perf_logger {
  public:
    void print_timings(bool force = false) {
        if (timings.empty()) {
            return;
        }
        print_count++;
        if ((print_count % vk_perf_logger_frequency) != 0 && !force) {
            return;
        }
        print_count = 0;
        uint64_t total_all_op_times = 0;
        r_ggml_printf("----------------\nVulkan Timings:\n");
        for (const auto & t : timings) {
            uint64_t total_op_times = 0;
            for (const auto & time : t.second) {
                total_op_times += time;
            }
            r_ggml_printf("%s: %zu x %.2f us = %.2f us",
                t.first.c_str(), t.second.size(),
                total_op_times / (double)t.second.size() / 1000.0,
                total_op_times / 1000.0);

            auto it = flops.find(t.first);
            if (it != flops.end() && (it->second).size() == t.second.size()) {
                uint64_t total_op_flops = 0;
                for (const auto & elem : it->second) {
                    total_op_flops += elem;
                }
                r_ggml_printf(" (%.2f GFLOPS/s)",
                    (double(total_op_flops) / 1e9) / (double(total_op_times) / 1e9));
            }

            total_all_op_times += total_op_times;
            r_ggml_printf("\n");
        }

        if (timings.size() > 0) {
            r_ggml_printf("Total time: %.2f us.\n", total_all_op_times / 1000.0);
        }

        timings.clear();
        flops.clear();
    }

    std::string get_node_fusion_name(const ggml_tensor * node, const char *fusion_name, uint64_t *n_flops) {
        *n_flops = ggml_vk_get_node_flops(node);
        std::string fusion_str;
        if (fusion_name) {
            fusion_str = fusion_name + std::string(" ");
        }
        if (node->op == GGML_OP_UNARY) {
            return fusion_str + ggml_unary_op_name(ggml_get_unary_op(node));
        }
        if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
            const uint64_t m     = node->ne[0];
            const uint64_t n     = node->ne[1];
            const uint64_t k     = node->src[1]->ne[0];
            const uint64_t batch = node->ne[2] * node->ne[3];
            std::string    name  = ggml_op_name(node->op);
            if ((node->op == GGML_OP_MUL_MAT && n <= mul_mat_vec_max_cols) ||
                (node->op == GGML_OP_MUL_MAT_ID && node->src[2]->ne[1] == 1)) {
                name += "_VEC";
            }
            name += " ";
            name += ggml_type_name(node->src[0]->type);
            name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
            if (node->op == GGML_OP_MUL_MAT_ID) {
                name += " n_expert=" + std::to_string(node->src[0]->ne[2]);
            }
            if (batch > 1) {
                name += " batch=" + std::to_string(batch);
            }
            name = fusion_str + name;
            return name;
        }
        if (node->op == GGML_OP_CONV_2D || node->op == GGML_OP_CONV_TRANSPOSE_2D) {
            std::string   name    = ggml_op_name(node->op);
            ggml_tensor * knl     = node->src[0];
            uint64_t      OW      = node->ne[0];
            uint64_t      OH      = node->ne[1];
            uint64_t      N       = node->ne[3];
            uint64_t      Cout    = node->ne[2];
            uint64_t      KW      = knl->ne[0];
            uint64_t      KH      = knl->ne[1];
            uint64_t      Cin     = node->src[1]->ne[2];
            // KxCRS @ CRSxNPQ = KxNPQ -> M=K, K=CRS, N=NPQ
            uint64_t      size_M  = Cout;
            uint64_t      size_K  = Cin * KW * KH;
            uint64_t      size_N  = N * OW * OH;
            name += " M=Cout=" + std::to_string(size_M) + ", K=Cin*KW*KH=" + std::to_string(size_K) +
                    ", N=N*OW*OH=" + std::to_string(size_N);
            name = fusion_str + name;
            return name;
        }
        if (node->op == GGML_OP_RMS_NORM) {
            std::string   name    = ggml_op_name(node->op);
            name += "(" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," + std::to_string(node->ne[2]) + "," + std::to_string(node->ne[3]) + ")";
            name = fusion_str + name;
            return name;
        }
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            const ggml_tensor * dst = node;
            const ggml_tensor * q = node->src[0];
            const ggml_tensor * k = node->src[1];
            const ggml_tensor * v = node->src[2];
            const ggml_tensor * m = node->src[3];
            std::stringstream name;
            name << fusion_str;
            name << ggml_op_name(node->op) <<
                " dst(" << dst->ne[0] << "," << dst->ne[1] << "," << dst->ne[2] << "," << dst->ne[3] << "), " <<
                " q(" << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3] << "), " <<
                " k(" << k->ne[0] << "," << k->ne[1] << "," << k->ne[2] << "," << k->ne[3] << "), " <<
                " v(" << v->ne[0] << "," << v->ne[1] << "," << v->ne[2] << "," << v->ne[3] << "), " <<
                " m(" << (m?m->ne[0]:0) << "," << (m?m->ne[1]:0) << "," << (m?m->ne[2]:0) << "," << (m?m->ne[3]:0) << ")";
            return name.str();
        }
        if (node->op == GGML_OP_TOP_K) {
            std::stringstream name;
            name << fusion_str;
            name << ggml_op_name(node->op) <<
                " K=" << node->ne[0] <<
                " (" << node->src[0]->ne[0] << "," << node->src[0]->ne[1] << "," << node->src[0]->ne[2] << "," << node->src[0]->ne[3] << ")";
            return name.str();
        }
        if (node->op == GGML_OP_CPY || node->op == GGML_OP_CONT || node->op == GGML_OP_DUP) {
            const ggml_tensor * src = node->src[0];
            const bool is_5d = (src->ne[4] > 1) || (node->ne[4] > 1);
            const bool contig = ggml_is_contiguous(src) && ggml_is_contiguous(node);
            std::stringstream name;
            name << fusion_str << ggml_op_name(node->op);
            name << " " << ggml_type_name(src->type) << "->" << ggml_type_name(node->type);
            name << " src(" << src->ne[0] << "," << src->ne[1] << "," << src->ne[2] << "," << src->ne[3] << "," << src->ne[4] << ")";
            name << " dst(" << node->ne[0] << "," << node->ne[1] << "," << node->ne[2] << "," << node->ne[3] << "," << node->ne[4] << ")";
            if (contig) {
                name << " [contig]";
            } else if (is_5d) {
                name << " [5d]";
            } else {
                name << " [4d]";
            }
            return name.str();
        }
        return fusion_str + ggml_op_name(node->op);
    }

    void log_timing(const ggml_tensor * node, const char *fusion_name, uint64_t time) {
        uint64_t n_flops;
        std::string name = get_node_fusion_name(node, fusion_name, &n_flops);
        if (n_flops) {
            flops[name].push_back(n_flops);
        }
        timings[name].push_back(time);
    }

    void log_timing(const std::vector<ggml_tensor *> &nodes, const std::vector<const char *> &names, uint64_t time) {
        uint64_t total_flops = 0;
        std::string name;
        for (size_t n = 0; n < nodes.size(); ++n) {
            uint64_t n_flops = 0;
            name += get_node_fusion_name(nodes[n], names[n], &n_flops);
            total_flops += n_flops;

            if (n != nodes.size() - 1) {
                name += ", ";
            }
        }
        if (total_flops) {
            flops[name].push_back(total_flops);
        }
        timings[name].push_back(time);
    }

  private:
    std::map<std::string, std::vector<uint64_t>> timings;
    std::map<std::string, std::vector<uint64_t>> flops;
    uint32_t print_count {};
};

struct ggml_backend_vk_context {
    std::string name;

    vk_device device;

    size_t semaphore_idx, event_idx;
    ggml_vk_garbage_collector gc;
    size_t prealloc_size_x, prealloc_size_y, prealloc_size_split_k, prealloc_size_add_rms_partials, prealloc_size_add_rms_partials_offset;
    vk_buffer prealloc_x, prealloc_y, prealloc_split_k, prealloc_add_rms_partials, sync_staging;
    vk::Fence fence, almost_ready_fence;
    bool submit_pending {};
    bool almost_ready_fence_pending {};
    // Set before op_add and unset after op_rms_norm to indicate that the add should
    // write partial sums to accumulate the square of the vector components
    bool do_add_rms_partials_offset_calculation;
    bool do_add_rms_partials;

    uint64_t last_total_flops {UINT64_MAX};

    // Cache most recent tensor that was converted into prealloc_y, and what pipeline it used to convert.
    vk_pipeline_struct * prealloc_y_last_pipeline_used {};
    const ggml_tensor * prealloc_y_last_tensor_used {};
    // True when the K dimension in prealloc_y is padded.
    bool prealloc_y_last_k_padded {};

    // Track which nodes have been used since the last sync, and whether they were written to
    std::vector<const ggml_tensor *> unsynced_nodes_written;
    std::vector<const ggml_tensor *> unsynced_nodes_read;
    // Track which prealloc buffers have pending reads that need to be synchronized.
    // These are checked before writing to the buffer (and call ggml_vk_sync_buffers if set),
    // and set to true after the buffer contents are consumed.
    bool prealloc_x_need_sync, prealloc_y_need_sync, prealloc_split_k_need_sync;

    vk_context_ref compute_ctx;

    vk_context_ref transfer_ctx;
    vk_semaphore transfer_semaphore;
    uint64_t transfer_semaphore_last_submitted {};

    std::vector<vk_context_ref> tensor_ctxs;

    std::vector<vk::DescriptorPool> descriptor_pools;
    std::vector<vk::DescriptorSet> descriptor_sets;
    // last bindings written to each set; descriptor_sets is append-only so an index always names the same set
    std::vector<std::vector<vk::DescriptorBufferInfo>> descriptor_set_bindings;
    uint64_t descriptor_set_bindings_destroy_count {};
    uint32_t descriptor_set_idx {};
    uint32_t pipeline_descriptor_set_requirements {};

    vk_command_pool compute_cmd_pool;
    vk_command_pool transfer_cmd_pool;

    // number of additional consecutive nodes that are being fused with the
    // node currently being processed
    int num_additional_fused_ops {};
    // Bitmask of which fused ops need to write an intermediate value to memory.
    // Bit 'i' means nodes[start_of_fusion + i] writes to memory.
    // If there's no fusion, bit 0 is still set.
    int fused_ops_write_mask {};
    topk_moe_mode fused_topk_moe_mode {};
    bool fused_topk_moe_scale {};
    // QSA indexer gather+add+top_k fused into one radix-select
    bool fused_topk_qsa {};
    bool fused_hc_post_gate {};
    rms_norm_mode fused_rms_norm_mode {RMS_NORM_COUNT};

    // for GGML_VK_PERF_LOGGER
    std::unique_ptr<vk_perf_logger> perf_logger;
    vk::QueryPool query_pool;
    std::vector<const char *> query_fusion_names;
    std::vector<int> query_fusion_node_count;
    std::vector<ggml_tensor *> query_nodes;
    std::vector<int> query_node_idx;
    int32_t num_queries {};
    int32_t query_idx {};
};

// Declarations carrying the default arguments (upstream: ggml-vulkan-common.h).
static void ggml_vk_host_get(const vk_device& device, const void * ptr, vk_buffer& buf, size_t& buf_offset);
static vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags = vk::MemoryPropertyFlags(0));
static vk_subbuffer ggml_vk_subbuffer(const ggml_backend_vk_context* ctx, const vk_buffer& buf, size_t offset = 0);
static void deferred_memcpy(void * dst, const void * src, size_t size, std::vector<vk_staging_memcpy>* memcpys = nullptr);
static void deferred_memset(void * dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets = nullptr);
static vk_subbuffer ggml_vk_tensor_subbuffer(const ggml_backend_vk_context * ctx, const ggml_tensor * tensor, bool allow_misalign = false);
static bool ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t size, bool sync_staging = false);
static bool ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging = false);
static bool ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging = false);

static void ggml_vk_print_device_fault_info(const vk_device& device) {
    if (!device->device_fault || !device->pfn_vkGetDeviceFaultInfoEXT) {
        return;
    }

    VkDeviceFaultCountsEXT fault_counts {};
    fault_counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    VkResult res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, nullptr);
    if (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (counts) failed: %d\n", res);
        return;
    }

    std::vector<VkDeviceFaultAddressInfoEXT> address_infos(fault_counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor_infos(fault_counts.vendorInfoCount);

    VkDeviceFaultInfoEXT fault_info {};
    fault_info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    fault_info.pAddressInfos = address_infos.data();
    fault_info.pVendorInfos = vendor_infos.data();

    res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, &fault_info);
    if (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (info) failed: %d\n", res);
        return;
    }

    if (fault_counts.addressInfoCount == 0 && fault_counts.vendorInfoCount == 0 && fault_info.description[0] == '\0') {
        return;
    }

    if (fault_info.description[0] != '\0') {
        GGML_LOG_ERROR("ggml_vulkan: device fault on %s: %s\n", device->name.c_str(), fault_info.description);
    }

    for (uint32_t i = 0; i < fault_counts.addressInfoCount; i++) {
        const auto& info = address_infos[i];
        GGML_LOG_CONT("  address fault %u: type=%d address=0x%llx precision=0x%llx\n",
                i, (int)info.addressType,
                (unsigned long long)info.reportedAddress,
                (unsigned long long)info.addressPrecision);
    }
    for (uint32_t i = 0; i < fault_counts.vendorInfoCount; i++) {
        const auto& info = vendor_infos[i];
        GGML_LOG_CONT("  vendor fault %u: %s (code=0x%llx data=0x%llx)\n",
                i, info.description,
                (unsigned long long)info.vendorFaultCode,
                (unsigned long long)info.vendorFaultData);
    }
}

static void ggml_vk_print_node_list(const ggml_cgraph * cgraph, int start, int end) {
    uint64_t total_flops = 0;
    int n_ops = 0;
    for (int j = start; j <= end && j < cgraph->n_nodes; j++) {
        uint64_t flops = ggml_vk_get_node_flops(cgraph->nodes[j]);
        total_flops += flops;
        n_ops++;
        if (flops > 0) {
            GGML_LOG_CONT("  node %d: %s (%s) [%.2f GFLOP]\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op),
                    flops / 1e9);
        } else {
            GGML_LOG_CONT("  node %d: %s (%s)\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op));
        }
    }
    GGML_LOG_CONT("  total: %d ops, %.2f GFLOP\n", n_ops, total_flops / 1e9);
}
static void ggml_vk_print_device_lost_info(const vk_device& device) {
    ggml_vk_print_device_fault_info(device);
    if (device->serialize_submissions && device->diag_cgraph != nullptr && device->diag_prev_start >= 0) {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s, likely caused by previous submission (nodes %d to %d):\n",
                device->name.c_str(), device->diag_prev_start, device->diag_prev_end);
        ggml_vk_print_node_list(device->diag_cgraph, device->diag_prev_start, device->diag_prev_end);
    } else {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s\n", device->name.c_str());
    }
}

// out-of-lined header method definitions (upstream: end of ggml-vulkan.cpp)


void vk_queue_handle_synchronized::submit(vk::ArrayProxy<const vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    if (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    std::lock_guard<std::mutex> guard(mutex);
    try {
        queue.submit(submits, fence);
    } catch (vk::DeviceLostError &) {
        if (auto dev = device.lock()) {
            ggml_vk_print_device_lost_info(dev);
        }
        throw;
    }
}

void vk_queue_handle_unsynchronized::submit(vk::ArrayProxy<const vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    if (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    try {
        queue.submit(submits, fence);
    } catch (vk::DeviceLostError &) {
        if (auto dev = device.lock()) {
            ggml_vk_print_device_lost_info(dev);
        }
        throw;
    }
}

static void * const vk_ptr_base = (void *)(uintptr_t) 0x1000;  // NOLINT

static uint64_t vk_tensor_offset(const ggml_tensor * tensor) {
    if (tensor->view_src) {
        return (uint8_t *) tensor->view_src->data - (uint8_t *) vk_ptr_base;
    }
    return (uint8_t *) tensor->data - (uint8_t *) vk_ptr_base;
}

static size_t ggml_vk_tensor_buffer_offset(const ggml_backend_vk_context * ctx, const ggml_tensor * t) {
    // vk_tensor_offset() is relative to vk_ptr_base, but mapped host tensors need an offset relative to their Vulkan buffer.
    if (ctx->device->uma) {
        vk_buffer buf = nullptr;
        size_t off = 0;
        ggml_vk_host_get(ctx->device, t->data, buf, off);
        if (buf) {
            return off;
        }
    }
    return (size_t)(vk_tensor_offset(t) + t->view_offs);
}
static size_t ggml_vk_descriptor_offset(size_t tensor_offset, size_t alignment, size_t type_size) {
    // Move the descriptor back until its distance to the tensor is divisible by the tensor type size.
    size_t descriptor_offset = tensor_offset & ~(alignment - 1);
    while ((tensor_offset - descriptor_offset) % type_size != 0) {
        GGML_ASSERT(descriptor_offset >= alignment);
        descriptor_offset -= alignment;
    }

    return descriptor_offset;
}

static uint32_t get_misalign_bytes(const ggml_backend_vk_context * ctx, const ggml_tensor * t) {
    const size_t tensor_offset = ggml_vk_tensor_buffer_offset(ctx, t);
    const size_t descriptor_offset = ggml_vk_descriptor_offset(
        tensor_offset, ctx->device->properties.limits.minStorageBufferOffsetAlignment, ggml_type_size(t->type));
    GGML_ASSERT(tensor_offset - descriptor_offset <= UINT32_MAX);
    return tensor_offset - descriptor_offset;
}

template <typename T> void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, T &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    GGML_UNUSED(p);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
    GGML_UNUSED(dst);
    static_assert(!std::is_const<T>::value, "unexpected type");
    GGML_ASSERT(!src0 || get_misalign_bytes(ctx, src0) == 0);
    GGML_ASSERT(!src1 || get_misalign_bytes(ctx, src1) == 0);
    GGML_ASSERT(!src2 || get_misalign_bytes(ctx, src2) == 0);
    GGML_ASSERT(!src3 || get_misalign_bytes(ctx, src3) == 0);
    GGML_ASSERT(!dst  || get_misalign_bytes(ctx, dst) == 0);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_glu_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    const uint32_t b_offset = src1 ? get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type) : a_offset;
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    GGML_ASSERT(a_offset < (1u << 8));
    GGML_ASSERT(b_offset < (1u << 8));
    GGML_ASSERT(d_offset < (1u << 8));

    p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_p021_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t b_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.b_offset = b_offset;
    p.d_offset = d_offset;

    GGML_UNUSED(src0);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_mat_vec_nc_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t b_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.b_offset = b_offset;
    p.d_offset = d_offset;

    GGML_UNUSED(src0);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv3d_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.src_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    p.dst_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_fwht_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    p.src_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.dst_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_rope_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    p.a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.d_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_conv2d_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    p.knl_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.src_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    p.dst_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

static uint32_t ggml_vk_concat_unit_size(ggml_type type) {
    const uint32_t type_size = ggml_type_size(type);

    if (!ggml_is_quantized(type)) {
        return type_size;
    }

    // Use the widest existing concat shader that evenly divides a quant block.
    if (type_size % 8 == 0) {
        return 8;
    }
    if (type_size % 4 == 0) {
        return 4;
    }
    if (type_size % 2 == 0) {
        return 2;
    }
    return 1;
}
static bool ggml_vk_concat_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != src1->type || src0->type != dst->type) {
        return false;
    }

    if (!ggml_is_quantized(src0->type)) {
        const size_t type_size = ggml_type_size(src0->type);
        return type_size == 1 || type_size == 2 || type_size == 4 || type_size == 8;
    }

    // Quantized tensor rows are block-aligned when created.
    return ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(src1) && ggml_is_contiguous_rows(dst);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_concat_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t unit_size = ggml_vk_concat_unit_size(dst->type);
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / unit_size;
    const uint32_t b_offset = get_misalign_bytes(ctx, src1) / unit_size;
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / unit_size;

    p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_comb_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    p.m_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.s_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    p.b_offset = get_misalign_bytes(ctx, src2) / ggml_type_size(src2->type);
    p.d_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_pre_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    p.x_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.w_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    p.d_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_dsv4_hc_post_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    p.x_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    p.r_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    p.p_offset = get_misalign_bytes(ctx, src2) / ggml_type_size(src2->type);
    p.c_offset = src3 ? get_misalign_bytes(ctx, src3) / ggml_type_size(src3->type) : 0;
    p.d_offset = get_misalign_bytes(ctx, dst)  / ggml_type_size(dst->type);
}


struct ggml_backend_vk_buffer_context {
    vk_device_ref device;
    vk_buffer dev_buffer;
    std::string name;

    ggml_backend_vk_buffer_context(vk_device_ref device, vk_buffer&& dev_buffer, std::string& name) :
        device(device),
        dev_buffer(dev_buffer),
        name(name) {
    }

    ~ggml_backend_vk_buffer_context();

};


struct vk_host_allocation {
    void *    base = nullptr;
    size_t    size = 0;
    vk_buffer views[GGML_VK_MAX_DEVICES];  // indexed by device->idx; null = not (yet) mapped here
};

struct vk_instance_t {
    vk::Instance instance;

    bool debug_utils_support = false;  // VK_EXT_debug_utils enabled
    PFN_vkSetDebugUtilsObjectNameEXT pfn_vkSetDebugUtilsObjectNameEXT = {};
    PFN_vkQueueBeginDebugUtilsLabelEXT pfn_vkQueueBeginDebugUtilsLabelEXT = {};
    PFN_vkQueueEndDebugUtilsLabelEXT   pfn_vkQueueEndDebugUtilsLabelEXT   = {};
    PFN_vkCmdBeginDebugUtilsLabelEXT   pfn_vkCmdBeginDebugUtilsLabelEXT   = {};
    PFN_vkCmdEndDebugUtilsLabelEXT pfn_vkCmdEndDebugUtilsLabelEXT = {};
    PFN_vkCmdInsertDebugUtilsLabelEXT  pfn_vkCmdInsertDebugUtilsLabelEXT  = {};

    std::vector<size_t> device_indices;
    std::vector<bool>   device_supports_membudget;
    vk_device devices[GGML_VK_MAX_DEVICES];

    std::vector<vk_host_allocation> host_allocations;
    std::shared_mutex               host_allocations_mutex;

    size_t host_import_alignment = 0;

    bool multi_device = false;
};

static constexpr uint32_t kSpvOpCooperativeMatrixLoadTensorNV = 5367;

static constexpr uint32_t kSpvCapabilityCooperativeMatrixDecodeVectorNV = 5447;

static constexpr uint32_t kSpvTensorAddressingDecodeVectorFuncBit = 0x4;


struct CompileTask {
    vk_pipeline pipeline;
    size_t spv_size;
    const void * spv_data;
    std::string entrypoint;
    uint32_t parameter_count;
    std::array<uint32_t, 3> wg_denoms;
    std::vector<uint32_t> specialization_constants;
    bool disable_robustness;
    bool require_full_subgroups;
    uint32_t required_subgroup_size;
};

struct ggml_vk_debug_label {
    // at most one of these is set, depending on the scope the label was opened in
    vk_context_struct * subctx {};
    vk_queue_handle *   qhandle {};

    // one region per dispatch, e.g. "matmul_q4_k_f32_f16acc_aligned_m (192,8,1)".
    // RGP cannot recover the pipeline name on its own, it only has the hash
    ggml_vk_debug_label(vk_context & ctx, const std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2);


    // one region per graph node
    // fused nodes are joined with '+', e.g. "RMS_NORM+MUL+ROPE Qcur-19"
    ggml_vk_debug_label(vk_context & ctx, const ggml_cgraph * cgraph, int node_idx, int n_fused);


    // one region per graph evaluation, opened on the queue instead of a command buffer
    // so it spans every submit the evaluation makes
    ggml_vk_debug_label(vk_queue_handle * handle, const char * name);


    // call before the command buffer can end, the destructor covers the rest
    void close();


    ~ggml_vk_debug_label() {
        close();
    }

    ggml_vk_debug_label(const ggml_vk_debug_label &) = delete;
    ggml_vk_debug_label & operator=(const ggml_vk_debug_label &) = delete;

private:
    // the constructors check this too, so the name is not built when markers are off
    void begin(vk_context & ctx, const std::string & name);

};

static bool vk_instance_initialized = false;
static vk_instance_t vk_instance;


typedef void (*ggml_vk_func_t)(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

static void ggml_backend_vk_free(ggml_backend_t backend);

static VkDeviceSize ggml_vk_get_max_buffer_range(const ggml_backend_vk_context * ctx, const vk_buffer &buf, const VkDeviceSize offset) {
    const VkDeviceSize range = std::min(VkDeviceSize{buf->size - offset},
                                        VkDeviceSize{ctx->device->properties.limits.maxStorageBufferRange});
    return range;
}

// Wait for ctx->fence to be signaled.
static void ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {
    // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
    // during this wait.
    if (ctx->almost_ready_fence_pending) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, true, UINT64_MAX), "almost_ready_fence", ctx->device);
        ctx->device->device.resetFences({ ctx->almost_ready_fence });
        ctx->almost_ready_fence_pending = false;
    }

    // Spin (w/pause) waiting for the graph to finish executing. A blocking
    // waitForFences here would let the CPU sleep, but adds thread-wakeup
    // latency (syscall + scheduler) that dominates wall time for short graphs
    // (small ONNX inference: SuperResolution, EmotionFerPlus), so we keep the
    // upstream low-latency spin. The almost_ready_fence waitForFences above
    // already lets the CPU sleep through the bulk of longer graphs.
    vk::Result result;
    for (;;) {
        try {
            result = ctx->device->device.getFenceStatus(ctx->fence);
        } catch (vk::DeviceLostError &) {
            ggml_vk_print_device_lost_info(ctx->device);
            GGML_ABORT("ggml_vulkan: device lost (getFenceStatus)");
        }
        if (result == vk::Result::eSuccess) {
            break;
        }
        if (result != vk::Result::eNotReady) {
            /* GGML_ABORT, not exit(1): both reach R as an error (exit is
             * redirected to r_ggml_exit above), but only this one carries the
             * fence status into the message and runs r_ggml_abort_hook. */
            GGML_ABORT("ggml_vulkan: fence status %s", to_string(result).c_str());
        }
        for (uint32_t i = 0; i < 100; ++i) {
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
        }
    }
    ctx->device->device.resetFences({ ctx->fence });
}


static bool ggml_vk_intel_windows_driver_in_range(uint32_t driver_version, uint32_t lower_major, uint32_t lower_minor, uint32_t upper_major, uint32_t upper_minor) {
#if defined(_WIN32)
    // Intel Windows encodes xxx.yyyy as [31:14].[13:0].
    const uint32_t major = driver_version >> 14;
    const uint32_t minor = driver_version & 0x3fff;

    const bool ge_lower = major > lower_major || (major == lower_major && minor >= lower_minor);
    const bool lt_upper = major < upper_major || (major == upper_major && minor < upper_minor);

    return ge_lower && lt_upper;
#else
    GGML_UNUSED(driver_version);
    GGML_UNUSED(lower_major);
    GGML_UNUSED(lower_minor);
    GGML_UNUSED(upper_major);
    GGML_UNUSED(upper_minor);
    return true;
#endif
}

static bool ggml_vk_strip_decode_vector(const uint32_t * code, size_t word_count, std::vector<uint32_t> & out) {
    static const char kDecodeVectorExt[] = "SPV_NV_cooperative_matrix_decode_vector";

    if (word_count < 5) {
        return false;
    }

    bool uses_decode_vector = false;
    for (size_t pos = 5; pos < word_count; ) {
        uint32_t word = code[pos];
        uint32_t wc   = word >> spv::WordCountShift;
        uint32_t op   = word & spv::OpCodeMask;
        GGML_ASSERT(wc > 0 && pos + wc <= word_count);
        if (op == spv::OpExtension && wc >= 2) {
            const char * s = reinterpret_cast<const char *>(&code[pos + 1]);
            if (strcmp(s, kDecodeVectorExt) == 0) {
                uses_decode_vector = true;
                break;
            }
        }
        pos += wc;
    }

    if (!uses_decode_vector) {
        return false;
    }

    VK_LOG_DEBUG("ggml_vk_strip_decode_vector: stripping SPV_NV_cooperative_matrix_decode_vector");

    // Bulk-copy unchanged runs and only break the run when an instruction needs to
    // be dropped or patched. Use reserve + insert/push_back so the destination buffer
    // is touched exactly once (no zero-initialization pass from resize()).
    out.clear();
    out.reserve(word_count);

    size_t run_start = 0;
    auto flush_run = [&](size_t up_to) {
        if (up_to > run_start) {
            out.insert(out.end(), code + run_start, code + up_to);
        }
    };

    for (size_t pos = 5; pos < word_count; ) {
        uint32_t word = code[pos];
        uint32_t wc   = word >> spv::WordCountShift;
        uint32_t op   = word & spv::OpCodeMask;
        GGML_ASSERT(wc > 0 && pos + wc <= word_count);

        if (op == spv::OpExtension && wc >= 2) {
            const char * s = reinterpret_cast<const char *>(&code[pos + 1]);
            if (strcmp(s, kDecodeVectorExt) == 0) {
                flush_run(pos);
                pos += wc;
                run_start = pos;
                continue;
            }
        }

        if (op == spv::OpCapability && wc == 2 && code[pos + 1] == kSpvCapabilityCooperativeMatrixDecodeVectorNV) {
            flush_run(pos);
            pos += wc;
            run_start = pos;
            continue;
        }

        if (op == kSpvOpCooperativeMatrixLoadTensorNV) {
            // [opcode/wc][ResultType][Result][Pointer][Object][TensorLayout][MemOperand mask][mem extras...][TA mask][ta extras...]
            GGML_ASSERT(wc >= 8);

            uint32_t mem_mask = code[pos + 6];
            size_t   cur      = pos + 7;
            // Each of these MemoryAccess bits (when set) carries one trailing operand.
            cur += (mem_mask & 0x2)     ? 1 : 0; // Aligned
            cur += (mem_mask & 0x8)     ? 1 : 0; // MakePointerAvailable
            cur += (mem_mask & 0x10)    ? 1 : 0; // MakePointerVisible
            cur += (mem_mask & 0x10000) ? 1 : 0; // AliasScopeINTELMask
            cur += (mem_mask & 0x20000) ? 1 : 0; // NoAliasINTELMask
            GGML_ASSERT(cur < pos + wc);

            uint32_t ta_mask = code[cur];
            if ((ta_mask & kSpvTensorAddressingDecodeVectorFuncBit) == 0) {
                pos += wc;
                continue; // leave instruction inside the current unchanged run
            }

            flush_run(pos);

            // Append unchanged prefix of the instruction (header through the mem-extras).
            size_t inst_start = out.size();
            size_t pre_n      = cur - pos;
            out.insert(out.end(), code + pos, code + pos + pre_n);

            // Emit TA mask with the DecodeVectorFunc bit cleared.
            out.push_back(ta_mask & ~kSpvTensorAddressingDecodeVectorFuncBit);

            // TA extras: TensorView (0x1) and DecodeFunc (0x2) are kept verbatim;
            // DecodeVectorFunc (0x4) is dropped along with its trailing id operand.
            size_t keep_ta_extras = ((ta_mask & 0x1) ? 1 : 0) + ((ta_mask & 0x2) ? 1 : 0);
            if (keep_ta_extras) {
                out.insert(out.end(), code + cur + 1, code + cur + 1 + keep_ta_extras);
            }

            GGML_ASSERT(wc == pre_n + 1 + keep_ta_extras + 1);

            // Patch the instruction header with the new (one-shorter) word count.
            uint32_t new_wc = wc - 1;
            out[inst_start] = (new_wc << spv::WordCountShift) | op;

            pos += wc;
            run_start = pos;
            continue;
        }

        pos += wc;
    }

    flush_run(word_count);
    return true;
}

static bool ggml_vk_roll_bk_loop(const uint32_t * code, size_t word_count, std::vector<uint32_t> & out) {
    if (word_count < 5) {
        return false;
    }

    struct vk_spv_loop {
        size_t   header;
        size_t   end;
        uint32_t control;
    };

    std::vector<vk_spv_loop> loops;

    // Collect a list of all loops in the module.
    for (size_t pos = 5; pos < word_count; ) {
        const uint32_t wc = code[pos] >> spv::WordCountShift;
        const uint32_t op = code[pos] & spv::OpCodeMask;
        if (wc == 0 || pos + wc > word_count) {
            return false;
        }

        if (op == spv::OpLoopMerge && wc >= 4) { loops.push_back({ pos, 0, code[pos + 3] }); }

        if (op == spv::OpLabel && wc >= 2) {
            for (auto & l : loops) {
                if (l.end == 0 && code[l.header + 1] == code[pos + 1]) { l.end = pos; }
            }
        }

        pos += wc;
    }

    auto encloses = [](const vk_spv_loop & a, const vk_spv_loop & b) {
        return a.header < b.header && b.header < a.end;
    };

    // Find the BK loop.
    const vk_spv_loop * bk = nullptr;
    for (const auto & h : loops) {
        if (h.control != spv::LoopControlUnrollMask) {
            continue;
        }
        const vk_spv_loop * parent = nullptr;
        bool has_child = false;
        for (const auto & g : loops) {
            if (encloses(g, h) && (!parent || g.header > parent->header)) {
                parent = &g;
            }
            if (encloses(h, g)) {
                has_child = true;
            }
        }
        // BK loop should be the last loop nested inside the loop with no hint
        // and have at least one child loop.
        if (parent &&
            parent->control == spv::LoopControlMaskNone &&
            has_child &&
            (!bk || h.header > bk->header)) {
            bk = &h;
        }
    }
    if (!bk) {
        return false;
    }

    // set DontUnroll instead of Unroll
    out.assign(code, code + word_count);
    out[bk->header + 3] = spv::LoopControlDontUnrollMask;
    return true;
}

static void ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_size, const void* spv_data, const std::string entrypoint,
                                         uint32_t parameter_count, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         bool disable_robustness, bool require_full_subgroups, uint32_t required_subgroup_size) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_count <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_full_subgroups << ", " << required_subgroup_size << ")");
    GGML_ASSERT(parameter_count > 0);
    GGML_ASSERT(parameter_count <= MAX_PARAMETER_COUNT);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_size, reinterpret_cast<const uint32_t *>(spv_data));

    // Patch SPIR-V to enable supported FP16 float controls, avoiding the need
    // for separate shader variants.
    std::vector<uint32_t> spirv;
    if (device->float_controls_rte_fp16 || device->float_controls_denorm_preserve_fp16) {
        const uint32_t* spv_words = reinterpret_cast<const uint32_t *>(spv_data);
        size_t word_count = spv_size / sizeof(uint32_t);
        spirv.assign(spv_words, spv_words + word_count);

        // Find insertion points respecting SPIR-V layout order:
        //   Header(5) -> OpCapability -> OpExtension -> ... -> OpEntryPoint -> OpExecutionMode -> ...
        size_t pos = 5; // skip header
        size_t cap_insert_pos = pos;
        size_t ext_insert_pos = pos;
        size_t exec_insert_pos = pos;
        uint32_t entry_point_id = 0;

        while (pos < spirv.size()) {
            uint32_t opcode = spirv[pos] & spv::OpCodeMask;
            uint32_t len    = spirv[pos] >> spv::WordCountShift;
            if (len == 0) break;

            if (opcode == spv::OpCapability) {
                cap_insert_pos = pos + len;
                ext_insert_pos = pos + len;
            } else if (opcode == spv::OpExtension) {
                ext_insert_pos = pos + len;
            } else if (opcode == spv::OpEntryPoint) {
                entry_point_id = spirv[pos + 2];
                exec_insert_pos = pos + len;
            } else if (opcode == spv::OpExecutionMode || opcode == spv::OpExecutionModeId) {
                exec_insert_pos = pos + len;
            } else if (entry_point_id != 0) {
                break;
            }

            pos += len;
        }

        // Insert from latest position first so earlier indices stay valid.

        if (device->float_controls_rte_fp16) {
            // OpExecutionMode %entrypoint RoundingModeRTE 16
            uint32_t exec_mode[] = { (4u << spv::WordCountShift) | spv::OpExecutionMode, entry_point_id, spv::ExecutionModeRoundingModeRTE, 16 };
            spirv.insert(spirv.begin() + exec_insert_pos, std::begin(exec_mode), std::end(exec_mode));
        }

        if (device->float_controls_denorm_preserve_fp16) {
            // OpExecutionMode %entrypoint DenormPreserve 16
            uint32_t exec_mode[] = { (4u << spv::WordCountShift) | spv::OpExecutionMode, entry_point_id, spv::ExecutionModeDenormPreserve, 16 };
            spirv.insert(spirv.begin() + exec_insert_pos, std::begin(exec_mode), std::end(exec_mode));
        }

        // OpExtension "SPV_KHR_float_controls"
        const char ext_str[] = "SPV_KHR_float_controls";
        size_t ext_str_words = CEIL_DIV(sizeof(ext_str), sizeof(uint32_t));
        std::vector<uint32_t> extension(1 + ext_str_words, 0);
        extension[0] = (uint32_t)((1 + ext_str_words) << spv::WordCountShift) | spv::OpExtension;
        memcpy(&extension[1], ext_str, sizeof(ext_str));
        spirv.insert(spirv.begin() + ext_insert_pos, extension.begin(), extension.end());

        if (device->float_controls_rte_fp16) {
            // OpCapability RoundingModeRTE
            uint32_t capability[] = { (2u << spv::WordCountShift) | spv::OpCapability, spv::CapabilityRoundingModeRTE };
            spirv.insert(spirv.begin() + cap_insert_pos, std::begin(capability), std::end(capability));
        }

        if (device->float_controls_denorm_preserve_fp16) {
            // OpCapability DenormPreserve
            uint32_t capability[] = { (2u << spv::WordCountShift) | spv::OpCapability, spv::CapabilityDenormPreserve };
            spirv.insert(spirv.begin() + cap_insert_pos, std::begin(capability), std::end(capability));
        }

        shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * sizeof(uint32_t), spirv.data());
    }

#if defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)
    if (device->coopmat2 && !device->coopmat2_decode_vector) {
        const uint32_t * src   = spirv.empty() ? reinterpret_cast<const uint32_t *>(spv_data) : spirv.data();
        size_t           src_n = spirv.empty() ? spv_size / sizeof(uint32_t) : spirv.size();
        std::vector<uint32_t> stripped;
        if (ggml_vk_strip_decode_vector(src, src_n, stripped)) {
            spirv = std::move(stripped);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * sizeof(uint32_t), spirv.data());
        }
    }
#endif

#if VK_HEADER_VERSION >= 287
    // Roll the mul_mm BK loop on Asahi Linux. Skip bf16 and the mul_mmq pipelines.
    if (device->driver_id == vk::DriverId::eMesaHoneykrisp &&
        pipeline->name.rfind("matmul", 0) == 0 &&
        pipeline->name.find("bf16") == std::string::npos &&
        pipeline->name.find("q8_1") == std::string::npos) {
        const uint32_t * src   = spirv.empty() ? reinterpret_cast<const uint32_t *>(spv_data) : spirv.data();
        size_t           src_n = spirv.empty() ? spv_size / sizeof(uint32_t) : spirv.size();
        std::vector<uint32_t> rolled;
        if (ggml_vk_roll_bk_loop(src, src_n, rolled)) {
            spirv = std::move(rolled);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * sizeof(uint32_t), spirv.data());
        }
    }
#endif

    try {
        pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);
    } catch (const vk::SystemError& e) {
        GGML_LOG_ERROR("ggml_vulkan: shader module creation failed for %s: %s\n", pipeline->name.c_str(), e.what());
        throw;
    }

    vk::PushConstantRange pcr(
        vk::ShaderStageFlagBits::eCompute,
        0,
        pipeline->push_constant_size
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);

    std::vector<vk::SpecializationMapEntry> specialization_entries(specialization_constants.size());

    for (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].offset = i * sizeof(uint32_t);
        specialization_entries[i].size = sizeof(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.data(),
        specialization_constants.size() * sizeof(uint32_t),
        specialization_constants.data()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    if (device->subgroup_require_full_support && require_full_subgroups) {
        pipeline_shader_stage_create_flags |= vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::ShaderStageFlagBits::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT pipeline_shader_stage_required_subgroup_size_create_info;
    pipeline_shader_stage_required_subgroup_size_create_info.requiredSubgroupSize = required_subgroup_size;
    if (device->subgroup_size_control && required_subgroup_size > 0) {
        GGML_ASSERT(device->subgroup_min_size <= required_subgroup_size && required_subgroup_size <= device->subgroup_max_size);
        pipeline_shader_create_info.setPNext(&pipeline_shader_stage_required_subgroup_size_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        device->pipeline_executable_properties_support ?
            vk::PipelineCreateFlagBits::eCaptureStatisticsKHR :
            vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    if (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
        rci.uniformBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
        compute_pipeline_create_info.setPNext(&rci);
    }

#if defined(VK_EXT_shader_64bit_indexing)
    vk::PipelineCreateFlags2CreateInfo pipelineFlags2CreateInfo;
    if (pipeline->is_64b_indexing)
    {
        pipelineFlags2CreateInfo.flags = vk::PipelineCreateFlagBits2::e64BitIndexingEXT;
        if (device->pipeline_executable_properties_support) {
            pipelineFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eCaptureStatisticsKHR;
        }
        pipelineFlags2CreateInfo.setPNext(compute_pipeline_create_info.pNext);
        compute_pipeline_create_info.setPNext(&pipelineFlags2CreateInfo);
    }
#endif

    try {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Compute pipeline creation failed for " << pipeline->name << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }

    if (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vkSetDebugUtilsObjectNameEXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    if (device->pipeline_executable_properties_support) {
        vk::PipelineExecutableInfoKHR executableInfo;
        executableInfo.pipeline = pipeline->pipeline;

        auto statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);
        for (auto & s : statistics) {
            // "Register Count" is reported by NVIDIA drivers.
            if (strcmp(s.name, "Register Count") == 0) {
                VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
                pipeline->register_count = (uint32_t)s.value.u64;
            }
        }
    }

    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        device->all_pipelines.push_back(pipeline);
        pipeline->compiled = true;
        pipeline->compile_pending = false;
    }
    device->compile_cv.notify_all();
}

static void ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
    ctx->pipeline_descriptor_set_requirements += n;
    if (!pipeline->compiled) {
        ggml_vk_load_shaders(ctx->device, pipeline);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx);
}

static void ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    if (ctx->device->push_descriptors) {
        // Push descriptors — no descriptor pool allocation needed
        return;
    }

    if (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_set_requirements) {
        // Enough descriptors are available
        return;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_set_requirements});
    uint32_t to_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_SIZE;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_SIZE;

    while (to_alloc > 0) {
        const uint32_t alloc_count = std::min(pool_remaining, to_alloc);
        to_alloc -= alloc_count;
        pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE;

        if (pool_idx >= ctx->descriptor_pools.size()) {
            vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, MAX_PARAMETER_COUNT * VK_DEVICE_DESCRIPTOR_POOL_SIZE);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_SIZE, descriptor_pool_size);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
        for (uint32_t i = 0; i < alloc_count; i++) {
            layouts[i] = device->dsl;
        }
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(ctx->descriptor_pools[pool_idx], alloc_count, layouts.data());
        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
    ctx->descriptor_set_bindings.resize(ctx->descriptor_sets.size());
}

static vk_command_buffer* ggml_vk_create_cmd_buffer(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");
    vk::CommandBufferAllocateInfo command_buffer_alloc_info(
        p.pool,
        vk::CommandBufferLevel::ePrimary,
        1);
    const std::vector<vk::CommandBuffer> cmd_buffers = device->device.allocateCommandBuffers(command_buffer_alloc_info);
    p.cmd_buffers.push_back({ cmd_buffers.front(), 0, true });
    return &p.cmd_buffers[p.cmd_buffers.size()-1];
}

// Get a command buffer from pool. Create a new one if no reusable buffer is available
static vk_command_buffer* ggml_vk_get_or_create_cmd_buffer(vk_device& device, vk_command_pool& pool) {
    for (auto& cmd_buffer : pool.cmd_buffers) {
        if (!cmd_buffer.in_use) {
            cmd_buffer.use_counter++;
            cmd_buffer.in_use = true;
            return &cmd_buffer;
        }
    }
    return ggml_vk_create_cmd_buffer(device, pool);
}

static void ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    if (ctx->seqs.empty()) {
        if (fence) {
            ctx->p->q->handle->submit({}, fence);
        }
        return;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    int idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    for (const auto& sequence : ctx->seqs) {
        reserve += sequence.size();
    }

    // Pre-reserve vectors to prevent reallocation, which invalidates pointers
    tl_wait_semaphores.reserve(reserve);
    tl_wait_vals.reserve(reserve);
    tl_signal_semaphores.reserve(reserve);
    tl_signal_vals.reserve(reserve);
    tl_submit_infos.reserve(reserve);
    submit_infos.reserve(reserve);
    stage_flags.reserve(reserve);

    for (const auto& sequence : ctx->seqs) {
        for (const auto& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            for (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->p->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            for (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].data(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].data(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].pNext = nullptr;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].data(),
                stage_flags[idx].data(),
                1,
                &submission.buffer->buf,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].data(),
            };
            si.setPNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    ctx->p->q->handle->submit(submit_infos, fence);

    ctx->seqs.clear();
}

static uint32_t ggml_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props, const vk::QueueFlags& required, const vk::QueueFlags& avoid, int32_t compute_index, uint32_t min_num_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    const uint32_t qfsize = queue_family_props.size();

    // Try with avoid preferences first
    for (uint32_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required && !(queue_family_props[i].queueFlags & avoid)) {
            return i;
        }
    }

    // Fall back to only required
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // Fall back to reusing compute queue
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // Fall back to ignoring min_num_queries
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    if (compute_index >= 0) {
        return compute_index;
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::endl;

    for(auto &q_family : queue_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.queueCount) << " flags: " + to_string(q_family.queueFlags) << std::endl;
    }
    abort();
}

static std::unique_ptr<vk_queue> ggml_vk_create_queue(vk_device& device, uint32_t queue_family_index, uint32_t queue_index, vk::PipelineStageFlags&& stage_flags, bool transfer_only) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    auto q = std::make_unique<vk_queue>();
    q->queue_family_index = queue_family_index;
    q->transfer_only = transfer_only;

    std::shared_ptr<vk_queue_handle> h;
    vk::DeviceQueueInfo2 queue_info2{};
    queue_info2.queueFamilyIndex = queue_family_index;
    queue_info2.queueIndex = queue_index;

    if (device->has_internally_synchronized_queues) {
        h = std::make_shared<vk_queue_handle_unsynchronized>();
        queue_info2.flags = eInternallySynchronizedKHR;
    } else {
        h = std::make_shared<vk_queue_handle_synchronized>();
    }

    h->queue = device->device.getQueue2(queue_info2);
    h->device = device;
    // Avoid concurrent submissions on NVIDIA due to driver bug.
    if (device->vendor_id == VK_VENDOR_ID_NVIDIA) {
        h->device_submit_mutex = &device->queue_submit_mutex;
    }
    q->handle = h;

    q->cmd_pool.init(device, q.get());

    q->stage_flags = stage_flags;
    return q;
}

static std::unique_ptr<vk_queue> ggml_vk_create_aliased_queue(vk_device& device, const std::unique_ptr<vk_queue>& source) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    auto q = std::make_unique<vk_queue>();
    q->handle = source->handle;
    q->queue_family_index = source->queue_family_index;
    q->stage_flags = source->stage_flags;
    q->transfer_only = source->transfer_only;
    q->cmd_pool.init(device, q.get());
    return q;
}


static vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p) {
    vk_context result = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << result << ")");
    ctx->gc.contexts.emplace_back(result);
    result->p = &p;
    return result;
}

static vk_context ggml_vk_create_temporary_context(vk_command_pool& p) {
    vk_context result = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << result << ")");
    result->p = &p;
    return result;
}

static vk_semaphore * ggml_vk_create_binary_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eBinary, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.setPNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    return &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

static vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    if (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.setPNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    return &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

static vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    if (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    return ctx->gc.events[ctx->event_idx++];
}

static void ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    // Don't clear the command buffers and mark them as not in use.
    // This allows us to reuse them
    for (auto& cmd_buffer : p.cmd_buffers) {
        cmd_buffer.in_use = false;
    }
}

static void ggml_vk_queue_command_pools_cleanup(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_queue_command_pools_cleanup()");

    // Arbitrary frequency to cleanup/reuse command buffers
    static constexpr uint32_t cleanup_frequency = 10;

    if (device->compute_queue && device->compute_queue->cmd_pool.buffers_in_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->compute_queue->cmd_pool);
    }
    if (device->transfer_queue && device->transfer_queue->cmd_pool.buffers_in_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->transfer_queue->cmd_pool);
    }
}

static std::vector<uint32_t> ggml_vk_find_memory_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        if ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    return indices;
}

static vk_buffer ggml_vk_create_buffer(vk_device& device, size_t size, const std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       void *import_ptr = nullptr,
                                       // ggmlR Tensor Parallelism (P2P), not upstream:
                                       //   export_fd  — allocate memory exportable as an opaque fd (owner side)
                                       //   import_fd  — import device memory from an opaque fd (consumer side); -1 = none
                                       //   import_type_index — memory type index the exporter used; when >= 0 the
                                       //                       fd import binds to exactly this type instead of
                                       //                       guessing from the mask (getMemoryFdPropertiesKHR is
                                       //                       unreliable on NVIDIA). -1 = fall back to the mask.
                                       bool export_fd = false,
                                       int import_fd = -1,
                                       int import_type_index = -1) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    if (size > device->max_buffer_size) {
        throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    if (size == 0) {
        buf->size = 0;
        return buf;
    }

    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    if (device->buffer_device_address) {
        usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }

    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        usage_flags,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    vk::ExternalMemoryBufferCreateInfo external_memory_bci;
    if (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        buffer_create_info.setPNext(&external_memory_bci);
    } else if (export_fd || import_fd >= 0) {
        // ggmlR Tensor Parallelism (P2P): mark the buffer as backed by opaque-fd
        // external memory so it can be exported/imported across Vulkan devices.
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
        buffer_create_info.setPNext(&external_memory_bci);
    }

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    const vk::MemoryPriorityAllocateInfoEXT mem_priority_info { 1.0f };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    if (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    // ggmlR Tensor Parallelism (P2P): NVIDIA (and some other drivers) require a
    // DEDICATED allocation for opaque-fd memory shared across physical devices —
    // otherwise the import on the consumer device silently binds to unrelated
    // memory (reads back as zeros). Tie the allocation to this specific buffer on
    // BOTH the export and import sides. dedicated_info.buffer must be the buffer
    // handle just created above (buf->buffer). Not upstream ggml.
    vk::MemoryDedicatedAllocateInfo dedicated_info;
    const bool want_dedicated = export_fd || import_fd >= 0;
    if (want_dedicated) {
        dedicated_info.buffer = buf->buffer;
        dedicated_info.setPNext(&mem_flags_info);
    }

    // ggmlR Tensor Parallelism (P2P): when exporting, chain a
    // VkExportMemoryAllocateInfo onto the allocation so the resulting device
    // memory can later be shared as an opaque fd via getMemoryFdKHR(). Placed at
    // the head of the pNext chain (before dedicated_info). Not upstream ggml.
    vk::ExportMemoryAllocateInfo export_mem_info;
    if (export_fd) {
        export_mem_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
        export_mem_info.setPNext(&dedicated_info);
    }
    vk::MemoryAllocateFlagsInfo * alloc_pnext = export_fd
        ? reinterpret_cast<vk::MemoryAllocateFlagsInfo *>(&export_mem_info)
        : &mem_flags_info;

    if (import_fd >= 0) {
        // ggmlR Tensor Parallelism (P2P): import device memory from an opaque fd
        // exported by the owning device. Ownership of the fd transfers to the
        // Vulkan driver on a successful import (do not close it here).
        //
        // For cross-device P2P the import should use a memory type the driver
        // reports as valid for THIS fd, via vkGetMemoryFdPropertiesKHR. Reusing
        // the buffer's generic memoryTypeBits picks an index that matches on
        // loopback (same physical device) but may be wrong on another GPU.
        //
        // BEST-EFFORT: this query returns ErrorUnknown on some NVIDIA drivers
        // (it does not dispatch for opaque fd the way we call it). Treat it as
        // advisory only: narrow memoryTypeBits ONLY when the query succeeds and
        // yields a non-empty intersection; otherwise keep the full mask and let
        // allocateMemory below try each candidate. Never fail the import here.
        // (getMemoryFdPropertiesKHR does NOT consume the fd.)
        const uint32_t orig_type_bits = mem_req.memoryTypeBits;

        // PRIMARY PATH (ggmlR TP): the exporter told us which memory type index it
        // actually used. Because every device in the split is the same physical
        // model, that index is valid on this device too, and it is the ONLY index
        // guaranteed to alias the exported allocation. Bind to exactly it — a
        // wrong-but-DEVICE_LOCAL type imports "successfully" yet reads back as all
        // zeros (the symptom this replaces). Only fall through to the mask scan
        // when no index was handed over.
        if (import_type_index >= 0) {
            const uint32_t mti = (uint32_t) import_type_index;
            if (mti >= mem_props.memoryTypeCount || !(orig_type_bits & (1u << mti))) {
                // The buffer's own requirements do not admit this type — the two
                // devices disagree; there is nothing safe to import onto.
                device->device.destroyBuffer(buf->buffer);
                GGML_LOG_ERROR("ggml_vulkan[TP]: fd-import type index %u invalid for this device (buffer bits=0x%x, count=%u)\n",
                               mti, orig_type_bits, mem_props.memoryTypeCount);
                throw vk::OutOfDeviceMemoryError("fd-import: exporter memory type index not usable on this device");
            }
            vk::ImportMemoryFdInfoKHR import_fd_info;
            import_fd_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
            import_fd_info.fd = import_fd;
            import_fd_info.setPNext(&dedicated_info);
            try {
                buf->device_memory = device->device.allocateMemory({ mem_req.size, mti, &import_fd_info });
            } catch (const vk::SystemError& e) {
                device->device.destroyBuffer(buf->buffer);
                throw e;
            }
            buf->memory_property_flags = mem_props.memoryTypes[mti].propertyFlags;
            buf->memory_type_index     = mti;
            GGML_LOG_INFO("ggml_vulkan[TP]: fd-import bound to exporter memory type index %u\n", mti);
        } else {
        // For cross-device P2P the import should use a memory type the driver
        // reports as valid for THIS fd, via vkGetMemoryFdPropertiesKHR. Reusing
        // the buffer's generic memoryTypeBits picks an index that matches on
        // loopback (same physical device) but may be wrong on another GPU.
        //
        // BEST-EFFORT: this query returns ErrorUnknown on some NVIDIA drivers
        // (it does not dispatch for opaque fd the way we call it). Treat it as
        // advisory only: narrow memoryTypeBits ONLY when the query succeeds and
        // yields a non-empty intersection; otherwise keep the full mask and let
        // allocateMemory below try each candidate. Never fail the import here.
        // (getMemoryFdPropertiesKHR does NOT consume the fd.)
        try {
            vk::MemoryFdPropertiesKHR fd_props = device->device.getMemoryFdPropertiesKHR(
                vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd, import_fd);
            const uint32_t narrowed = orig_type_bits & fd_props.memoryTypeBits;
            if (narrowed != 0) {
                mem_req.memoryTypeBits = narrowed;
            }
            GGML_LOG_INFO("ggml_vulkan[TP]: fd-import memoryTypeBits buffer=0x%x fd=0x%x -> usable=0x%x%s\n",
                          orig_type_bits, (unsigned) fd_props.memoryTypeBits, mem_req.memoryTypeBits,
                          narrowed == 0 ? " (empty intersection; kept full mask)" : "");
        } catch (const vk::SystemError& e) {
            GGML_LOG_INFO("ggml_vulkan[TP]: getMemoryFdPropertiesKHR unavailable (%s); keeping full memoryTypeBits=0x%x\n",
                          e.what(), orig_type_bits);
        }
        for (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            const auto & req_flags = *it;
            const std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);
            if (memory_type_indices.empty()) {
                continue;
            }
            bool done = false;
            for (auto mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                try {
                    vk::ImportMemoryFdInfoKHR import_fd_info;
                    import_fd_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
                    import_fd_info.fd = import_fd;
                    // ggmlR TP: chain through the dedicated-allocation info so the
                    // imported memory binds to this dst buffer (see export side).
                    import_fd_info.setPNext(&dedicated_info);
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, &import_fd_info });
                    buf->memory_type_index = *mtype_it;
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    done = true;
                    break;
                } catch (const vk::SystemError& e) {
                    if (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroyBuffer(buf->buffer);
                        throw e;
                    }
                }
            }
            if (done) {
                break;
            }
        }
        }
    } else if (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        try {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } catch (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroyBuffer(buf->buffer);
            return {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        for (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            if (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }
            if (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            if ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                break;
            }
        }
        if (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroyBuffer(buf->buffer);
            return {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        try {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
        } catch (const vk::SystemError& e) {
        }
    } else {
        for (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            const auto & req_flags = *it;

            const std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            if (memory_type_indices.empty()) {
                continue;
            }

            bool done = false;

            for (auto mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                try {
                    // alloc_pnext == &mem_flags_info normally, or the export-info
                    // head when export_fd is set (ggmlR TP, not upstream).
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, alloc_pnext });
                    buf->memory_type_index = *mtype_it;  // ggmlR TP: hand to importer
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    done = true;
                    break;
                } catch (const vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    if (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroyBuffer(buf->buffer);
                        throw e;
                    }
                }
            }

            if (done) {
                break;
            }
        }
    }

    if (!buf->device_memory) {
        device->device.destroyBuffer(buf->buffer);
        throw vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    if (import_ptr) {
        buf->ptr = import_ptr;
    } else if (import_fd < 0) {
        // Do not host-map fd-imported memory (ggmlR TP): it is remote device
        // memory shared for compute, not intended for CPU access here.
        if (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
        }
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;
    buf->exportable = export_fd;  // ggmlR TP: eligible for getMemoryFdKHR()

    if (device->buffer_device_address) {
        const vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    } else {
    }

    return buf;
}

// ggmlR Tensor Parallelism (P2P), not upstream ggml. Export an opaque fd for a
// buffer allocated with export_fd=true. The returned fd is owned by the caller;
// it must be passed to ggml_vk_create_buffer(import_fd=...) on the consumer
// device (which takes ownership) or closed. Returns -1 on failure.
static int ggml_vk_buffer_export_fd(vk_buffer& buf) {
    if (!buf || buf->size == 0 || !buf->exportable) {
        return -1;
    }
    if (!buf->device->external_memory_fd) {
        return -1;
    }
    try {
        vk::MemoryGetFdInfoKHR get_fd_info;
        get_fd_info.memory     = buf->device_memory;
        get_fd_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
        return buf->device->device.getMemoryFdKHR(get_fd_info);
    } catch (const vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: getMemoryFdKHR failed (%s)\n", e.what());
        return -1;
    }
}

static vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags) {
    try {
        return ggml_vk_create_buffer(device, size, {req_flags, fallback_flags});
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
}

static vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
    vk_buffer buf;
    try {
        if (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } else if (device->uma) {
            // DIVERGENCE: upstream 0.27 tries DeviceLocal|HostVisible|HostCoherent first on UMA
            // ("direct tensor borrowing", part of buffer_from_host_ptr). ggmlR keeps 0.11 order
            // until buffer_from_host_ptr is enabled with a measurement (TODO 9.5-later).
            // Fall back to host memory type
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } else if (device->disable_host_visible_vidmem) {
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        } else {
            // use rebar if available, otherwise fallback to device only visible memory
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        }
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }

    return buf;
}

static void ggml_vk_destroy_buffer(vk_buffer& buf) {
    if (buf == nullptr) {
        return;
    }


    buf.reset();
}

ggml_backend_vk_buffer_context::~ggml_backend_vk_buffer_context() {
    ggml_vk_destroy_buffer(dev_buffer);
}

static vk_subbuffer ggml_vk_subbuffer(const ggml_backend_vk_context* ctx, const vk_buffer& buf, size_t offset) {
    return { buf, offset, ggml_vk_get_max_buffer_range(ctx, buf, offset) };
}

static void ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");

    const bool transfer_queue = subctx->p->q->transfer_only;

    if (ctx) {
        ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;
    }

    subctx->s->buffer->buf.pipelineBarrier(
        subctx->p->q->stage_flags,
        subctx->p->q->stage_flags,
        {},
        { {
          { !transfer_queue ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) },
          { !transfer_queue ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) }
        } },
        {},
        {}
    );
}

static void ggml_vk_set_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->buffer->buf.setEvent(
        event,
        ctx->p->q->stage_flags
    );
}

static void ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    if (events.empty()) {
        return;
    }

    ctx->s->buffer->buf.waitEvents(
        events,
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        {},
        {}
    );
}

// number of rows/cols for flash attention shader
static constexpr uint32_t flash_attention_num_small_rows = 32;
static constexpr uint32_t scalar_flash_attention_num_small_rows = 1;

static uint32_t get_fa_scalar_num_large_rows(uint32_t hsk, uint32_t hsv, bool small_cache) {
    if (hsv >= 192) {
        return 2;
    } else if ((hsv | hsk) & 8 || small_cache) {
        return 4;
    } else {
        return 8;
    }
}

// The FA coopmat1 shader assumes 16x16x16 matrix multiply support.
// 128 threads split into four subgroups, each subgroup does 1/4
// of the Bc dimension.
static constexpr uint32_t coopmat1_flash_attention_num_large_rows = 16;
static constexpr uint32_t scalar_flash_attention_Bc = 64;
static constexpr uint32_t scalar_flash_attention_workgroup_size = 128;

static uint32_t get_fa_num_small_rows(FaCodePath path) {
    if (path == FA_COOPMAT2) {
        return flash_attention_num_small_rows;
    } else {
        return scalar_flash_attention_num_small_rows;
    }
}

static std::array<uint32_t, 2> fa_rows_cols(FaCodePath path, uint32_t hsk, uint32_t hsv, uint32_t clamp, ggml_type type, bool small_rows, bool small_cache) {
    GGML_UNUSED(clamp);

    if (path == FA_SCALAR) {
        if (small_rows) {
            return {scalar_flash_attention_num_small_rows, 64};
        } else {
            if ((hsv | hsk) & 8) {
                // HSV/HSK not being a multiple of 16 makes D_split smaller, which makes cols_per_iter
                // larger, and Bc needs to be >= cols_per_thread. 64 is large enough, 32 is not.
                return {get_fa_scalar_num_large_rows(hsk, hsv, small_cache), 64};
            } else {
                return {get_fa_scalar_num_large_rows(hsk, hsv, small_cache), 32};
            }
        }
    }

    if (path == FA_COOPMAT1) {
        if (small_rows) {
            return {scalar_flash_attention_num_small_rows, scalar_flash_attention_Bc};
        } else {
            return {coopmat1_flash_attention_num_large_rows, scalar_flash_attention_Bc};
        }
    }

    // small rows, large cols
    if (small_rows) {
        return {get_fa_num_small_rows(FA_COOPMAT2), 32};
    }

    // small cols to reduce register count
    if (ggml_is_quantized(type) || hsk >= 256 || hsv >= 256) {
        if (hsk >= 512 || hsv >= 512) {
            return {32, 32};
        } else {
            return {64, 32};
        }
    }
    return {64, 64};
}

static uint32_t fa_align(FaCodePath path, uint32_t hsk, uint32_t hsv, ggml_type type, bool small_rows, bool small_cache) {
    return fa_rows_cols(path, hsk, hsv, 0, type, small_rows, small_cache)[1];
}

static uint32_t fa_block_bytes(ggml_type t) {
    if (t == GGML_TYPE_F32) {
        return 16u;
    }
    return (uint32_t) ggml_type_size(t);
}

static bool ggml_vk_fa_type_needs_shmem(ggml_type type) {
    switch (type) {
    case GGML_TYPE_IQ4_NL:
        return true;
    default:
        return false;
    }
}

static bool ggml_vk_fa_scalar_uses_mmq(const vk_device& device, ggml_type k_type, ggml_type v_type) {
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    return device->integer_dot_product && device->subgroup_clustered &&
           !ggml_vk_fa_type_needs_shmem(v_type) &&
           (k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q4_1 ||
            k_type == GGML_TYPE_Q5_0 || k_type == GGML_TYPE_Q5_1 ||
            k_type == GGML_TYPE_Q8_0);
#else
    GGML_UNUSED(device);
    GGML_UNUSED(k_type);
    GGML_UNUSED(v_type);
    return false;
#endif
}

static std::vector<uint32_t> get_fa_spec_constants(const vk_fa_pipeline_state& state) {
    return {
        /* 0 WorkGroupSize   */ state.workgroup_size,
        /* 1 Br              */ state.Br,
        /* 2 Bc              */ state.Bc,
        /* 3 HSK             */ state.HSK,
        /* 4 HSV             */ state.HSV,
        /* 5 Clamp           */ static_cast<uint32_t>(!state.aligned),
        /* 6 D_split         */ state.D_split,
        /* 7 row_split       */ state.row_split,
        /* 8 SubGroupSize    */ state.subgroup_size,
        /* 9 SHMEM_STAGING   */ state.shmem_staging ? 1u : 0u,
        /*10 Flags           */ state.flags,
        /*11 LIMIT_OCCUPANCY_SHMEM */ state.limit_occupancy_shmem,
        /*12 FaTypeK         */ static_cast<uint32_t>(state.k_type),
        /*13 FaTypeV         */ static_cast<uint32_t>(state.v_type),
        /*14 FaBlockBytesK   */ fa_block_bytes(state.k_type),
        /*15 FaBlockBytesV   */ fa_block_bytes(state.v_type),
    };
}

static vk_fa_pipeline_state get_fa_pipeline_state(const vk_device& device, const vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc,
                                                  bool use_mask, bool use_mask_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type) {
    const bool old_amd_windows = device->vendor_id == VK_VENDOR_ID_AMD && device->driver_id == vk::DriverId::eAmdProprietary &&
                                 (device->architecture == AMD_GCN || device->architecture == AMD_RDNA1 || device->architecture == AMD_RDNA2);

    uint32_t flags = (use_mask_opt      ? 1 : 0) |
                     (use_mask          ? 2 : 0) |
                     (use_logit_softcap ? 4 : 0) |
                     (old_amd_windows   ? 8 : 0) |
                     (use_sparse        ? 16 : 0);

    const uint32_t subgroup_size = params.disable_subgroups ? 0 : params.subgroup_size;

    return vk_fa_pipeline_state{hsk, hsv, params.block_rows, params.block_cols, params.d_split, params.row_split, params.shmem_staging, params.path, params.workgroup_size, subgroup_size, aligned, f32acc, flags, params.limit_occupancy_shmem, k_type, v_type};
}

// scalar_f32: size for the _fp32 SPIR-V variant (f32 tiles, no coopmat staging)
// instead of the variant the device loads by default. Element size depends on
// the shader variant, not on the device -- ggmlR, for pipeline_matmul_f32_prec32.
static bool ggml_vk_matmul_shmem_support(const vk_device& device, const std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type, bool scalar_f32 = false) {

    uint32_t lut_size = 0;
    switch (src0_type) {
    case GGML_TYPE_IQ1_S:
    case GGML_TYPE_IQ1_M:
        // Regular matmul uses the compact uint16_t IQ1 grid; the expanded
        // uint32_t grid is only enabled for the q8_1/int-dot vector path.
        lut_size = 2*2048;
        break;
    case GGML_TYPE_IQ2_XXS:
        lut_size = 8*256;
        break;
    case GGML_TYPE_IQ2_XS:
        lut_size = 8*512;
        break;
    case GGML_TYPE_IQ2_S:
        lut_size = 8*1024;
        break;
    case GGML_TYPE_IQ3_XXS:
        lut_size = 4*256;
        break;
    case GGML_TYPE_IQ3_S:
        lut_size = 4*512;
        break;
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_MXFP4:
        lut_size = 4*16;
        break;
    case GGML_TYPE_NVFP4:
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128] (types.glsl, DATA_A_NVFP4).
        lut_size = 4*16 + 128u * (uint32_t)sizeof(float);
        break;
    default:
        break;
    }

    // Needs to be kept up to date on shader changes
    // Needs to stay aligned with ggml_vk_mul_mm_spec.
    const bool intel_shmem_stride_pad_zero = device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
                                              device->driver_id == vk::DriverId::eIntelProprietaryWindows;
    // ggmlR: scalar_f32 = the exact-f32-tile (prec32) variant, scalar tiles in f32.
    const bool coopmat_variant = device->coopmat_support && !scalar_f32;
    const uint32_t bank_conflict_offset = intel_shmem_stride_pad_zero && !scalar_f32 ? 0 : (coopmat_variant ? 8 : 1);
    const uint32_t type_size = (device->fp16 && !scalar_f32) ? sizeof(ggml_fp16_t) : sizeof(float);
    const uint32_t warps = warptile[0] / warptile[10];

    const uint32_t load_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_offset) * type_size;
    const uint32_t mmid_row_ids = mul_mat_id ? (warptile[2] * 2 * sizeof(uint16_t)) : 0;
    const uint32_t coopmat_stage = coopmat_variant ? warptile[7] * warptile[8] / warps * sizeof(float) : 0;

    const uint32_t total_size = load_bufs + mmid_row_ids + coopmat_stage + lut_size;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    return supported;
}

static bool ggml_vk_matmul_int_shmem_support(const vk_device& device, const std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type) {

    // FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float.
    const uint32_t fp_size   = device->fp16 ? 2u : 4u;
    const uint32_t fp_align  = fp_size;
    const uint32_t fp2_size  = 2u * fp_size;
    const uint32_t fp2_align = device->fp16 ? 4u : 8u;

    struct member { uint32_t size, align; };
    auto std430_size = [](std::initializer_list<member> members) {
        uint32_t off = 0, struct_align = 1;
        for (const auto &m : members) {
            off = (off + m.align - 1) & ~(m.align - 1);
            off += m.size;
            struct_align = std::max(struct_align, m.align);
        }
        return (off + struct_align - 1) & ~(struct_align - 1);
    };

    uint32_t block_a_size = 0;
    switch (src0_type) {
        case GGML_TYPE_Q2_0:    block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + dm
        case GGML_TYPE_Q4_0:    block_a_size = std430_size({{16, 4}, {fp_size,  fp_align}});                  break; // qs[16/4] + dm
        case GGML_TYPE_Q4_1:    block_a_size = std430_size({{16, 4}, {fp2_size, fp2_align}});                 break; // qs[16/4] + dm(vec2)
        case GGML_TYPE_Q5_0:    block_a_size = std430_size({{16, 4}, {4, 4}, {fp_size,  fp_align}});          break; // qs[16/4] + qh + dm
        case GGML_TYPE_Q5_1:    block_a_size = std430_size({{16, 4}, {4, 4}, {fp2_size, fp2_align}});         break; // qs[16/4] + qh + dm(vec2)
        case GGML_TYPE_Q8_0:    block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + dm
        case GGML_TYPE_IQ4_XS:  block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + d
        case GGML_TYPE_MXFP4:   block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + d
        case GGML_TYPE_IQ4_NL:  block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + d
        case GGML_TYPE_NVFP4:   block_a_size = std430_size({{32, 4}, {fp2_size, fp2_align}});                 break; // qs[8] + d_scales(vec2)
        case GGML_TYPE_Q2_K:    block_a_size = std430_size({{ 8, 4}, {2, 2}, {fp2_size, fp2_align}});         break; // qs[2] + scales(u8vec2) + dm(vec2)
        case GGML_TYPE_Q3_K:    block_a_size = std430_size({{16, 4}, {fp2_size, fp2_align}});                 break; // qs[4] + d_scales(vec2)
        case GGML_TYPE_Q4_K:    block_a_size = std430_size({{16, 4}, {fp2_size, fp2_align}});                 break; // qs[4] + dm(vec2)
        case GGML_TYPE_Q5_K:    block_a_size = std430_size({{32, 4}, {fp2_size, fp2_align}});                 break; // qs[8] + dm(vec2)
        case GGML_TYPE_Q6_K:    block_a_size = std430_size({{32, 4}, {fp2_size, fp2_align}});                 break; // qs[8] + d_scales(vec2)
        case GGML_TYPE_IQ3_S:   block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + d
        default:
            return false;
    }

    // IQ3_S also copies its 512-entry grid into shared memory (types.glsl, init_iq_shmem)
    const uint32_t lut_size = (src0_type == GGML_TYPE_IQ3_S) ? 4*512 : 0;

    // block_b_cache: { int32_t qs[8]; FLOAT_TYPEV2 ds; }
    const uint32_t block_b_size = std430_size({{32, 4}, {fp2_size, fp2_align}});

    const uint32_t BM = warptile[1];
    const uint32_t BN = warptile[2];
    // mul_mmq.comp: BK_STEP=1 for MUL_MAT_ID, 4 otherwise.
    const uint32_t BK_STEP = mul_mat_id ? 1u : 4u;

    const uint32_t buf_a_size = BM * BK_STEP * block_a_size;
    const uint32_t buf_b_size = BN * BK_STEP * block_b_size;
    const uint32_t mmid_row_ids = mul_mat_id ? (BN * 2u * (uint32_t)sizeof(uint16_t)) : 0u;

    const uint32_t total_size = buf_a_size + buf_b_size + mmid_row_ids + lut_size;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_matmul_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << total_size << ", supported=" << supported);

    return supported;
}

static bool ggml_vk_matmul_cm1_int_shmem_support(const vk_device& device, const std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type) {

    bool kscales2 = false;    // two scale sets per block
    bool has_dm   = false;    // d+m as vec2 + b-side sum
    bool has_kvalues = false;
    switch (src0_type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q5_0: case GGML_TYPE_Q8_0:
            break;
        case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
            has_dm = true;                          break;
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: case GGML_TYPE_MXFP4:
            has_kvalues = true;                     break;
        case GGML_TYPE_Q3_K: case GGML_TYPE_Q6_K:
            kscales2 = true;                        break;
        case GGML_TYPE_NVFP4:
            kscales2 = true; has_kvalues = true;    break;
        default:
            return false;
    }

    const uint32_t BM         = warptile[1];
    const uint32_t BN         = warptile[2];

    const uint32_t BK      = 32;
    const uint32_t BK_STEP = mul_mat_id ? 2u : 4u;
    const uint32_t QPITCH  = BK_STEP * (BK / 4u) + 4u;
    const uint32_t KSCALES = kscales2 ? 2u : 1u;

    uint32_t total = 0;
    total += BM * QPITCH * (uint32_t)sizeof(uint32_t);   // buf_a_qs
    total += BN * QPITCH * (uint32_t)sizeof(uint32_t);   // buf_b_qs
    total += has_dm ? (BM * BK_STEP * 2u * (uint32_t)sizeof(float))   // buf_a_dm (vec2)
                    : (BM * BK_STEP * KSCALES * (uint32_t)sizeof(float)); // buf_a_d
    total += BN * BK_STEP * (uint32_t)sizeof(float);     // buf_b_d
    if (has_dm) {
        total += BN * BK_STEP * (uint32_t)sizeof(float); // buf_b_s
    }
    if (has_kvalues) {
        total += 16u * (uint32_t)sizeof(int8_t);         // cm1_kvalues[16]
    }
    if (src0_type == GGML_TYPE_NVFP4 && !device->ocp_fp4) {
        total += 128u * (uint32_t)sizeof(float);         // ue4m3_fp32_lut[128]
    }
    if (mul_mat_id) {
        total += BN * 2u * (uint32_t)sizeof(uint16_t);   // row_ids[BN] (u16vec2)
    }

    const bool supported = total <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_matmul_cm1_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << total << ", supported=" << supported);

    return supported;
}

struct GpuPipelineConfig {
    // GPU architecture identifier.
    // Example: vk_device_architecture::AMD_GCN
    vk_device_architecture arch;

    // Mapping of pipeline names to their specific subgroup sizes.
    // Example: {"soft_max_f32", 64}
    std::unordered_map<std::string, uint32_t> pipelines;

    // Default subgroup size for this GPU.
    // Defaults to 0 if not explicitly provided.
    uint32_t default_subgroup_size = 0;
};

// Pipeline configuration for RDNA1 GPUs.
static const std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

// Pipeline configuration for RDNA2 GPUs.
static const std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

static constexpr uint32_t RDNA_DEFAULT_SUBGROUP_SIZE = 32;

// Define configurations for different GPUs.
static std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_SIZE
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_SIZE
    },
};

static uint32_t get_subgroup_size(const std::string &pipeline_name, const vk_device_architecture &arch) {
    for (const auto &config : gpu_pipeline_configs) {
        if (config.arch == arch) {
            auto pipIt = config.pipelines.find(pipeline_name);
            if (pipIt != config.pipelines.end()) {
                return pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });
            for (const auto &entry : sorted_pipelines) {
                if (pipeline_name.find(entry.first) != std::string::npos) {
                    return entry.second;
                }
            }
            return config.default_subgroup_size;
        }
    }
    return 0; // If no matching configuration is found
}

