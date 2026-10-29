// ggmlR: fallback for the SPIR-V constants used by the pipeline-creation
// SPIR-V patches (float controls, decode-vector strip, Asahi BK-loop roll).
//
// Upstream ggml requires SPIRV-Headers (spirv/unified1/spirv.hpp). That header
// ships with the Vulkan SDK but not with a plain distro libvulkan-dev, so when
// it is not found we define just the constants we use. Values are fixed by the
// SPIR-V specification and match spirv.hpp.

#pragma once

#include <cstdint>

namespace spv {

static constexpr uint32_t OpCodeMask     = 0xffff;
static constexpr uint32_t WordCountShift = 16;

static constexpr uint32_t OpExtension       = 10;
static constexpr uint32_t OpEntryPoint      = 15;
static constexpr uint32_t OpExecutionMode   = 16;
static constexpr uint32_t OpCapability      = 17;
static constexpr uint32_t OpLoopMerge       = 246;
static constexpr uint32_t OpLabel           = 248;
static constexpr uint32_t OpExecutionModeId = 331;

static constexpr uint32_t ExecutionModeDenormPreserve  = 4459;
static constexpr uint32_t ExecutionModeRoundingModeRTE = 4462;

static constexpr uint32_t CapabilityDenormPreserve  = 4464;
static constexpr uint32_t CapabilityRoundingModeRTE = 4467;

static constexpr uint32_t LoopControlMaskNone       = 0x0;
static constexpr uint32_t LoopControlUnrollMask     = 0x1;
static constexpr uint32_t LoopControlDontUnrollMask = 0x2;

} // namespace spv
