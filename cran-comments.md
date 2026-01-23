## R CMD check results

0 errors | 0 warnings | 6 notes

Note: Local check shows 6 NOTEs, but NOTE #6 (non-portable compilation flag '-mno-omit-leaf-frame-pointer') is added by the local R configuration and will not appear on CRAN build systems.

### NOTEs

**1. New submission**

This is a new package submission.

**2. Installed size is 6.0Mb**

This package provides R bindings to the GGML tensor library (<https://github.com/ggml-org/ggml>), which requires substantial compiled code. The size is unavoidable for this type of package.

**3. GNU make is a SystemRequirements**

GNU make is correctly listed in SystemRequirements field.

**4. Files which contain pragma(s) suppressing diagnostics**

Upstream ggml library (https://github.com/ggerganov/ggml):
- ggml-cpu/arch/x86/mmq.cpp: AMX intrinsics (Intel Advanced Matrix Extensions for Sapphire Rapids CPUs)
- ggml-cpu/arch/x86/repack.cpp: AVX512 intrinsics (cross-platform SIMD packing optimizations)
- ggml-cpu/llamafile/sgemm.cpp: Llama.cpp matrix multiplication (production-tested optimizations)

Pragmas are required for:
- Cross-compiler compatibility (supports both clang-21 and gcc-14)
- Suppressing architecture-specific intrinsics warnings on unsupported platforms
- Maintaining compatibility with upstream: removing pragmas would require patching upstream code and complicate future updates

These files are unmodified from the upstream library to ensure stability and easy maintenance.

**5. C++17 specification**

C++17 is required by the upstream GGML library. SystemRequirements correctly lists this dependency.

## Test environments

* Local: Linux Mint 22.3, R 4.3.3, GCC 13.3.0

## Downstream dependencies

This is a new package with no downstream dependencies.
