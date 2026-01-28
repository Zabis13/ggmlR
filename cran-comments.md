## R CMD check results

0 errors | 0 warnings | 5 notes

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

**5. SHLIB_OPENMP_CFLAGS is included in PKG_CFLAGS but not in PKG_LIBS**

This package contains mixed C and C++ code. The upstream GGML CPU backend
(ggml-cpu-backend.c) uses OpenMP for parallel tensor computation in C code.
Since the package also contains C++ files, R links via the C++ compiler,
requiring SHLIB_OPENMP_CXXFLAGS in PKG_LIBS. The configuration is:

- PKG_CFLAGS = $(SHLIB_OPENMP_CFLAGS) — enables OpenMP in C compilation
- PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS) — matching macro for C++
- PKG_LIBS = $(SHLIB_OPENMP_CXXFLAGS) — matches the C++ linker

On all CRAN platforms (Linux gcc/g++, Windows Rtools gcc/g++), both macros
expand to the same value (-fopenmp), so OpenMP is correctly linked.
On macOS (Apple clang), both macros are empty, and the code compiles
without OpenMP using built-in pthreads fallback (#ifndef GGML_USE_OPENMP).

As noted in Writing R Extensions sect 1.2.1.1: "It is not portable to use
OpenMP with more than one of C, C++ and Fortran in a single package."
This NOTE is expected and unavoidable for our mixed-language codebase.

## Test environments

* Local: Linux Mint 22.3, R 4.3.3, GCC 13.3.0

## Downstream dependencies

This is a new package with no downstream dependencies.
