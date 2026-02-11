## Reason for quick resubmission

This is a critical bugfix release addressing:
- Installation failures on macOS ARM64 (M1/M2/M3) platforms
- Missing symbol errors preventing package loading
- CRAN check failures on 3 of 8 platforms

Previous version 0.5.1 had ERROR status on M1 Mac checks.

## R CMD check results

0 errors | 0 warnings | 3 notes

* **installed package size**: The package includes a static library (lib/)
  for downstream packages (llamaR, sdR) and a shared library (libs/) with
  the GGML tensor computation engine. The size is inherent to the compiled
  C/C++ codebase.

* **GNU make is a SystemRequirements**: GNU make is listed in
  SystemRequirements and is needed for the build.

* **Non-portable compilation flags** (-msse3, -mssse3, -msse4.1, -msse4.2,
  -mavx, -mavx2, -mfma, -mf16c): These x86 SIMD flags are detected and
  added only on x86_64 platforms by the configure script. On ARM and other
  architectures no SIMD flags are added. The flags are essential for
  performant tensor computation — without them, inference speed drops
  significantly.

## Test environments

* Windows Server 2022 x64 (win-builder), R-devel, GCC 14.3.0
* Local: Linux Mint 22.3, R 4.3.3, GCC 13.3.0

## Downstream dependencies

This is a new package with no downstream dependencies.
