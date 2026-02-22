## Resubmission

Changes since previous submission:
- Removed non-portable SIMD compilation flags (-mavx2, -msse4.2, etc.)
  that triggered NOTE on CRAN. SIMD is now opt-in via
  `configure.args = "--with-simd"`.
- Fixed misspelled words in DESCRIPTION ('Vulkan', 'SDK', 'autograd').

## R CMD check results

0 errors | 0 warnings | 4 notes

* **installed package size**: The package includes a static library (lib/)
  for downstream packages (llamaR, sdR) and a shared library (libs/) with
  the GGML tensor computation engine. The size is inherent to the compiled
  C/C++ codebase.

* **checking for future file timestamps**: Unable to verify current time.
  This is a transient network/NTP issue on the check machine, not a
  package problem.

* **GNU make is a SystemRequirements**: GNU make is listed in
  SystemRequirements and is needed for the build.

* **compilation flags used**: `-mno-omit-leaf-frame-pointer` is added by
  the system compiler/R configuration, not by the package itself.

## Test environments

* Windows Server 2022 x64 (win-builder), R-devel, GCC 14.3.0
* Local: Linux Mint 22.3, R 4.3.3, GCC 13.3.0

## Downstream dependencies

This is a new package with no downstream dependencies.
