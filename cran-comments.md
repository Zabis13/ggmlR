## Resubmission (0.7.0)

This version addresses the gcc-UBSAN issue flagged for the last
released version on CRAN. The undefined behaviour has been fixed.

Other changes in 0.7.0:

- Vignettes switched to prebuilt HTML via R.rsp::asis engine
  (no rendering on CRAN runners).
- Removed rmarkdown from Suggests.
- Fixed misspelled words in DESCRIPTION (cpp, pre, dequantization).
- Suppressed spurious test output (C-level warnings captured in tests).

## R CMD check results

0 errors | 0 warnings | 4 notes

* **installed package size**: The package includes a static library (lib/)
  for downstream packages (llamaR, sd2R) and a shared library (libs/) with
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
* Debian Linux (win-builder), R-devel
* Local: Linux Mint 22.3, R 4.3.3, GCC 13.3.0

## Downstream dependencies

* sd2R — checked, no issues.
