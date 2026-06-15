## Submission (0.7.9)

This version fixes a platform-specific silent crash on Windows/MinGW and
preserves reverse-dependency compatibility.

### Windows/MinGW silent-crash fix

On Windows builds using the MinGW toolchain, GPU model loading could abort
the R process with no error message during buffer allocation (observed when
loading FLUX.1/FLUX.2 diffusion models through the Vulkan or meta backend).

Root cause: the internal function `ggml_backend_buffer_init()` is declared
`extern "C"` and took its `ggml_backend_buffer_i` interface argument **by
value**. That struct holds 11 function pointers (~88 bytes). On MinGW,
passing such a large POD by value across an `extern "C"` translation-unit
boundary mis-marshals the argument between caller and callee, terminating
the process at the call instruction (via `std::terminate()` -> `abort()`).
The failure was silent because, under the R build, stdio is redirected to
non-flushing `REprintf` wrappers, so any buffered diagnostic is lost on
`abort()`. The same defect was not observed on Linux/GCC, where the by-value
ABI for this struct is consistent across TUs.

Fix: the parameter is now passed by pointer
(`const ggml_backend_buffer_i *`). All in-tree call sites were updated
accordingly: the CPU, multi-buffer, Vulkan, meta and AMX backends. Behaviour
is otherwise unchanged (the function still constructs the same
`ggml_backend_buffer`).

`ggml_backend_buffer_init()` is an internal backend function declared only
in `ggml-backend-impl.h`, not in the public `ggml-backend.h`. The reverse
dependencies `llamaR` and `sd2R` do not call it directly, so their sources
compile unchanged against this release; they pick up the fix automatically
when rebuilt against the updated static library.

## Submission (0.7.7)

This version preserves reverse-dependency compatibility. The
`GGML_BACKEND_DEVICE_TYPE_META` enumerator (added upstream to
`ggml_backend_dev_type`) is temporarily withheld from the public enum in
the installed `ggml-backend.h`. This keeps the published `llamaR`, whose
`switch` over `ggml_backend_dev_type` does not yet handle that value,
compiling without a `-Wswitch` warning. The meta backend itself remains
functional (the device is identified by its interface pointer, not by the
enum value). The enumerator will be restored in a later release once a
`llamaR` update carrying a `default:` branch is on CRAN.

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

* llamaR — checked; compiles without the `-Wswitch` warning previously
  triggered by the new device-type enumerator (see Submission note above).
* sd2R — checked, no issues.
